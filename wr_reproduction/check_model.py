"""Original model + first original logical-rank batch, no optimizer updates.

This integration check is not an eight-rank optimizer step or convergence run.
All model dimensions, step-zero loss terms and data packing come from the record.
"""
import argparse,json,time
from unittest.mock import patch
import torch
from wr_reproduction.backend import install

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='pr393',choices=['pr393','record360'])
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(393)
    install(args.version)
    from track_1_short.model.gpt import GPT,ForwardScheduleConfig
    from track_1_short.config import Hyperparameters,TRAINING_STAGES,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT,BLOCK_SIZE,VIRTUAL_SEQ_CAP
    from track_1_short.schedule import TrainingSchedule
    from track_1_short.data import distributed_data_generator,cu_seqlens_rows
    from track_1_short.perf.pinned_batches import PinnedBatchStaging
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    cfg=Hyperparameters();dev=torch.device('cuda');stage=TRAINING_STAGES[0]
    schedule=TrainingSchedule(TRAINING_STAGES,cfg.num_scheduled_iterations,cfg.num_extension_iterations,
        dev,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
    print('constructing original model',flush=True)
    kwargs=dict(ngram_dim=768,world_size=8,device=dev)
    if args.version=='pr393':kwargs['rank']=0
    model=GPT(50257,11,6,128,768,cfg.val_batch_size//8,**kwargs).cuda()
    model.cast_matrix_weights_bf16();model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True)
    if args.version=='pr393':model.quantize_moe_full()
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size,0,1))
    local=stage.batch_size//8
    staging=PinnedBatchStaging(local,cu_seqlens_rows(local),dev)
    with patch('torch.distributed.get_rank',return_value=0),patch('torch.distributed.get_world_size',return_value=8):
        batch=next(distributed_data_generator(cfg.train_files,stage.batch_size,stage.train_max_seq_len,staging))
    # At initialization the original full ngram table and row cache are all zero.
    _,slots=torch.unique(batch.ngram_ids,sorted=True,return_inverse=True)
    from track_1_short.sampled_softmax import SampledSoftmax
    sampled=SampledSoftmax(schedule,model.vocab_size,768,local,0,8,dev)
    sampled.set_prefix_table(model.prefix_table.cpu().numpy())
    sampled.upload(0,batch.targets_cpu)
    sampled_loss=sampled.gather(0,model._lm_head_f8_col)
    fcfg=ForwardScheduleConfig(schedule.mtp_weights[0],schedule.prefix_weights[0],
        stage.window_sizes[0]*BLOCK_SIZE,stage.window_sizes[1]*BLOCK_SIZE,min(stage.train_max_seq_len,VIRTUAL_SEQ_CAP),sampled_loss)
    sink=torch.zeros(2*local,768,device=dev,dtype=torch.bfloat16,requires_grad=True)
    ve=torch.zeros_like(model.value_embeds,dtype=torch.float16)
    print('original first-rank forward/backward',flush=True);torch.cuda.synchronize();start=time.perf_counter()
    compiled=torch.compile(model,dynamic=False,fullgraph=True)
    loss=compiled(batch.inputs,batch.targets,batch.cum_seqlens,slots.to(torch.int32),fcfg,ngram_sink=sink,value_embed_grad=ve)
    print('forward loss',loss.float().mean().item(),flush=True)
    loss.sum().backward();torch.cuda.synchronize()
    grads=[p.grad for p in model.parameters() if p.grad is not None]+[sink.grad,ve]
    if args.version=='pr393':grads.append(model.moe_bank_grad)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    print(json.dumps(dict(version=args.version,passed=True,logical_rank_tokens=local,global_update_tokens=stage.batch_size,
        loss=loss.float().mean().item(),cold_forward_backward_seconds=time.perf_counter()-start,
        optimizer_updates=0,finite_gradient_tensors=len(grads),peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)

if __name__=='__main__':main()
