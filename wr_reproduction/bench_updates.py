"""Short original-record update timing with eight serialized rank microbatches.

This executes the original model, data, sampled objectives and optimizers. It is
an update-prefix verification, not a full convergence/validation reproduction.
The original full schedule stays intact; --updates only ends the diagnostic.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from wr_reproduction.backend import install
from wr_reproduction.local_collectives import LocalCollectives
from wr_reproduction.rank_state import RankStates
from wr_reproduction.virtual_tables import VirtualTables
from wr_reproduction.accumulation import OriginalGradientAccumulator

class Sinks:
    def __init__(self,device):self.device=device;self.sinks={}
    def grad_sink(self,n):
        if n not in self.sinks:self.sinks[n]=torch.zeros(2*n,768,device=self.device,dtype=torch.bfloat16,requires_grad=True)
        return self.sinks[n]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['pr393','record360'],default='pr393')
    parser.add_argument('--updates',type=int,default=3);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not 2<=args.updates<=8:raise ValueError('This diagnostic is restricted to 2–8 original updates')
    torch.set_num_threads(8);install(args.version)
    from track_1_short.model.gpt import GPT,FP8_EXACT_SCALE_CALLS
    from track_1_short.config import Hyperparameters,TRAINING_STAGES,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT
    from track_1_short.schedule import TrainingSchedule
    from track_1_short.training import TrainingManager
    from track_1_short.data import distributed_data_generator,cu_seqlens_rows,ScheduledBatches
    from track_1_short.perf.pinned_batches import PinnedBatchStaging
    from track_1_short.sampled_softmax import SampledSoftmax
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from track_1_short.perf.cuda_graphs.capture_plan import step_graph_key
    from track_1_short.perf.cuda_graphs.step_graphs import StepGraphs
    from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs
    from track_1_short.optim.anvil import anvil_bank_update
    cfg=Hyperparameters();dev=torch.device('cuda');stage=TRAINING_STAGES[0]
    if cfg.train_seed is not None:torch.manual_seed(cfg.train_seed)
    schedule=TrainingSchedule(TRAINING_STAGES,cfg.num_scheduled_iterations,cfg.num_extension_iterations,
        dev,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
    kwargs=dict(ngram_dim=768,world_size=8,device=dev)
    if args.version=='pr393':kwargs['rank']=0
    model=GPT(50257,11,6,128,768,cfg.val_batch_size//8,**kwargs).cuda();model.cast_matrix_weights_bf16()
    model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True)
    if args.version=='pr393':model.quantize_moe_full()
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size,0,1))
    if args.version=='pr393':
        from track_1_short.perf.kernels import sampled_head
        sampled_head.init(dev,rank=0)
    def publish_moe():
        model._quantize_moe(0,model._moe_scales.numel(),exact=model.mlp_quantize_calls<FP8_EXACT_SCALE_CALLS)
        model.build_moe_cols()
        return []
    with LocalCollectives():
        optimizer_graphs=AnvilBankGraphs(dev)
        tm=TrainingManager(model,schedule,bank_update=optimizer_graphs.update,
            **({'moe_publish':publish_moe} if args.version=='pr393' else {}))
        tm.advance_schedule(0)
        # All data and sampled-softmax builders retain their original eight ranks.
        batches=[];samplers=[];local=stage.batch_size//8
        for rank in range(8):
            staging=PinnedBatchStaging(local,cu_seqlens_rows(local),dev)
            generator=distributed_data_generator(cfg.train_files,stage.batch_size,stage.train_max_seq_len,staging)
            loader=ScheduledBatches(generator,schedule,range(args.updates))
            with patch('torch.distributed.get_rank',return_value=rank),patch('torch.distributed.get_world_size',return_value=8):loader.peek(0)
            batches.append(loader)
            sampler=SampledSoftmax(schedule,model.vocab_size,768,local,rank,8,dev)
            sampler.set_prefix_table(model.prefix_table.cpu().numpy())
            if samplers:
                sampler.loss_inputs=samplers[0].loss_inputs
                sampler.candidates=samplers[0].candidates
                sampler.staging=samplers[0].staging  # one reader event for the shared device buffers
            samplers.append(sampler)
        compiled=torch.compile(model,dynamic=False,fullgraph=True)
        shared_value_grad=torch.zeros_like(model.value_embeds,dtype=torch.float16)
        keys=[step_graph_key(schedule,samplers[0].counts,s,8) for s in range(args.updates)]
        graphs=StepGraphs(compiled,tm.optimizer,keys,Sinks(dev),SimpleNamespace(grad_accum=shared_value_grad),local,dev)
        first=batches[0].peek(0)
        _,slots=torch.unique(first.ngram_ids,sorted=True,return_inverse=True)
        print('Capturing original step graphs and checking replay',flush=True)
        for _ in range(2):
            samplers[0].upload(0,first.targets_cpu)
            sampled=samplers[0].gather(0,model._lm_head_f8_col)
            model.zero_grad(set_to_none=True)
            graphs.forward(0,first,slots.int(),tm.get_forward_args(sampled))
            graphs.backward();samplers[0].mark_readers_done()
        graphs.seal()
        # Optimizer capture snapshots/restores parameters, rails and mantissas itself.
        for bank in tm.optimizer.banks.values():bank.grad.zero_()
        print('Capturing original optimizer graphs',flush=True)
        optimizer_graphs.capture(list(tm.optimizer.banks.values()))
        # Warmup above made no optimizer steps. Restore rank-local side effects.
        model.zero_grad(set_to_none=True);shared_value_grad.zero_()
        model.rearm_fp8_bootstrap();model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True)
        if args.version=='pr393':model.quantize_moe_full()
        for sampler in samplers:sampler.reset()
        rank_states=RankStates(model)
        print('Allocating complete original n-gram table and rank-local cycle accumulators',flush=True)
        tables=VirtualTables(model,tm.optimizer,schedule.total_steps,args.version)
        counter_states=[torch.tensor(r,device=dev,dtype=torch.int64) for r in range(8)]
        report=dict(version=args.version,total_scheduled_steps=schedule.total_steps,seed=cfg.train_seed,
            logical_ranks=8,physical_optimizer_owners=1,training_gemm_precision='original FP8 formats/scales',
            diagnostic_only=True,validation_performed=False,updates=[])
        from track_1_short.perf.kernels.transpose import transpose_add
        gradients=OriginalGradientAccumulator(p for p,c in tm.optimizer.param_cfgs.items() if c.optim=='adam')
        for step in range(args.updates):
            tm.advance_schedule(step);torch.cuda.synchronize();start=time.perf_counter()
            gradients.begin(tm.is_adam_step(step));losses=[]
            for rank in range(8):
                rank_states.enter(rank)
                if step:
                    if rank==0:
                        model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=(step-1)%2==1)
                    else:rank_states.refresh_activations()
                if args.version=='pr393':sampled_head.COUNTER.copy_(counter_states[rank])
                shared_value_grad.copy_(tables.value_grads[rank])
                batch=batches[rank].peek(step)
                slots=tables.load(batch.ngram_ids)
                sampler=samplers[rank];sampler.upload(step,batch.targets_cpu)
                sampled=sampler.gather(step,model._lm_head_f8_col)
                model.zero_grad(set_to_none=True)
                graphs.forward(step,batch,slots,tm.get_forward_args(sampled))
                ngram_gradient=graphs.backward()
                # Graph buffers are reused by the next rank, so retain this cycle contribution.
                tables.hold(rank,batch.ngram_ids,ngram_gradient.clone())
                tables.value_grads[rank].copy_(shared_value_grad)
                gradients.add(rank,{p:p.grad for p in tm.optimizer.param_cfgs if p.grad is not None},
                    tied_pair=None if tm.optimizer.split_embed else (model.lm_head.weight,model.embed.weight),
                    transpose_add=transpose_add)
                losses.append(graphs.current.loss.detach().clone()/batch.inputs.numel())
                rank_states.leave(rank)
                if args.version=='pr393':counter_states[rank].copy_(sampled_head.COUNTER)
                sampler.mark_readers_done();batches[rank].take(step)
            model.zero_grad(set_to_none=True)
            for p,g in gradients.finish().items():p.grad=g
            tm.step_optimizers(step,tables,frozenset())
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            item=dict(step=step+1,global_tokens=stage.batch_size,microbatch_tokens=local,microbatches=8,
                seconds=seconds,mean_training_objective=torch.stack(losses).mean().item(),
                table_events=tables.event,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            report['updates'].append(item);print(json.dumps(item),flush=True)
            args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
        optimizer_graphs.seal()
        report['complete']=True;args.out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
