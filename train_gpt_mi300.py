"""Single-MI300X BF16 capacity port of track-1 record #360 (4ea6b937337a).

Launch ./run_mi300.sh. Use --smoke for two real-data training updates and a small eval.
The original H100 trainer/package are untouched. See MI300.md for explicit differences.
"""
import argparse
import os
import json
import hashlib
import time
from pathlib import Path
import torch
import torch.distributed as dist
from mi300.config import Hyperparameters, TRAINING_STAGES, LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT
from mi300.data import ScheduledBatches, cu_seqlens_rows, distributed_data_generator
from mi300.distributed import setup_distributed
from mi300.model.gpt import GPT
from mi300.model.prefix_prediction import build_prefix_table_bucket
from mi300.ngram_table import NgramTable, NGRAM_DIM, NGRAM_VOCAB_SIZE, MAX_CYCLE_STEPS, is_update_step
from mi300.perf.pinned_batches import PinnedBatchStaging
from mi300.perf.row_prefetch import PrefetchStaging, RowPrefetch
from mi300.perf.value_embed_pull import ValueEmbedPull
from mi300.optim.anvil import anvil_bank_update
from mi300.sampled_softmax import SampledSoftmax
from mi300.schedule import TrainingSchedule
from mi300.training import TrainingManager
from mi300.tail_average import TailAverages
from mi300.canonical_mask import build_canonical_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Two full-sized stage-0 steps, then 1024-token eval; no checkpoint')
    parser.add_argument('--smoke-stages', action='store_true', help='32 updates across schedule transitions plus full-sized validation batch; no checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true', help='Save complete inference weights including n-gram table after full run')
    options = parser.parse_args()
    torch.set_num_threads(8)
    env = setup_distributed()
    torch.manual_seed(options.seed)
    args = Hyperparameters()
    schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
                                env.device, LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT)
    if options.smoke_stages and schedule.total_steps != 1194:
        parser.error('--smoke-stages requires the default 1194-step schedule')
    if options.smoke and options.smoke_stages:
        parser.error('Choose only one smoke mode')
    if options.smoke_stages:
        # Contiguous complete sparse cycles around each distinct path: stage changes,
        # period-4 sparse updates, twin rails, sampled-softmax growth, and embedding split.
        steps = [0, 1, 320, 321]
        for start in (336, 512, 680, 964, 1104, 1172, 1176):
            steps.extend(range(start, start+4))
    elif options.smoke:
        steps = [0, 1]
    else:
        steps = list(range(schedule.total_steps))
    smoke = options.smoke or options.smoke_stages
    max_tokens = max(stage.batch_size for stage in TRAINING_STAGES)
    capacity = max(args.val_batch_size, MAX_CYCLE_STEPS*max_tokens)
    staging = PinnedBatchStaging(capacity, cu_seqlens_rows(capacity), env.device)
    model = GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
                max_seq_len=capacity, ngram_dim=NGRAM_DIM, world_size=1, device=env.device).to(env.device)
    model.cast_matrix_weights_bf16()
    table = NgramTable(model.ngram_cache, max_tokens, schedule.total_steps, 0, 1, env.device)
    manager = TrainingManager(model, schedule, bank_update=anvil_bank_update)
    ve = ValueEmbedPull(model.value_embeds, model.vocab_size, manager.optimizer, rank=0, world_size=1, device=env.device)
    prefetch_staging = PrefetchStaging(2*MAX_CYCLE_STEPS*max_tokens, model.value_embeds.shape[0], env.device)
    sampled = SampledSoftmax(schedule, model.vocab_size, 768, max_tokens, 0, 1, env.device)
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size, bucket=0, num_buckets=1))
    sampled.set_prefix_table(model.prefix_table.cpu().numpy())
    loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size,
                                         TRAINING_STAGES[0].train_max_seq_len, staging)
    batches = ScheduledBatches(loader, schedule, steps)
    update = is_update_step
    pulls = RowPrefetch(table, ve, batches, steps, update, prefetch_staging)
    tails = None if options.smoke else TailAverages(manager.optimizer, 0, is_update_step, schedule.total_steps)
    run_id = time.strftime('mi300-%Y%m%d-%H%M%S') + ('-smoke' if smoke else '')
    logdir = Path('logs')/run_id
    logdir.mkdir(parents=True)
    metadata = dict(base_commit='4ea6b937337a4889b8cfe3f38a93d120048d8f71', gpu=torch.cuda.get_device_name(),
                    gpu_arch=torch.cuda.get_device_properties(0).gcnArchName,
                    vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                    torch=torch.__version__, hip=torch.version.hip, seed=options.seed, steps=steps,
                    ngram_rows=NGRAM_VOCAB_SIZE, batch_tokens=[s.batch_size for s in TRAINING_STAGES],
                    boundaries=schedule.boundaries, precision='bf16', smoke=smoke)
    metadata['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), *sorted(Path('mi300').rglob('*.py'))]}
    (logdir/'config.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('steps', 'source_sha256')}), flush=True)
    parameters = list(manager.optimizer.param_cfgs)
    log = (logdir/'metrics.jsonl').open('w', buffering=1)
    started = time.perf_counter()
    try:
        for step in steps:
            manager.advance_schedule(step)
            batch = batches.peek(step)
            slots = pulls.forward_slots(step, batch.ngram_ids)
            ve.complete_land()
            sampled.upload(step, batch.targets_cpu)
            sample = sampled.gather(step, model.lm_head.weight)
            sink = table.grad_sink(batch.inputs.numel())
            loss = model(batch.inputs, batch.targets, batch.cum_seqlens, slots, manager.get_forward_args(sample),
                         ngram_sink=sink, value_embed_grad=ve.grad_accum).sum()
            pulls.after_forward(step)
            grads = torch.autograd.grad(loss, parameters+[sink], allow_unused=True)
            for param, grad in zip(parameters, grads[:-1]):
                if grad is None:
                    continue  # value embeddings accumulate into ve.grad_accum in their custom backward
                if manager.optimizer.param_cfgs[param].optim == 'anvil' or param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
            table.accumulate_grad(slots, grads[-1])
            sampled.mark_readers_done()
            pulls.post_prep(step)
            manager.step_optimizers(step, pulls, frozenset())
            ve.complete_land()
            batches.take(step)
            if tails is not None:
                tails.tick(step)
            # Synchronize and check actual loss/gradients on every smoke update.
            if smoke or (step+1)%25 == 0 or step == steps[-1]:
                value = float(loss.detach())/batch.inputs.numel()
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                if smoke:
                    for param, grad in zip(parameters, grads[:-1]):
                        if grad is not None and not torch.isfinite(grad).all().item():
                            raise RuntimeError(f'Nonfinite gradient: {param.label} at step {step}')
                    if not torch.isfinite(grads[-1]).all().item():
                        raise RuntimeError(f'Nonfinite n-gram gradient at step {step}')
                    for param in parameters:
                        if not torch.isfinite(param).all().item():
                            raise RuntimeError(f'Nonfinite parameter: {param.label} at step {step}')
                metric = dict(step=step, loss=value, seconds=time.perf_counter()-started,
                              peak_gib=torch.cuda.max_memory_allocated()/2**30)
                print(json.dumps(metric), flush=True)
                log.write(json.dumps(metric)+'\n')
            del loss, grads, sink
        if options.smoke_stages:
            assert manager.optimizer.split_embed, "Embedding did not untie"
            assert not table.pending, "Sparse cycle did not flush"
        pulls.close()
        ve.gather_replica()
        if tails is not None:
            tails.ship()
            manager.apply_final_ws_ext()
        # Use the same canonical mask for both full and smoke validation.
        model.canon_mask.copy_(torch.from_numpy(build_canonical_mask(model.vocab_size)))
        model.complete_yarn_tables()
        model.eval()
        val_batch = 1024 if options.smoke else args.val_batch_size
        val_tokens = val_batch if smoke else args.val_tokens
        val_loader = distributed_data_generator(args.val_files, val_batch, -1, staging, align_to_bos=False)
        total = 0.0
        with torch.no_grad():
            for _ in range(val_tokens//val_batch):
                batch = next(val_loader)
                slots = table.load_eval_batch(batch.ngram_ids)
                value = model(batch.inputs, batch.targets, batch.cum_seqlens, slots, manager.get_forward_args()).sum()
                if not torch.isfinite(value).item():
                    raise RuntimeError('Nonfinite validation loss')
                total += float(value)
        result = dict(val_loss=total/val_tokens, val_tokens=val_tokens, smoke=smoke,
                      peak_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.perf_counter()-started)
        print(json.dumps(result), flush=True)
        log.write(json.dumps(result)+'\n')
        if options.save_checkpoint and not smoke:
            torch.save(dict(model=model.state_dict(), ngram_table=table.shard,
                            prefix_table=model.prefix_table, canon_mask=model.canon_mask,
                            yarn=[dict(angular_freq=y.angular_freq, attn_scale=y.attn_scale) for y in model.yarns],
                            config=metadata, step=schedule.total_steps), logdir/'inference.pt')
    finally:
        log.close()
        prefetch_staging.prep.shutdown(wait=True)
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
