"""Full-size MI300 KS3 research: full validation loss versus complete-run wall time."""
import argparse
import json
import os
from pathlib import Path
import time
from itertools import islice
import torch
import torch.distributed as dist

from mi300.config import Hyperparameters
from mi300.data import ScheduledBatches, cu_seqlens_rows, distributed_data_generator
from mi300.distributed import setup_distributed
from mi300.model.prefix_prediction import build_prefix_table_bucket
from mi300.ngram_table import NGRAM_DIM, NGRAM_VOCAB_SIZE, MAX_CYCLE_STEPS, is_update_step
from mi300.perf.pinned_batches import PinnedBatchStaging
from mi300.perf.row_prefetch import PrefetchStaging, RowPrefetch
from mi300.perf.value_embed_pull import ValueEmbedPull
from ks3_research.sampling import KS3SampledSoftmax
from mi300.canonical_mask import build_canonical_mask
from ks3_research.model import KS3GPT
from ks3_research.optim import make_manager
from ks3_research.planning import schedule_for, signature
from ks3_research.continuation import (HoldLast, ContinuingBatches, ContinuingPrefetch,
    ContinuingNgramTable, ContinuingTails, TargetLoop, canonical_validation)
from ks3_research.ops.screening import TRAIN_DEFINITION, VAL_DEFINITION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    options = parser.parse_args()
    out = options.run_dir
    config = json.loads((out/'config.json').read_text())
    wall_start = float(os.environ['KS3_WALL_START'])
    elapsed = lambda: time.perf_counter()-wall_start
    torch.set_num_threads(8)
    env = setup_distributed()
    torch.manual_seed(config['seed'])
    args = Hyperparameters()
    schedule = schedule_for(config, config['plan']['base_steps'], env.device)
    pilot, smoke = config['mode'] == 'pilot', config['mode'] == 'smoke'
    checking = pilot or smoke
    if checking and config.get('verify_attention_buffers'):
        from test_ks3_attention import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks = run_device_checks(env.device,config.get('attention_backend','buffered_sdpa'))
        (out/'attention-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_loss_head'):
        from test_ks3_loss import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,config['loss_slab_tokens'],config['loss_backend'])
        (out/'loss-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_ks3_backward'):
        from test_ks3_backward import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,config['ks3_backward'],config['ks3_forward'])
        (out/'backward-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_optimizer_graphs'):
        from test_ks3_optimizer_graphs import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device)
        (out/'optimizer-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    target_loop = TargetLoop(config['target_val_loss'], config['validation_interval_seconds'],
                             config['first_validation_seconds'])
    start_step = schedule.total_steps - 24 if smoke else 0
    if pilot:
        steps = [0, 1, 320, 321]
        for start in (336, 512, 680, 964, 1104, 1172, 1176):
            steps.extend(range(start, start+4))
    elif smoke:
        # Diagnostic only: continuous real updates across the split and initial
        # horizon, followed by twelve further updates using the production path.
        steps = islice(target_loop.steps(start_step), 36)
    else:
        steps = target_loop.steps()
    max_tokens = max(s.batch_size for s in schedule.stages)
    capacity = max(args.val_batch_size, MAX_CYCLE_STEPS*max_tokens)
    staging = PinnedBatchStaging(capacity, cu_seqlens_rows(capacity), env.device)
    model = KS3GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128,
                   model_dim=768, max_seq_len=capacity, ngram_dim=NGRAM_DIM,
                   world_size=1, device=env.device, charts=config['charts'],
                   rank=config['rank'], nonlinear=config['nonlinear'],
                   attention_backend=config.get('attention_backend','original_sdpa'),
                   loss_backend=config['loss_backend'],loss_slab_tokens=config['loss_slab_tokens'],
                   ks3_backward=config['ks3_backward'],ks3_forward=config['ks3_forward']).to(env.device)
    model.cast_matrix_weights_bf16()
    table = ContinuingNgramTable(model.ngram_cache, max_tokens,
                                 2 if smoke else schedule.total_steps, 0, 1, env.device)
    manager = make_manager(model, schedule, config['lr_scale'], config['geometry_lr_mul'],config['lr_stage_scales'],
                           config['optimizer_backend'])
    ve = ValueEmbedPull(model.value_embeds, model.vocab_size, manager.optimizer,
                        rank=0, world_size=1, device=env.device)
    prefetch_staging = PrefetchStaging(2*MAX_CYCLE_STEPS*max_tokens, model.value_embeds.shape[0], env.device)
    sampled = KS3SampledSoftmax(schedule, model.vocab_size, 768, max_tokens, 0, 1,
                                env.device, config['sampled_candidates'])
    sampled.counts = HoldLast(sampled.counts)
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size, bucket=0, num_buckets=1))
    sampled.set_prefix_table(model.prefix_table.cpu().numpy())
    initial_stage, _ = schedule.lookup(start_step)
    loader = distributed_data_generator(args.train_files, initial_stage.batch_size,
                                        initial_stage.train_max_seq_len, staging)
    if pilot:
        batches = ScheduledBatches(loader, schedule, steps)
        pulls = RowPrefetch(table, ve, batches, steps, is_update_step, prefetch_staging)
    else:
        batches = ContinuingBatches(loader, schedule, start_step)
        pulls = ContinuingPrefetch(table, ve, batches, is_update_step, prefetch_staging)
    tails = ContinuingTails(manager.optimizer, 0, is_update_step, schedule.total_steps)
    if smoke:
        for start, _ in schedule.boundaries[1:]:
            if start <= start_step:
                manager.advance_schedule(start)
    parameters = list(manager.optimizer.param_cfgs)
    metadata = dict(gpu=torch.cuda.get_device_name(), arch=torch.cuda.get_device_properties(0).gcnArchName,
                    torch=torch.__version__, hip=torch.version.hip, seed=config['seed'],
                    model_parameters=sum(p.numel() for p in model.parameters()),
                    ffn_parameters=model.ffn_parameters(), ngram_rows=NGRAM_VOCAB_SIZE,
                    ngram_parameters=NGRAM_VOCAB_SIZE*NGRAM_DIM,
                    planned_steps=schedule.total_steps, boundaries=schedule.boundaries,
                    trainable_geometry=True, projection_master_dtype='float32',
                    mode=config['mode'], target_loss=config['target_val_loss'],
                    attention_backend=model.attention_backend,
                    loss_backend=model.loss_backend,loss_slab_tokens=model.loss_slab_tokens,
                    ks3_backward=model.ks3_backward,float32_matmul_precision=torch.get_float32_matmul_precision(),
                    ks3_forward=model.ks3_forward,
                    optimizer_backend=config['optimizer_backend'],
                    optimizer_graphs=(manager.optimizer.bank_graphs.stats()
                        if hasattr(manager.optimizer,'bank_graphs') else None),
                    global_lr_scale=config['lr_scale'],lr_stage_scales=config['lr_stage_scales'],
                    sampled_candidates=config['sampled_candidates'],
                    control_wall_seconds=config['control_wall_seconds'],
                    termination='diagnostic_steps' if checking else 'full_validation_target_only',
                    validation_interval_seconds=config['validation_interval_seconds'],
                    continuation='final_stage_lr_floor_clamped_rail_persistent_tail_ema')
    (out/'model.json').write_text(json.dumps(metadata,indent=2)+'\n')
    log = (out/'metrics.jsonl').open('w', buffering=1)
    def emit(value):
        log.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    emit(metadata)
    profile = []
    validations = []
    total_tokens = 0
    canon_ready = False
    started_training = elapsed()
    reason = 'diagnostic_complete' if checking else None

    def stage_index(step):
        return next((i for i,(_,end) in enumerate(schedule.boundaries) if step < end),
                    len(schedule.stages)-1)

    @torch.no_grad()
    def evaluate(step, tokens, *, final=False):
        nonlocal canon_ready
        begun = time.perf_counter()
        ve.complete_land()
        ve.gather_replica()
        if not canon_ready:
            model.canon_mask.copy_(torch.from_numpy(build_canonical_mask(model.vocab_size)))
            canon_ready = True
        averaged = pilot or step+1 >= schedule.total_steps
        if smoke:
            preserved = (table.adam_events, id(table.exp_avg_sq), id(table.last_event),
                         batches.next_index, tuple((s,id(b)) for s,b in batches.ahead.items()),
                         sampled.builder.sweep_offset, id(sampled.staging),
                         tuple(pulls.live_cycle), id(pulls.prep_pending), manager.optimizer.split_embed)
        with tails.evaluation_weights(averaged) as weights, canonical_validation(manager, step):
            batch_size = min(tokens, args.val_batch_size)
            assert tokens % batch_size == 0
            val_loader = distributed_data_generator(args.val_files, batch_size, -1, staging, align_to_bos=False)
            preparation_seconds = time.perf_counter()-begun
            batch_seconds = []
            total = 0.
            for index in range(tokens//batch_size):
                batch_started = time.perf_counter()
                batch = next(val_loader)
                slots = table.load_eval_batch(batch.ngram_ids)
                value = model(batch.inputs, batch.targets, batch.cum_seqlens, slots,
                              manager.get_forward_args()).sum()
                if not torch.isfinite(value).item():
                    raise RuntimeError('Nonfinite validation loss')
                total += float(value)
                batch_seconds.append(time.perf_counter()-batch_started)
                if tokens == args.val_tokens and (index+1)%10 == 0:
                    emit(dict(event='validation_progress', batches=index+1,
                              total_batches=tokens//batch_size, wall_seconds=elapsed()))
        if smoke:
            assert preserved == (table.adam_events, id(table.exp_avg_sq), id(table.last_event),
                         batches.next_index, tuple((s,id(b)) for s,b in batches.ahead.items()),
                         sampled.builder.sweep_offset, id(sampled.staging),
                         tuple(pulls.live_cycle), id(pulls.prep_pending), manager.optimizer.split_embed)
        # Include restoration overhead, not just the last validation batch.
        torch.cuda.synchronize()
        metric = dict(event='validation', step=step, val_loss=total/tokens,
                      val_tokens=tokens, full_validation=tokens==args.val_tokens,
                      wall_seconds=elapsed(), evaluation_seconds=time.perf_counter()-begun,
                      preparation_seconds=preparation_seconds, batch_seconds=batch_seconds,
                      training_tokens=total_tokens, final=final, weights=weights,
                      attention_windows=[schedule.stages[-1].window_sizes[0]*128,
                                         schedule.ws_post_yarn_ext*128],
                      evaluation_signature=dict(loss_definition=VAL_DEFINITION,val_tokens=tokens,
                                                attention_windows=[768,2560]),
                      peak_gib=torch.cuda.max_memory_allocated()/2**30)
        emit(metric)
        validations.append(metric)
        return metric

    try:
        last_step = -1
        for step in steps:
            profiler = None
            if pilot and step == config.get('profile_step'):
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True)
                profiler.start()
            t = time.perf_counter()
            manager.advance_schedule(step)
            batch = batches.peek(step)
            slots = pulls.forward_slots(step, batch.ngram_ids)
            ve.complete_land()
            sampled.upload(step, batch.targets_cpu)
            sample = sampled.gather(step, model.lm_head.weight)
            sink = table.grad_sink(batch.inputs.numel())
            loss = model(batch.inputs, batch.targets, batch.cum_seqlens, slots,
                         manager.get_forward_args(sample), ngram_sink=sink,
                         value_embed_grad=ve.grad_accum).sum()
            pulls.after_forward(step)
            grads = torch.autograd.grad(loss, parameters+[sink], allow_unused=True)
            for param, grad in zip(parameters, grads[:-1]):
                if grad is None:
                    continue
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
            total_tokens += batch.inputs.numel()
            last_step = step
            # Synchronize pilots for timings; full runs retain normal launch overlap.
            if checking:
                torch.cuda.synchronize()
                stage = stage_index(step)
                profile.append(dict(step=step, stage=stage, step_seconds=time.perf_counter()-t))
            if checking or (step+1)%max(1,int(config.get('log_every_steps',25))) == 0:
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f'Nonfinite objective at step {step}')
                if checking:
                    for param, grad in zip(parameters, grads[:-1]):
                        if grad is not None and not torch.isfinite(grad).all().item():
                            raise RuntimeError(f'Nonfinite gradient: {param.label}')
                        if not torch.isfinite(param).all().item():
                            raise RuntimeError(f'Nonfinite parameter: {param.label}')
                    if not torch.isfinite(grads[-1]).all().item():
                        raise RuntimeError('Nonfinite n-gram gradient')
                emit(dict(event='training', step=step, train_objective=float(loss.detach())/batch.inputs.numel(),
                          wall_seconds=elapsed(), training_tokens=total_tokens,
                          objective_signature=dict(
                              stage=stage_index(step),
                              candidates=sampled.counts[step],
                              loss_definition=TRAIN_DEFINITION,
                              attention_windows=[manager.ws_short*128,manager.ws_long*128],
                              train_max_seq_len=manager.train_max_seq_len,
                              attention_segment_cap=2560,batch_tokens=batch.inputs.numel(),
                              mtp=manager.mtp_weights.detach().cpu().tolist(),
                              prefix=float(manager.prefix_weight.detach().item())),
                          sparse_adam_events=table.adam_events,
                          lr_multiplier=manager.effective_lr_multiplier(step),
                          global_lr_scale=config['lr_scale'],
                          peak_gib=torch.cuda.max_memory_allocated()/2**30))
            del loss, grads, sink
            if profiler is not None:
                profiler.stop()
                profiler.export_chrome_trace(str(out/'profile-trace.json'))
                timing = [dict(name=e.key, calls=e.count, shapes=e.input_shapes,
                               self_cpu_us=e.self_cpu_time_total, cpu_us=e.cpu_time_total,
                               self_device_us=e.self_device_time_total, device_us=e.device_time_total)
                          for e in profiler.key_averages(group_by_input_shape=True)]
                timing.sort(key=lambda r:r['self_device_us'], reverse=True)
                (out/'profile-operators.json').write_text(json.dumps(timing,indent=2)+'\n')
                emit(dict(event='profile_complete', step=step, operators=len(timing)))
            if step+1 == schedule.total_steps:
                emit(dict(event='schedule_boundary', step=step, wall_seconds=elapsed(),
                          continuing=True, sparse_adam_events=table.adam_events))
            if smoke and step in (start_step+3, start_step+23, start_step+31):
                evaluate(step, 1024)
            if not checking and target_loop.due(elapsed(), is_update_step(step)):
                metric = evaluate(step, args.val_tokens)
                if target_loop.observe(metric):
                    reason = 'full_validation_target_reached'
                    emit(dict(event='target_hit', **{k:v for k,v in metric.items() if k != 'event'}))
                    break
        pulls.close()
        if pilot:
            assert manager.optimizer.split_embed and not table.pending
            activity = dict(coefficients_changed=int(torch.count_nonzero(model.ks3_coefficients)),
                            scales_changed=int(torch.count_nonzero(model.ks3_scale-.75)),
                            shifts_changed=int(torch.count_nonzero(model.ks3_shift)))
            if not all(activity.values()):
                raise RuntimeError(f'Geometry did not train: {activity}')
            emit(dict(event='geometry_activity', **activity))
        if checking:
            tokens = 1024 if smoke else 3*args.val_batch_size if pilot else args.val_tokens
            metric = evaluate(last_step, tokens, final=True)
        else:
            assert target_loop.first_hit is not None, 'A full trial must not end at a schedule boundary'
            metric = validations[-1]
        if smoke:
            assert last_step >= schedule.total_steps+11 and manager.optimizer.split_embed
            assert not table.pending and table.adam_events == 9
            assert len(validations) == 4 and table.beta2_history.numel() > 3
            emit(dict(event='continuation_verified', last_step=last_step,
                      planned_steps=schedule.total_steps, sparse_adam_events=table.adam_events,
                      history_capacity=table.beta2_history.numel(), evaluations=len(validations)))
        if pilot:
            (out/'pilot.json').write_text(json.dumps(dict(signature=signature(config), steps=profile,
                setup_seconds=started_training, validation_seconds=metric['evaluation_seconds'],
                validation_tokens=metric['val_tokens'], geometry_activity=activity,
                validation_setup_seconds=metric['preparation_seconds'],
                validation_batch_seconds=metric['batch_seconds'],
                source_sha256=config['source_sha256']),indent=2)+'\n')
        result = dict(val_loss=metric['val_loss'], validation_tokens=metric['val_tokens'],
                      full_validation=metric['full_validation'], reason=reason,
                      last_step=last_step, training_tokens=total_tokens,
                      wall_seconds_at_evaluation=metric['wall_seconds'],
                      first_target_hit=target_loop.first_hit,
                      time_to_target_seconds=(target_loop.first_hit or {}).get('wall_seconds'),
                      weights=metric['weights'],
                      peak_gib=torch.cuda.max_memory_allocated()/2**30,
                      ffn_parameters=model.ffn_parameters(), validations=validations)
        if hasattr(manager.optimizer,'bank_graphs'):
            result['optimizer_graphs']=manager.optimizer.bank_graphs.stats()
        if config.get('save_checkpoint') and not checking:
            with tails.evaluation_weights(metric['weights']=='tail_average'), canonical_validation(manager, last_step):
                torch.save(dict(model=model.state_dict(), ngram_table=table.shard,
                                prefix_table=model.prefix_table, canon_mask=model.canon_mask,
                                yarn=[dict(angular_freq=y.angular_freq,attn_scale=y.attn_scale) for y in model.yarns],
                                config=config, step=last_step),out/'inference.pt')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        log.close()
        prefetch_staging.prep.shutdown(wait=True)
        sampled.staging.prep.shutdown(wait=True)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
