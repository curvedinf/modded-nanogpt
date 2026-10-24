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
from control393.sampling import SampledSoftmax as KS3SampledSoftmax
from mi300.canonical_mask import build_canonical_mask
from control393.model import MoEGPT, norm_map
from control393.optim import make_manager
from ks3_research.planning import schedule_for
def signature(config): return {k:config.get(k) for k in ("architecture","charts","rank","batch_tokens","stage_durations","attention_backend","loss_slab_tokens","sampled_head","ks3_dynamic_rows","ks3_bf16_products","ks3_expert_unbind","optimizer_backend")}
from control393.continuation import (HoldLast, ContinuingBatches, ContinuingPrefetch,
    ContinuingNgramTable, ContinuingTails, TargetLoop, canonical_validation)
TRAIN_DEFINITION="pr393-cplm-moe-aux-sampled-ce-v1"
VAL_DEFINITION="pr393-cplm-full10485760-canonical-final-yarn-v1"


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
    if pilot and config.get('tiled_attention_probe_only'):
        from test_control393_tiled_attention import run_device_checks
        checks=run_device_checks(env.device,compact=config.get('tiled_attention_compact',False),
                                 async_metadata=config.get('tiled_attention_async_metadata',False))
        (out/'tiled-attention-probe.json').write_text(json.dumps(checks,indent=2)+'\n')
        result=dict(val_loss=None,validation_tokens=0,full_validation=False,
                    reason='tiled_attention_probe_complete',tiled_attention_probe=checks,
                    wall_seconds_at_evaluation=elapsed(),time_to_target_seconds=None,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        dist.destroy_process_group()
        return
    if pilot and config.get('native_flash_probe_only'):
        from test_control393_native_flash import run_device_checks
        checks=run_device_checks(env.device,config.get('native_probe_providers',('aotriton','ck')))
        (out/'native-flash-probe.json').write_text(json.dumps(checks,indent=2)+'\n')
        result=dict(val_loss=None,validation_tokens=0,full_validation=False,
                    reason='native_flash_probe_complete',native_flash_probe=checks,
                    wall_seconds_at_evaluation=elapsed(),time_to_target_seconds=None,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        dist.destroy_process_group()
        return
    if checking and config.get('verify_tiled_attention'):
        from test_control393_tiled_attention import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,compact=config.get('tiled_attention_compact',False),
                                     async_metadata=config.get('tiled_attention_async_metadata',False))
        (out/'tiled-attention-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not checks['passed']:raise RuntimeError('Tiled attention failed mandatory numerical checks')
    if checking and config.get('verify_training_loss'):
        from test_control393_training_loss import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,rows=config['loss_slab_tokens'])
        (out/'training-loss-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_loss_slabs'):
        from test_control393_slabs import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,config['loss_slab_tokens'],
                                     tiled_head=config.get('head_projection_rows',0)==4096)
        (out/'loss-slab-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_eval_loss'):
        from test_control393_eval_loss import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,config.get('eval_loss_block',4096))
        (out/'eval-loss-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_eval_buckets'):
        from test_control393_eval import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device)
        (out/'eval-attention-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_eval_metadata'):
        from test_control393_eval_metadata import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device)
        (out/'eval-metadata-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not checks['passed']:raise RuntimeError('Canonical metadata evaluation failed numerical checks')
    if checking and config['architecture'] == 'pr393-moe-ks3':
        from test_control393_ks3 import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks = run_device_checks(env.device,dynamic_rows=config.get('ks3_dynamic_rows',False),
                                       bf16_products=config.get('ks3_bf16_products',False),charts=config['charts'],
                                       expert_unbind=config.get('ks3_expert_unbind',False))
        (out/'moe-ks3-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_attention_buffers'):
        from test_ks3_attention import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks = run_device_checks(env.device,config.get('attention_backend','buffered_sdpa'))
        (out/'attention-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checking and config.get('verify_training_buckets'):
        from test_control393_bucket_training import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device)
        (out/'training-bucket-verification.json').write_text(json.dumps(checks,indent=2)+'\n')
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
        from test_control393_optimizer_graphs import run_device_checks
        with torch.random.fork_rng(devices=[env.device]):
            checks=run_device_checks(env.device,charts=config['charts'])
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
    from control393.attention_metadata import HostBoundaryStaging
    staging_type=(HostBoundaryStaging if config['attention_backend'] in
                  ('gathered_sdpa128_tiledasync','gathered_sdpa128_tiledasync_fasteval')
                  else PinnedBatchStaging)
    staging = staging_type(capacity, cu_seqlens_rows(capacity), env.device)
    import mi300.data as data_module
    from mi300.ngram_table import ngram_row_ids as original_ngram_ids
    normalized=norm_map()
    data_module.ngram_row_ids=lambda x: original_ngram_ids(normalized[x.long()])
    model_type=MoEGPT
    model_extra={}
    if config['architecture']=='pr393-moe-ks3':
        from control393.ks3_model import MoEKS3GPT
        model_type=MoEKS3GPT;model_extra=dict(charts=config['charts'],rank=config.get('rank',1),
                                           dynamic_rows=config.get('ks3_dynamic_rows',False),
                                           bf16_products=config.get('ks3_bf16_products',False),
                                           expert_unbind=config.get('ks3_expert_unbind',False))
    model = model_type(vocab_size=50257,num_layers=11,num_heads=6,head_dim=128,
                   model_dim=768,max_seq_len=capacity,ngram_dim=NGRAM_DIM,world_size=1,device=env.device,
                   attention_backend=config['attention_backend'],loss_slab_tokens=config['loss_slab_tokens'],
                   sampled_head=config['sampled_head'],**model_extra).to(env.device)
    model.cast_matrix_weights_bf16()
    model.eval_loss_backend=config.get('eval_loss_backend','eager')
    model.eval_loss_block=config.get('eval_loss_block',4096)
    if model.eval_loss_backend not in ('eager','fused_canonical_v1'):
        raise ValueError('Unknown evaluation loss backend')
    model.training_loss_backend=config.get('training_loss_backend','eager')
    model.head_projection_rows=config.get('head_projection_rows',0)
    if model.head_projection_rows not in (0,4096):raise ValueError('Unsupported head projection tile')
    if model.training_loss_backend not in ('eager','compiled_cplm_v1'):
        raise ValueError('Unknown training loss backend')
    ks3_before = None
    if checking and hasattr(model, 'ks3_coefficients'):
        ks3_before = dict(bank=model.moe_bank[:, :16, :16].detach().clone(),
                          router=model.moe_router.detach().clone())
    table = ContinuingNgramTable(model.ngram_cache, max_tokens,
                                 2 if smoke else schedule.total_steps, 0, 1, env.device)
    manager=make_manager(model,schedule,config['lr_scale'],config['lr_stage_scales'],
                         config.get('geometry_lr_mul',.125),config.get('expert_lr_mul',1.),
                         config.get('optimizer_backend','eager'))
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
                    architecture=config['architecture'], projection_master_dtype=str(model.moe_bank.dtype),
                    mode=config['mode'], target_loss=config['target_val_loss'],
                    attention_backend=model.attention_backend,
                    loss_backend=model.loss_backend,loss_slab_tokens=model.loss_slab_tokens,
                    eval_loss_backend=model.eval_loss_backend,eval_loss_block=model.eval_loss_block,
                    training_loss_backend=model.training_loss_backend,
                    head_projection_rows=model.head_projection_rows,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    optimizer_backend=config['optimizer_backend'],
                    optimizer_graphs=(manager.optimizer.bank_graphs.stats()
                        if hasattr(manager.optimizer,'bank_graphs') else None),
                    global_lr_scale=config['lr_scale'],lr_stage_scales=config['lr_stage_scales'],
                    geometry_lr_mul=config.get('geometry_lr_mul',.125),
                    expert_lr_mul=config.get('expert_lr_mul',1.),
                    ks3_bf16_products=config.get('ks3_bf16_products',False),
                    ks3_expert_unbind=config.get('ks3_expert_unbind',False),
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
        tail_ready=tails.ready()
        averaged = pilot or step+1 >= schedule.total_steps or (
            config.get('eval_tail_when_seeded',False) and tail_ready)
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
                val_profiler=None
                if pilot and index==config.get('profile_validation_batch'):
                    val_profiler=torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False)
                    val_profiler.start()
                batch_started = time.perf_counter()
                batch = next(val_loader)
                slots = table.load_eval_batch(batch.ngram_ids)
                value = model(batch.inputs, batch.targets, batch.cum_seqlens, slots,
                              manager.get_forward_args()).sum()
                if not torch.isfinite(value).item():
                    raise RuntimeError('Nonfinite validation loss')
                total += float(value)
                batch_seconds.append(time.perf_counter()-batch_started)
                if val_profiler is not None:
                    val_profiler.stop()
                    val_profiler.export_chrome_trace(str(out/'validation-profile-trace.json'))
                    timing=[dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,
                                 cpu_us=e.cpu_time_total,self_device_us=e.self_device_time_total,
                                 device_us=e.device_time_total)
                            for e in val_profiler.key_averages()]
                    timing.sort(key=lambda r:r['self_device_us'],reverse=True)
                    (out/'validation-profile-operators.json').write_text(json.dumps(timing,indent=2)+'\n')
                    emit(dict(event='validation_profile_complete',batch=index,operators=len(timing)))
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
                      tail_ready=tail_ready,
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
            selected_profiles=config.get('profile_steps',[config.get('profile_step')])
            if pilot and step in selected_profiles:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=config.get('profile_record_shapes',True))
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
            if profiler is not None:
                # Keep the diagnostic all-parameter finite checks out of the
                # profile of actual training/prefetch/optimizer work.
                profiler.stop()
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
                              attention_mask_semantics=config.get('attention_mask_semantics',
                                                                  'legacy-pr393-grouped-mask-v1'),
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
                suffix=f'-{step}' if 'profile_steps' in config else ''
                profiler.export_chrome_trace(str(out/f'profile-trace{suffix}.json'))
                timing = [dict(name=e.key, calls=e.count, shapes=e.input_shapes,
                               self_cpu_us=e.self_cpu_time_total, cpu_us=e.cpu_time_total,
                               self_device_us=e.self_device_time_total, device_us=e.device_time_total)
                          for e in profiler.key_averages(group_by_input_shape=True)]
                timing.sort(key=lambda r:r['self_device_us'], reverse=True)
                (out/f'profile-operators{suffix}.json').write_text(json.dumps(timing,indent=2)+'\n')
                emit(dict(event='profile_complete', step=step, operators=len(timing)))
            if step+1 == schedule.total_steps:
                emit(dict(event='schedule_boundary', step=step, wall_seconds=elapsed(),
                          continuing=True, sparse_adam_events=table.adam_events))
            if smoke and step in (start_step+3, start_step+23, start_step+31):
                evaluate(step, 8192)
            if not checking and config.get('screen_seconds') and elapsed()>=config['screen_seconds'] and is_update_step(step):
                reason='bounded_control_tuning'
                metric=evaluate(step,config.get('screen_validation_tokens',262144))
                break
            if not checking and target_loop.due(elapsed(), is_update_step(step)):
                metric = evaluate(step, args.val_tokens)
                if target_loop.observe(metric):
                    reason = 'full_validation_target_reached'
                    emit(dict(event='target_hit', **{k:v for k,v in metric.items() if k != 'event'}))
                    break
        pulls.close()
        if ks3_before is not None:
            activity = dict(
                expert_matrices_changed=int(((model.moe_bank[:, :16, :16]-ks3_before['bank'])
                                             .abs().flatten(1).sum(1)>0).sum()),
                routers_changed=int(((model.moe_router-ks3_before['router'])
                                     .abs().flatten(1).sum(1)>0).sum()),
                coefficients_changed=int(torch.count_nonzero(model.ks3_coefficients)),
                scales_changed=int(torch.count_nonzero(model.ks3_scale-.75)),
                shifts_changed=int(torch.count_nonzero(model.ks3_shift)))
            assert activity['expert_matrices_changed'] == 80, activity
            assert activity['routers_changed'] == 5, activity
            assert all(activity[k] > 0 for k in ('coefficients_changed','scales_changed','shifts_changed')), activity
            (out/'moe-ks3-updates.json').write_text(json.dumps(activity,indent=2)+'\n')
        if pilot:
            assert manager.optimizer.split_embed and not table.pending
            activity=dict(moe_router_norm=float(model.moe_router.detach().float().norm()),copy_gain=float(model.copy_qk_gain.detach()))
            if hasattr(model,'ks3_coefficients'):
                activity.update(coefficients_changed=int(torch.count_nonzero(model.ks3_coefficients)),
                    scales_changed=int(torch.count_nonzero(model.ks3_scale-.75)),shifts_changed=int(torch.count_nonzero(model.ks3_shift)))
                assert all(activity[k]>0 for k in ('coefficients_changed','scales_changed','shifts_changed'))
        if checking:
            if pilot and config.get('verify_eval_metadata'):
                fast=evaluate(last_step,args.val_batch_size)
                from control393.eval_attention import bucketed_attention
                try:
                    for layer in model.attn.values():layer.attention=bucketed_attention
                    original=evaluate(last_step,args.val_batch_size)
                finally:
                    for layer in model.attn.values():del layer.attention
                warmed=evaluate(last_step,args.val_batch_size)
                assert fast['val_loss']==original['val_loss']==warmed['val_loss'],(fast,original,warmed)
                (out/'eval-metadata-model-verification.json').write_text(json.dumps(dict(
                    passed=True,bitwise_equal=True,tokens=args.val_batch_size,
                    original_loss=original['val_loss'],fast_loss=fast['val_loss'],
                    original_batch_seconds=original['batch_seconds'],
                    fast_batch_seconds=warmed['batch_seconds']),indent=2)+'\n')
            if pilot and config.get('verify_loss_slabs'):
                larger=evaluate(last_step,16384)
                previous=model.loss_slab_tokens
                try:
                    model.loss_slab_tokens=4096
                    original=evaluate(last_step,16384)
                finally:
                    model.loss_slab_tokens=previous
                difference=abs(larger['val_loss']-original['val_loss'])
                assert difference<=2e-6,(larger,original)
                (out/'slab-model-verification.json').write_text(json.dumps(dict(
                    passed=True,tokens=16384,slab4096_loss=original['val_loss'],
                    slab8192_loss=larger['val_loss'],absolute_difference=difference),indent=2)+'\n')
            if pilot and config.get('verify_eval_loss'):
                fused=evaluate(last_step,8192)
                previous=model.eval_loss_backend
                try:
                    model.eval_loss_backend='eager'
                    eager=evaluate(last_step,8192)
                finally:
                    model.eval_loss_backend=previous
                difference=abs(fused['val_loss']-eager['val_loss'])
                assert difference<=2e-6,(fused,eager)
                (out/'eval-loss-model-verification.json').write_text(json.dumps(dict(
                    passed=True,tokens=8192,fused_loss=fused['val_loss'],eager_loss=eager['val_loss'],
                    absolute_difference=difference),indent=2)+'\n')
            if pilot and config.get('verify_eval_buckets'):
                grouped=evaluate(last_step,8192)
                from mi300.portable import varlen_attention
                try:
                    for layer in model.attn.values():layer.attention=varlen_attention
                    original=evaluate(last_step,8192)
                finally:
                    for layer in model.attn.values():del layer.attention
                assert grouped['val_loss']==original['val_loss'],(grouped,original)
                (out/'eval-model-verification.json').write_text(json.dumps(dict(
                    passed=True,bitwise_equal=True,tokens=8192,grouped_loss=grouped['val_loss'],
                    original_loss=original['val_loss']),indent=2)+'\n')
            tokens = 8192 if smoke else 3*args.val_batch_size if pilot else args.val_tokens
            metric = evaluate(last_step, tokens, final=True)
        else:
            assert target_loop.first_hit is not None or reason=='bounded_control_tuning', 'A control must reach target'
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
