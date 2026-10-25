"""Isolated original-prefix profiling; never a convergence or timing reference."""
from collections import defaultdict
from contextlib import ExitStack
import gc
import json
import math
from pathlib import Path
import time
from unittest.mock import patch


def summarize_trace(trace):
    groups=defaultdict(lambda:dict(count=0,total_us=0.))
    starts=[];ends=[]
    for event in trace.get('traceEvents',[]):
        category=str(event.get('cat','')).lower()
        if event.get('ph')!='X' or category not in ('kernel','gpu_kernel','hip_kernel'):
            continue
        duration=event.get('dur');start=event.get('ts')
        if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
            raise ValueError('Invalid GPU kernel duration')
        if not isinstance(start,(int,float)) or not math.isfinite(start):
            raise ValueError('Invalid GPU kernel timestamp')
        row=groups[event['name']];row['count']+=1;row['total_us']+=duration
        starts.append(start);ends.append(start+duration)
    total=sum(row['total_us'] for row in groups.values())
    if not groups or total<=0:raise RuntimeError('Profiler captured no positive GPU kernel activity')
    ranked=[dict(name=name,**row,mean_us=row['total_us']/row['count'],fraction_of_kernel_duration_sum=row['total_us']/total)
            for name,row in groups.items()]
    ranked.sort(key=lambda row:row['total_us'],reverse=True)
    return dict(kernel_count=sum(row['count'] for row in groups.values()),distinct_kernels=len(groups),
                summed_kernel_duration_us=total,gpu_activity_span_us=max(ends)-min(starts),
                warning='Kernel durations can overlap across streams; sums/fractions are not elapsed-time shares.',
                top_kernels=ranked[:50])


def instrument(original,label):
    def wrapped(*args,**kwargs):
        import torch
        with torch.profiler.record_function(label):return original(*args,**kwargs)
    return wrapped


def run(runner,count=4):
    import torch
    from wr_reproduction.research import STATE,atomic,sha
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from track_1_short.canonical_mask import BackgroundCanonicalMask
    if not runner.research_config or runner.research_config['kind']!='profile':
        raise ValueError('Profiling requires its own noncontinuous registry kind')
    if not 1<=count<=8:raise ValueError('Profile1-8 original updates')
    directory=runner.research_run;trace_path=directory/'profile-trace.json'
    if trace_path.exists():raise FileExistsError(trace_path)
    model=runner.model;batches=runner.loaders(list(range(runner.schedule.total_steps)))
    model.limit_yarn_rebuild(runner.max_tokens)
    canonical=BackgroundCanonicalMask(model.vocab_size,owner=True,print0=lambda *args,**kwargs:None)
    canonical.start()
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size,0,1))
    prefix=model.prefix_table.cpu().numpy()
    for sampler in runner.samplers:sampler.set_prefix_table(prefix)
    gc.collect();gc.freeze();gc.disable()
    preparation=25;losses=[]
    try:
        for step in range(preparation):
            if (STATE/'STOP').exists():raise RuntimeError('Explicit study STOP')
            loss=runner.step(step,batches,next_step=step+1,prefetch=True);runner.tail.tick(step)
            value=loss.item()
            if not math.isfinite(value):raise RuntimeError('Nonfinite profiling preparation loss')
            runner.log(event='profile_preparation_update',step=step+1,objective=value)
        torch.cuda.synchronize()
        targets=[(runner.tables,'load','WR/ngram_load'),(runner.tables,'hold','WR/ngram_hold'),
                 (runner.graphs,'forward','WR/forward_graph'),(runner.graphs,'backward','WR/backward_graph'),
                 (runner.gradients,'add','WR/gradient_accumulate'),(runner.gradients,'finish','WR/gradient_finish'),
                 (runner.rank_states,'enter','WR/rank_enter'),(runner.rank_states,'leave','WR/rank_leave'),
                 (runner.tm,'step_optimizers','WR/optimizers')]
        for sampler in runner.samplers:
            for method in ('upload','gather','prefetch'):
                targets.append((sampler,method,'WR/sampled_head_'+method))
        started=time.perf_counter()
        with ExitStack() as stack:
            for owner,method,label in targets:
                stack.enter_context(patch.object(owner,method,instrument(getattr(owner,method),label)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False,profile_memory=False,with_stack=False,acc_events=True) as profiler:
                for step in range(preparation,preparation+count):
                    if (STATE/'STOP').exists():raise RuntimeError('Explicit study STOP')
                    with torch.profiler.record_function('WR/original_update'):
                        loss=runner.step(step,batches,next_step=step+1,prefetch=True);runner.tail.tick(step)
                        losses.append(loss.detach().clone())
                    profiler.step()
                torch.cuda.synchronize()
        instrumented_seconds=time.perf_counter()-started
        values=torch.stack(losses).cpu().tolist()
        if not all(math.isfinite(value) for value in values):raise RuntimeError('Nonfinite profiled loss')
        profiler.export_chrome_trace(str(trace_path))
        summary=summarize_trace(json.loads(trace_path.read_text()))
        cpu=[]
        for row in profiler.key_averages():
            cpu.append(dict(name=row.key,count=row.count,self_cpu_time_us=row.self_cpu_time_total,
                            cpu_time_total_us=row.cpu_time_total,self_device_time_us=row.self_device_time_total,
                            device_time_total_us=row.device_time_total))
        cpu.sort(key=lambda row:row['self_cpu_time_us'],reverse=True)
        summary.update(first_profiled_update=preparation+1,last_profiled_update=preparation+count,
                       profiled_updates=count,original_updates_before_profile=preparation,
                       instrumented_seconds=instrumented_seconds,objectives=values,top_cpu_events=cpu[:40],
                       trace_path=str(trace_path),trace_sha256=sha(trace_path),validation_performed=False,
                       full_convergence_verified=False,usable_as_timing_reference=False,
                       continuous_from_initialization=False)
        path=directory/'profile-summary.json';atomic(path,summary)
        runner.log(event='profile_complete',summary_path=str(path),summary_sha256=sha(path),
                   trace_path=str(trace_path),trace_sha256=summary['trace_sha256'],
                   **{k:summary[k] for k in ('first_profiled_update','last_profiled_update','profiled_updates',
                       'original_updates_before_profile','instrumented_seconds','kernel_count','distinct_kernels',
                       'summed_kernel_duration_us','gpu_activity_span_us','validation_performed',
                       'full_convergence_verified','usable_as_timing_reference','continuous_from_initialization')})
    finally:
        canonical.wait();gc.enable()
