"""Full-run rank-zero CUPTI profile. Its timing is never a record result."""
from collections import defaultdict
from pathlib import Path
import hashlib,json,time
import torch

PROFILER=None
REGION=None

def start(rank):
    global PROFILER,REGION
    if rank!=0:return
    PROFILER=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
    PROFILER.start()
    REGION=torch.profiler.record_function('WR/full_training');REGION.__enter__()

def validation(rank):
    global REGION
    if rank!=0:return
    REGION.__exit__(None,None,None)
    REGION=torch.profiler.record_function('WR/full_validation');REGION.__enter__()

def summarize(events,phase):
    bounds=[e for e in events if e.get('name')==phase and e.get('ph')=='X']
    lo=min(e['ts'] for e in bounds);hi=max(e['ts']+e['dur'] for e in bounds)
    groups=defaultdict(lambda:dict(count=0,total_us=0.,max_us=0.));intervals=[]
    for e in events:
        if e.get('cat') not in ('kernel','gpu_kernel') or e.get('ph')!='X':continue
        t=e['ts'];d=e['dur']
        if t<lo or t>=hi:continue
        x=groups[e['name']];x['count']+=1;x['total_us']+=d;x['max_us']=max(x['max_us'],d);intervals.append((t,t+d))
    total=sum(x['total_us'] for x in groups.values());union=0.;right=-float('inf')
    for a,b in sorted(intervals):union+=max(0,b-max(a,right));right=max(right,b)
    ranked=sorted([dict(name=k,**v,mean_us=v['total_us']/v['count'],summed_fraction=v['total_us']/total) for k,v in groups.items()],key=lambda x:x['total_us'],reverse=True)
    return dict(phase=phase,cpu_range_us=hi-lo,kernel_count=len(intervals),distinct_kernels=len(groups),sum_kernel_us=total,union_kernel_us=union,ranked_kernels=ranked,warning='Kernel duration sums include overlapping streams and are not wall-time shares. Profiling overhead is present.')

def stop(rank):
    global PROFILER,REGION
    if rank!=0:return
    torch.cuda.synchronize();REGION.__exit__(None,None,None);PROFILER.stop()
    path=Path('../full-profile-trace.json');PROFILER.export_chrome_trace(str(path))
    data=json.loads(path.read_text());events=data['traceEvents']
    results={name:summarize(events,'WR/full_'+name) for name in ['training','validation']
             if any(e.get('name')=='WR/full_'+name and e.get('ph')=='X' for e in events)}
    assert results['training']['kernel_count']>0,'No CUDA kernels captured'
    results.update(profiled_rank=0,profiled_device='H100',timing_reference_eligible=False,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trace_bytes=path.stat().st_size)
    Path('../full-profile-summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print('[profile] available training/validation CUDA kernel summaries exported',flush=True)
