"""Sparse KS3-only measurements; all costs remain on the continuous run clock."""
from contextlib import contextmanager
import torch

def sample_indices(elements,device):
    # Geometry stores six coefficients, scale and shift in contiguous groups of8.
    # Sample complete groups so neither a stride nor integer rounding can alias
    # one channel. The same bounded grouping is valid for other vector moments.
    if elements%8==0:
        groups=elements//8;count=min(groups,512)
        chosen=torch.arange(count,device=device,dtype=torch.int64)*(groups-1)//max(1,count-1)
        return (8*chosen[:,None]+torch.arange(8,device=device,dtype=torch.int64)).reshape(-1)
    count=min(elements,4096)
    return torch.arange(count,device=device,dtype=torch.int64)*(elements-1)//max(1,count-1)

@torch.no_grad()
def moments(t):
    flat=t.detach().reshape(-1);x=flat[sample_indices(flat.numel(),flat.device)].float()
    return torch.stack((x.mean(),x.square().mean().sqrt(),x.abs().max(),(x!=0).float().mean()))

@contextmanager
def activations(enabled,output):
    if not enabled:
        yield;return
    from wr_reproduction import ks3_moe as core
    original=core.forward
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        y,x,c,h=result
        output.append(dict(branch='dense' if args[3].shape[0]==1 else 'routed',
                           values=torch.stack((moments(c),moments(h),moments(y)))))
        return result
    core.forward=observe
    try:yield
    finally:core.forward=original

def activation_report(rows):
    result=[]
    for row in rows:
        values=row['values'].cpu().tolist()
        result.append(dict(branch=row['branch'],coordinate=dict(zip(('mean','rms','max_abs','nonzero_fraction'),values[0])),
                           nonlinear_output=dict(zip(('mean','rms','max_abs','nonzero_fraction'),values[1])),
                           projected_output=dict(zip(('mean','rms','max_abs','nonzero_fraction'),values[2])),
                           sampling='deterministic-evenly-spaced-integer4096-elements-first-probe-chunk',training_objective_used=False))
    return result

@torch.no_grad()
def before_update(model):
    result={}
    for name in ('ks3_geometry','ks3_dense_geometry','moe_bank','mlp_bank'):
        if not hasattr(model,name) or (name=='mlp_bank' and not hasattr(model,'ks3_dense_geometry')):continue
        p=getattr(model,name);indices=sample_indices(p.numel(),p.device)
        result[name]=(p.detach().reshape(-1)[indices].float().clone(),indices)
    return result

@torch.no_grad()
def after_update(model,before):
    result={}
    for name,(previous,indices) in before.items():
        p=getattr(model,name);current=p.detach().reshape(-1)[indices].float();delta=current-previous
        data=torch.stack((previous.square().mean().sqrt(),delta.square().mean().sqrt(),
                          p.grad.detach().reshape(-1)[indices].float().square().mean().sqrt() if p.grad is not None else current.new_tensor(float('nan')))).cpu().tolist()
        result[name]=dict(branch='dense' if name in ('ks3_dense_geometry','mlp_bank') else 'routed',
                          parameter_rms=data[0],update_rms=data[1],relative_update_rms=data[1]/max(data[0],1e-12),
                          gradient_sample_rms=data[2] if p.grad is not None else None,
                          gradient_l2_estimate=data[2]*p.numel()**.5 if p.grad is not None else None,
                          gradient_missing_reason=None if p.grad is not None else 'Adam-not-applied-this-original-update',
                          sample_elements=current.numel(),total_elements=p.numel(),
                          sampling='evenly-spaced-integer-indices; includes all geometry channels; bank padding may remain',
                          single_original_update=True)
    return result
