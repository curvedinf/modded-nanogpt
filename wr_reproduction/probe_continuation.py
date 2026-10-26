"""Same-checkpoint, same-eight-microbatch A/B check at an even original update.

The chosen update cannot mutate the130GB sparse table or Adam states. Snapshot
only the live graph-owned model/cache, ANVIL, rank and sampler state it can write.
This is an off-clock diagnostic, never a research run or training checkpoint.
"""
import dataclasses
import torch
from wr_reproduction.accumulation import OriginalGradientAccumulator

class RepeatedBatch:
    def __init__(self,batch):self.batch=batch
    def peek(self,step):return self.batch
    def take(self,step):return self.batch

@torch.no_grad()
def baseline_and_restore(r,batches,step):
    assert step%2==0 and not r.tables.event_policy(step)
    assert not any(r.tables.pending) and not any(r.gradients.pending)
    for sampler in r.samplers:
        if sampler.staging.pending:sampler.staging.pending[1].result()
    fixed=[RepeatedBatch(b.peek(step)) for b in batches]
    torch.cuda.synchronize()
    copies=[];arrays=[];seen=set()
    def remember(value):
        import numpy as np
        if isinstance(value,torch.Tensor):
            if id(value) not in seen:
                seen.add(id(value));copies.append((value,value.clone()))
        elif isinstance(value,np.ndarray):arrays.append((value,value.copy()))
        elif isinstance(value,dict):
            for v in value.values():remember(v)
        elif isinstance(value,(tuple,list)):
            for v in value:remember(v)
        elif dataclasses.is_dataclass(value):
            for f in dataclasses.fields(value):remember(getattr(value,f.name))
    for p in r.model.parameters():remember(p)
    for b in r.model.buffers():remember(b)
    for y in r.model.yarns:remember(y.angular_freq)
    remember(r.rank_states.states);remember(r.counter_states);remember(r.sampled_head.COUNTER)
    remember(r.shared_value_grad);remember(r.tables.value_grads)
    for bank in r.tm.optimizer.banks.values():remember(bank)
    scalar_objects=[r,r.model,r.tm,r.tm.optimizer]
    pending=[]
    for sampler in r.samplers:
        remember(vars(sampler.builder));remember(sampler.candidates);remember(sampler.loss_inputs)
        remember(sampler.staging.slots)
        pending.append((sampler.staging,sampler.staging.pending))
        scalar_objects.extend([sampler,sampler.builder,sampler.staging])
    scalars=[(obj,{k:v for k,v in vars(obj).items() if isinstance(v,(int,float,bool,str))}) for obj in scalar_objects]
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    # Runner.step needs autograd enabled only for its eager wrappers; captured graphs
    # themselves replay their original forward/backward regardless of this context.
    with torch.enable_grad():loss=r.step(step,fixed,prefetch=False)
    torch.cuda.synchronize()
    reference=dict(loss=loss.item(),parameters={n:p.detach().cpu().clone() for n,p in r.model.named_parameters()})
    for target,saved in copies:target.copy_(saved)
    for target,saved in arrays:target[:]=saved
    for obj,values in scalars:
        for k,v in values.items():setattr(obj,k,v)
    for staging,value in pending:staging.pending=value
    for entries in r.tables.pending:entries.clear()
    r.gradients=OriginalGradientAccumulator(p for p,c in r.tm.optimizer.param_cfgs.items() if c.optim=='adam')
    r.model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);torch.cuda.synchronize()
    return reference

@torch.no_grad()
def compare(r,loss,reference):
    values=[]
    for name,p in r.model.named_parameters():
        a=p.detach().float().cpu();b=reference['parameters'][name].float()
        delta=a-b
        values.append(dict(parameter=name,max_abs=delta.abs().max().item(),
                           relative_rms=(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)).item(),
                           exact=torch.equal(a,b)))
    maximum=max(v['relative_rms'] for v in values)
    # Nondeterministic GPU reductions may produce BF16 rounding differences.
    assert abs(loss-reference['loss'])<1e-5,(loss,reference['loss'])
    assert maximum<.002,sorted(values,key=lambda v:v['relative_rms'])[-5:]
    return dict(passed=True,same_checkpoint=True,same_eight_logical_microbatches=True,
                baseline_loss=reference['loss'],with_probe_loss=loss,
                maximum_parameter_relative_rms=maximum,all_parameters_bitwise_equal=all(v['exact'] for v in values),parameters=values)
