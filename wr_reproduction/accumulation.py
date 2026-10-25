"""Accumulate eight ORIGINAL rank contributions without changing optimizer cadence.

This is a tested primitive, not a replacement for the original trainer. The
runtime must supply complete logical-rank forwards and rank-local FP8/sampling
state; arbitrary token slicing would change MoE auxiliary and sampled objectives.
Scale gradients AFTER backward because original fused FP8 autograd operators do
not in general honor an outer scalar loss multiplier. Never update/requantize
weights between contributions. Sparse four-step updates and dense two-step Adam
accumulation remain responsibilities of the original step controller.
"""
class RankAccumulator:
    def __init__(self,global_tokens,world_size=8):
        if world_size!=8 or global_tokens%world_size:raise ValueError('Preserve eight logical ranks')
        self.global_tokens=global_tokens
        self.local_tokens=global_tokens//world_size
        self.world_size=world_size
        self.seen=set()
        self.sums={}

    def add(self,rank,tokens,gradients):
        if rank not in range(self.world_size) or rank in self.seen:raise ValueError('Missing/duplicate logical rank')
        if tokens!=self.local_tokens:raise ValueError('Microbatch must preserve the complete original rank objective')
        if self.seen and set(gradients)!=set(self.sums):raise ValueError('Gradient labels changed across ranks')
        for name,g in gradients.items():
            if g is None:raise ValueError(f'Missing gradient: {name}')
            if name not in self.sums:self.sums[name]=g.detach().float().clone()
            else:self.sums[name].add_(g.detach().float())
        self.seen.add(rank)

    def finish(self):
        if len(self.seen)!=self.world_size:raise RuntimeError('Cannot step before all eight rank contributions')
        return {name:g/self.world_size for name,g in self.sums.items()}

class OriginalGradientAccumulator:
    """Keep Adam's two-step addition on each logical rank before reducing ranks.

Averaging each step first and then adding would move the source's BF16 rounding
boundary. Tied embedding gradients likewise merge on each rank before reduction.
"""
    def __init__(self,adam_parameters):
        self.adam=set(adam_parameters);self.pending=[{} for _ in range(8)]
        self.begin(False)
    def begin(self,do_adam):
        self.do_adam=do_adam;self.sums={};self.seen=set();self.dtypes={}
    def add(self,rank,gradients,*,tied_pair=None,transpose_add=None):
        if rank not in range(8) or rank in self.seen:raise ValueError('Duplicate/invalid rank')
        current={}
        for key,gradient in gradients.items():
            if key in self.adam:
                if key in self.pending[rank]:self.pending[rank][key].add_(gradient)
                else:self.pending[rank][key]=gradient.detach().clone()
                if self.do_adam:current[key]=self.pending[rank][key]
            else:current[key]=gradient
        if self.do_adam and tied_pair is not None:
            lm,embed=tied_pair
            if lm in current and embed in current:
                transpose_add(current.pop(embed),current[lm])
        for key,g in current.items():
            self.dtypes[key]=g.dtype
            if key not in self.sums:self.sums[key]=g.detach().float().clone()
            else:self.sums[key].add_(g)
        self.seen.add(rank)
    def finish(self):
        if len(self.seen)!=8:raise RuntimeError('Missing logical ranks before optimizer update')
        result={key:(g/8).to(self.dtypes[key]) for key,g in self.sums.items()}
        if self.do_adam:
            for values in self.pending:values.clear()
        return result
