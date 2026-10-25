"""One physical optimizer owner, after eight logical-rank gradients accumulate.

Only use inside the serialized trainer. Logical data, sampling and scale states
still have eight ranks. Collectives here copy already-reduced tensors; they do
not divide gradients a second time or change any optimizer hyperparameters.
"""
from contextlib import ExitStack
from unittest.mock import patch
import torch

class Complete:
    def wait(self):return None
    def get_future(self):
        f=torch.futures.Future();f.set_result(None);return f

def _copy(output,input,**kwargs):
    if output.numel()!=input.numel():raise ValueError('Physical single-owner collective shape mismatch')
    if output.data_ptr()!=input.data_ptr():output.copy_(input.reshape(output.shape))
    return Complete()

def _reduce(tensor,*args,**kwargs):return Complete()

class LocalCollectives(ExitStack):
    def __enter__(self):
        super().__enter__()
        for name,fn in dict(get_rank=lambda *a,**k:0,get_world_size=lambda *a,**k:1,
            all_gather_into_tensor=_copy,reduce_scatter_tensor=_copy,all_reduce=_reduce,broadcast=_reduce,reduce=_reduce).items():
            self.enter_context(patch('torch.distributed.'+name,fn))
        return self
