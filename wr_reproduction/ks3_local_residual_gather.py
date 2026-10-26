"""Apply the audited exact BF16 producer to the compact-residual backward.

Only the gradient gather/gate producer changes. Local features, their original
BF16 gradient boundary, all projections, and the original Torch gate-gradient
row reduction remain identical to the unfused compact-residual implementation.
"""
import torch
from wr_reproduction import ks3_moe as core, ks3_local_residual as local
from wr_reproduction.ks3_bf16_gather import gather

ORIGINAL_BACKWARD=local.backward


@torch.library.custom_op('wr_ks3_local_gather::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,
             gate:torch.Tensor,src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,
             up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,raw:torch.Tensor,bound:float,
             bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    assert core.BF16_TUNED_KERNELS and core.KS3_RANK==1 and up.shape[0]==8
    core.EXECUTED_MOE_BACKWARDS.add(layer)
    gg,ds=gather(grad,y,src,gate)
    v=core.mm(gg,down,tiles);e=up.shape[0]
    dc,dg,dp=local.feature_vjp(v,c,g,raw,tiles,src,offs,grad.shape[0],bound)
    from wr_reproduction.ks3_bf16_gemm import dw
    dw(h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs)
    dw(dc,x,bank[2*layer*e:(2*layer+1)*e],offs)
    return core.mm(dc,up,tiles,dtype=grad.dtype),ds,dg,dp


@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,raw,bound,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g),torch.empty_like(raw)


def install():
    if getattr(local.backward,'_ks3_local_gather',False):return
    if local.backward is not ORIGINAL_BACKWARD:raise RuntimeError('Unexpected local-residual backward overlay')
    def dispatch(*args,**kwargs):
        return (backward if core.BF16_TUNED_KERNELS else ORIGINAL_BACKWARD)(*args,**kwargs)
    dispatch._ks3_local_gather=True
    local.backward=dispatch
