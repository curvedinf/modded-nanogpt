"""KS3-only fused BF16 gradient producer; original products and STE retained.

The incumbent backward materializes a gathered FP32 matrix, gate products,
masked matrices and a reduction. This producer writes the BF16 gated gradient
and the original FP32 dot-product terms. The original Torch row reduction is
retained: changing its reduction order can perturb BF16 router gradients.
"""
import torch
import triton as tr
import triton.language as tl
from wr_reproduction import ks3_moe as core


@tr.jit
def _gather(G,Y,SRC,GATE,OUT,PRODUCTS,N:tl.constexpr,M:tl.constexpr,D:tl.constexpr,
            BM:tl.constexpr=4,BD:tl.constexpr=1024):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.arange(0,BD)
    source=tl.load(SRC+rows,rows<M,other=N)
    valid=(rows<M)&(source<N)&(source>=0)
    mask=valid[:,None]&(cols[None,:]<D)
    grad=tl.load(G+source[:,None].to(tl.int64)*D+cols[None,:],mask,other=0.).to(tl.float32)
    y=tl.load(Y+rows[:,None].to(tl.int64)*D+cols[None,:],mask,other=0.).to(tl.float32)
    gate=tl.load(GATE+source,valid,other=0.).to(tl.bfloat16).to(tl.float32)
    value=(grad*gate[:,None]).to(tl.bfloat16)
    tl.store(OUT+rows[:,None].to(tl.int64)*D+cols[None,:],value,
             (rows[:,None]<M)&(cols[None,:]<D))
    tl.store(PRODUCTS+rows[:,None].to(tl.int64)*D+cols[None,:],grad*y,
             (rows[:,None]<M)&(cols[None,:]<D))


def gather(grad,y,src,gate):
    n,d=grad.shape;m=src.numel()
    assert d==768 and grad.dtype==y.dtype==torch.bfloat16
    assert grad.is_contiguous() and y.is_contiguous() and y.shape==(m,d)
    assert gate.dtype==torch.float32 and gate.shape==(n,) and gate.is_contiguous()
    assert src.dtype==torch.int32 and src.is_contiguous()
    out=torch.empty((m,d),device=grad.device,dtype=torch.bfloat16)
    products=torch.empty((m,d),device=grad.device,dtype=torch.float32)
    _gather[(tr.cdiv(m,4),)](grad,y,src,gate,out,products,n,m,d,num_warps=4,
                            enable_fp_fusion=False)
    return out,products.sum(1)


@torch.library.custom_op('wr_ks3_bf16_gather::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,gate:torch.Tensor,
             src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,
             bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    assert core.BF16_TUNED_KERNELS and core.KS3_RANK==1
    if up.shape[0]==8:core.EXECUTED_MOE_BACKWARDS.add(layer)
    gg,ds=gather(grad,y,src,gate)
    # The remaining operations, tensors and reduction topology are the
    # incumbent tuned BF16 backward, including its original64-row geometry VJP.
    v=core.mm(gg,down,tiles);m,n=c.shape;e=up.shape[0]
    dc=torch.empty_like(c,dtype=torch.bfloat16)
    part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,dc,part,tiles,src,m,n,grad.shape[0],RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    from wr_reproduction.ks3_bf16_gemm import dw
    dw(h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs)
    dw(dc,x,bank[2*layer*e:(2*layer+1)*e],offs)
    dx=core.mm(dc,up,tiles,dtype=grad.dtype)
    return dx,ds,dg


@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g)


def install():
    """Install once before model/graph construction; evaluation uses original math."""
    if getattr(core.backward,'_ks3_bf16_gather',False):return
    original=core.backward
    def dispatch(*args,**kwargs):
        if core.BF16_TUNED_KERNELS:return backward(*args,**kwargs)
        return original(*args,**kwargs)
    dispatch._ks3_bf16_gather=True
    dispatch._original=original
    core.backward=dispatch
