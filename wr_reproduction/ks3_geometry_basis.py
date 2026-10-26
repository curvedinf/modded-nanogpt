"""KS3-only cubic reparameterization; original quantized nonlinear math stays.

The first four master values are coefficients of T0..T3(t/4), t=4*tanh(x/4).
They map to the original Hermite values (p(0), p(1), p'(0), p'(1)) before
the unchanged coefficient GS32/FP16 producer. Backward applies the transpose
of that linear map to the original STE. The two warp coefficients and raw
scale/shift remain in their original coordinates. No new state or parameters.
"""
import torch
import triton as tr
import triton.language as tl
from wr_reproduction import ks3_moe as core
from wr_reproduction import ks3_bf16_gather as gather

ORIGINAL_FORWARD=core.forward
ORIGINAL_BACKWARD=core.backward
KIND='cubic-chebyshev-normalized-to-hermite-v1'


def reference_map(a):
    """Independent Torch expression, also usable for double-precision audits."""
    a0,a1,a2,a3,a4,a5,rs,rh=a.unbind(-1)
    return torch.stack((a0-a2,a0+a1*.25-a2*.875-a3*.6875,
                        a1*.25-a3*.75,a1*.25+a2*.25-a3*.5625,a4,a5,rs,rh),-1)


@tr.jit
def _map(A,O,N:tl.constexpr,TRANSPOSE:tl.constexpr,B:tl.constexpr=256):
    row=tl.program_id(0)*B+tl.arange(0,B);mask=row<N
    a0=tl.load(A+row*8+0,mask,other=0.);a1=tl.load(A+row*8+1,mask,other=0.)
    a2=tl.load(A+row*8+2,mask,other=0.);a3=tl.load(A+row*8+3,mask,other=0.)
    if TRANSPOSE:
        b0=a0+a1;b1=a1*.25+a2*.25+a3*.25
        b2=-a0-a1*.875+a3*.25;b3=-a1*.6875-a2*.75-a3*.5625
    else:
        b0=a0-a2;b1=a0+a1*.25-a2*.875-a3*.6875
        b2=a1*.25-a3*.75;b3=a1*.25+a2*.25-a3*.5625
    tl.store(O+row*8+0,b0,mask);tl.store(O+row*8+1,b1,mask)
    tl.store(O+row*8+2,b2,mask);tl.store(O+row*8+3,b3,mask)
    for col in tl.static_range(4,8):
        tl.store(O+row*8+col,tl.load(A+row*8+col,mask,other=0.),mask)


def transform(a,transpose=False):
    assert a.dtype==torch.float32 and a.is_contiguous() and a.shape[-1]==8
    out=torch.empty_like(a)
    _map[(tr.cdiv(a.numel()//8,256),)](a,out,a.numel()//8,transpose,enable_fp_fusion=False)
    return out


@torch.library.custom_op('wr_ks3_basis::forward',mutates_args=())
def forward(rows:torch.Tensor,src:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,
            down:torch.Tensor,g:torch.Tensor,gs32:bool=True
            )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return ORIGINAL_FORWARD(rows,src,tiles,up,down,transform(g),gs32)


@forward.register_fake
def _(rows,src,tiles,up,down,g,gs32=True):
    m=src.numel();d=rows.shape[1];n=up.shape[1]
    return rows.new_empty((m,d)),rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((m,n))


@torch.library.custom_op('wr_ks3_basis::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,
             gate:torch.Tensor,src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,
             up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,bank:torch.Tensor,layer:int
             )->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    original=gather.backward if core.BF16_TUNED_KERNELS else ORIGINAL_BACKWARD
    dx,ds,dg=original(grad,y,x,c,h,gate,src,offs,tiles,up,down,transform(g),bank,layer)
    return dx,ds,transform(dg,transpose=True)


@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g)


def install():
    """Process-fixed before model construction; never change graph-owned state."""
    if core.forward is forward:return
    if core.forward is not ORIGINAL_FORWARD or core.backward is not ORIGINAL_BACKWARD:
        raise RuntimeError('Unexpected prior KS3 core overlay')
    core.forward=forward;core.backward=backward
