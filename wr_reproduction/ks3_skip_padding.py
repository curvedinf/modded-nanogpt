"""Skip only whole invalid routing blocks in KS3 BF16 projection products.

Original dispatch starts every expert at a multiple of128 and places padding
after that expert's real rows. Every original projection row tile divides128.
Thus a sentinel in the first row proves the whole tile contains no real token.
The GEMM operands, dot loop, tile selection and valid-row stores are unchanged.
Evaluation delegates to the original implementation. No shared kernels change.
"""
import torch
import triton as tr
import triton.language as tl
from wr_reproduction import ks3_moe as core
from wr_reproduction.bf16_tiles import select_projection_tiles

ORIGINAL_FORWARD=core.forward
ORIGINAL_BACKWARD=core.backward
ORIGINAL_GEOMETRY=core._geometry
ORIGINAL_VJP=core._vjp
KIND='bf16-skip-whole-sentinel-routing-tiles-v1'
EXECUTED_MODES=set()
MATRIX_SHAPES=set()


@tr.jit
def mm_skip(A,W,O,TILES,SRC,TOKENS:tl.constexpr,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            W0:tl.constexpr,W1:tl.constexpr,W2:tl.constexpr,
            BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    tl.static_assert(BM<=128 and 128 % BM==0)
    r=tl.program_id(0)*BM+tl.arange(0,BM);c=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);e=tl.load(TILES+tl.program_id(0)*BM//128)
    acc=tl.zeros((BM,BN),tl.float32)
    first=tl.load(SRC+tl.program_id(0)*BM,tl.program_id(0)*BM<M,other=TOKENS)
    if first<TOKENS:
        for base in range(tl.cdiv(K,BK)):
            k=base*BK+kk
            a=tl.load(A+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),other=0.)
            w=tl.load(W+e*W0+k[:,None]*W1+c[None,:]*W2,(k[:,None]<K)&(c[None,:]<N),other=0.)
            acc=tl.dot(a,w,acc)
    tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<M)&(c[None,:]<N))


def mm(a,w,tiles,src,tokens,dtype=torch.float32):
    from wr_reproduction.ks3_bf16_gemm import TILE_MODE
    m,k=a.shape;e,kw,n=w.shape
    assert k==kw and a.is_contiguous() and a.dtype==w.dtype==torch.bfloat16
    assert src.numel()==m and TILE_MODE=='mi300-v1'
    bm,bn,bk,warps,stages,mfma=select_projection_tiles(m,n,k,w.stride(2)==1,TILE_MODE)
    assert bm<=128 and 128%bm==0
    MATRIX_SHAPES.add((tokens,m,n,k,bm,bn,bk,warps,stages,mfma))
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    mm_skip[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,w,out,tiles,src,tokens,m,n,k,*w.stride(),
        BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
    return out


@torch.library.custom_op('wr_ks3_skip_padding::forward',mutates_args=())
def forward(rows:torch.Tensor,src:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,
            down:torch.Tensor,g:torch.Tensor,gs32:bool=True)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if not core.BF16_TUNED_KERNELS:
        EXECUTED_MODES.add('original_evaluation')
        return ORIGINAL_FORWARD(rows,src,tiles,up,down,g,gs32)
    if not gs32 or core.KS3_RANK!=1:raise ValueError('Original rank1 GS32 experts required')
    EXECUTED_MODES.add('training_skip_padded_forward')
    m,d=src.numel(),rows.shape[1]
    x=torch.empty((m,d),device=rows.device,dtype=torch.bfloat16)
    core._pack[(tr.cdiv(m,128),tr.cdiv(d,32))](rows,x,m,d,*rows.stride(),
        SRC=src,INPUT_ROWS=rows.shape[0],ROWS_PER_TILE=128,GS32=True,num_warps=4)
    c=mm(x,up.transpose(1,2),tiles,src,rows.shape[0]);h=torch.empty_like(c,dtype=torch.bfloat16)
    core._geometry[(tr.cdiv(c.shape[0],64),tr.cdiv(c.shape[1],32))](
        c,g,h,tiles,*c.shape,BM=64,RANK=1,GS32=True,num_warps=4)
    y=mm(h,down.transpose(1,2),tiles,src,rows.shape[0],dtype=rows.dtype)
    return y,x,c,h


@forward.register_fake
def _(rows,src,tiles,up,down,g,gs32=True):
    m=src.numel();d=rows.shape[1];n=up.shape[1]
    return rows.new_empty((m,d)),rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((m,n))


@torch.library.custom_op('wr_ks3_skip_padding::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,gate:torch.Tensor,
             src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,
             bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if not core.BF16_TUNED_KERNELS:
        EXECUTED_MODES.add('original_untuned_backward')
        return ORIGINAL_BACKWARD(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,bank,layer)
    EXECUTED_MODES.add('training_skip_padded_backward')
    if up.shape[0]==8:core.EXECUTED_MOE_BACKWARDS.add(layer)
    valid=(src<grad.shape[0])[:,None];idx=src.clamp(0,grad.shape[0]-1).long()
    gathered=grad[idx].float();gg=torch.where(valid,gathered*gate[idx,None].to(torch.bfloat16).float(),0).to(torch.bfloat16)
    ds=torch.where(valid,gathered*y.float(),0).sum(1)
    v=mm(gg,down,tiles,src,grad.shape[0]);m,n=c.shape;e=up.shape[0]
    dc=torch.empty_like(c,dtype=torch.bfloat16);part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,dc,part,tiles,src,m,n,grad.shape[0],RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    from wr_reproduction.ks3_bf16_gemm import dw
    dw(h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs)
    dw(dc,x,bank[2*layer*e:(2*layer+1)*e],offs)
    dx=mm(dc,up,tiles,src,grad.shape[0],dtype=grad.dtype)
    return dx,ds,dg


@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g)


def install():
    if core.forward not in (ORIGINAL_FORWARD,forward) or core.backward not in (ORIGINAL_BACKWARD,backward):
        raise RuntimeError('Unexpected prior KS3 forward/backward overlay')
    if core._geometry is not ORIGINAL_GEOMETRY or core._vjp is not ORIGINAL_VJP:
        raise RuntimeError('Original geometry and VJP are required')
    core.forward=forward;core.backward=backward
