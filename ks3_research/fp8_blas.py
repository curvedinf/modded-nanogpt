"""Native MI300 FP8 BLAS products for the KS3 experimental BLAS path.

Prepared per-tensor E4M3FNUZ operands, FP32 accumulation/output, and no BF16
operand conversion. This adds FP8 rounding after GS32 dequantization; it is a
numerical variant, not a bitwise substitute for the source KS3 kernel.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _amax(A,P,M:tl.constexpr,N:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.load(A+(i//N)*S0+(i%N)*S1,i<M*N,other=0.).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(a),0))

@tr.jit
def _quant(A,S,Q,M:tl.constexpr,N:tl.constexpr,PM:tl.constexpr,PN:tl.constexpr,
           S0:tl.constexpr,S1:tl.constexpr,COLUMN:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    if COLUMN:row=i%PM;col=i//PM
    else:row=i//PN;col=i%PN
    a=tl.load(A+row*S0+col*S1,(row<M)&(col<N)&(i<PM*PN),other=0.).to(tl.float32)
    scale=tl.load(S)
    q=tl.minimum(tl.maximum(a/scale,-240.),240.)
    tl.store(Q+i,q,i<PM*PN)

def quantize(a,shape,*,column=False):
    if not a.is_cuda or a.ndim!=2 or not torch.version.hip:raise ValueError('Native ROCm matrix required')
    m,n=a.shape;pm,pn=shape
    partial=torch.empty((tr.cdiv(a.numel(),4096),),device=a.device,dtype=torch.float32)
    _amax[(partial.numel(),)](a,partial,m,n,*a.stride(),4096)
    scale=partial.amax().clamp_min(1e-12)/240.
    storage=torch.empty((pn,pm) if column else (pm,pn),device=a.device,dtype=torch.float8_e4m3fnuz)
    _quant[(tr.cdiv(pm*pn,4096),)](a,scale,storage,m,n,pm,pn,*a.stride(),column,4096)
    return (storage.T if column else storage),scale

def matrix_product(a,b):
    """Standalone reference utility; the training block uses Operand/product reuse."""
    if a.shape[1]!=b.shape[0]:raise ValueError('GEMM inner dimensions differ')
    m,k=a.shape;n=b.shape[1]
    if not m or not n or not k:return torch.zeros((m,n),device=a.device,dtype=torch.float32)
    pm,pk,pn=(tr.cdiv(s,16)*16 for s in (m,k,n))
    aa,sa=quantize(a,(pm,pk));bb,sb=quantize(b,(pk,pn),column=True)
    with torch.autocast('cuda',enabled=False):
        out=torch._scaled_mm(aa,bb,scale_a=sa,scale_b=sb,out_dtype=torch.float32,use_fast_accum=False)
    return out[:m,:n].contiguous()


@tr.jit
def _pack_pair(A, S, SCALE, ROW, COL, M:tl.constexpr,N:tl.constexpr,
               PM:tl.constexpr,PN:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,
               GS32:tl.constexpr,B:tl.constexpr):
    # One read creates both layouts; no intermediate unpacked full matrix.
    r=tl.program_id(0)*B+tl.arange(0,B)
    c=tl.program_id(1)*B+tl.arange(0,B)
    valid=(r[:,None]<M)&(c[None,:]<N)
    a=tl.load(A+r[:,None]*A0+c[None,:]*A1,valid,other=0.).to(tl.float32)
    if GS32:
        scales=tl.load(S+r[:,None]*tl.cdiv(N,32)+c[None,:]//32,valid,other=0.).to(tl.float32)
        a=a*scales
    scale=tl.load(SCALE)
    q=tl.minimum(tl.maximum(a/scale,-240.),240.).to(ROW.dtype.element_ty)
    padded=(r[:,None]<PM)&(c[None,:]<PN)
    tl.store(ROW+r[:,None]*PN+c[None,:],q,padded)
    tl.store(COL+c[None,:]*PM+r[:,None],q,padded)


@tr.jit
def _finish_scale(A,S,N:tl.constexpr,FACTOR:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    a=tl.load(A+i,i<N,other=0.).to(tl.float32)
    value=tl.maximum(tl.max(tl.abs(a),0),1e-12)*FACTOR
    tl.store(S,value)


def operand_scale(a,factor):
    # Fuse the final reduction, zero guard and scale arithmetic in one launch.
    values=a.contiguous()
    if a.numel()>32768:
        values=torch.empty((tr.cdiv(a.numel(),4096),),device=a.device,dtype=torch.float32)
        _amax[(values.numel(),)](a,values,*a.shape,*a.stride(),4096)
    scale=torch.empty((),device=a.device,dtype=torch.float32)
    _finish_scale[(1,)](values,scale,values.numel(),factor,tr.next_power_of_2(values.numel()))
    return scale


class Operand:
    """One FP8 value set, two physical layouts, reused across forward/backward."""
    def __init__(self,a,scales=None):
        m,n=a.shape;pm,pn=(tr.cdiv(v,16)*16 for v in (m,n))
        self.shape=(m,n)
        if scales is not None:
            # Existing GS32 scales bound decoded values; no full-matrix amax pass.
            self.scale=operand_scale(scales,127./240.)
        else:
            self.scale=operand_scale(a,1./240.)
        self.row=torch.empty((pm,pn),device=a.device,dtype=torch.float8_e4m3fnuz)
        self.col=torch.empty((pn,pm),device=a.device,dtype=torch.float8_e4m3fnuz).T
        _pack_pair[(tr.cdiv(pm,32),tr.cdiv(pn,32))](a,scales if scales is not None else a,
            self.scale,self.row,self.col,m,n,pm,pn,*a.stride(),scales is not None,32)


def product(a,b,transpose_a=False,transpose_b=False):
    aa=a.col.T if transpose_a else a.row
    bb=b.row.T if transpose_b else b.col
    m=a.shape[1] if transpose_a else a.shape[0]
    n=b.shape[0] if transpose_b else b.shape[1]
    with torch.autocast('cuda',enabled=False):
        out=torch._scaled_mm(aa,bb,scale_a=a.scale,scale_b=b.scale,
                             out_dtype=torch.float32,use_fast_accum=False)
    return out[:m,:n].contiguous()
