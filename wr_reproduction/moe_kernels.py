"""Native FP8 replacements for PR393 grouped Hopper GEMMs.

Routing, padding, quantization, auxiliary loss, autograd and gradient bank layout
are the ORIGINAL source implementation. Each grouped product is one GPU launch with device-side expert selection.
"""
import torch
def grouped_mm(a,b,tiles,scales,scale,n):
    from wr_reproduction.fused_fp8 import run
    return run(a,b.T,scales,scale,out_dtype=torch.bfloat16,tiles=tiles,n=n)[0]

def grouped_lrs(a,b,tiles,scales,scale,n,forward,post_scale,amax,aux,dpre_scale):
    from wr_reproduction.fused_fp8 import run
    q,qt,_,_=run(a,b.T,scales,scale,mode=1 if forward else 2,
        post_scale=post_scale,dpre_scale=dpre_scale,amax=amax,aux=aux,
        emit_t=True,tiles=tiles,n=n)
    return q,qt

def grouped_weight_grad(a,b,offs,sa,sb,experts,out,acc_flag):
    from wr_reproduction.fused_fp8 import grouped_dw
    grouped_dw(a,b,offs,sa,sb,experts,out,acc_flag)

def install(upstream_moe):
    upstream_moe._mm_op.register_kernel('cuda',grouped_mm)
    upstream_moe._lrs_op.register_kernel('cuda',grouped_lrs)
    upstream_moe._grouped_weight_grad_impl=grouped_weight_grad

# Validation's original BF16 grouped products. Pointer tiles replace the
# platform-specific torch._grouped_mm provider; routing and relu^2 stay in source.
import triton as tr
import triton.language as tl

@tr.jit
def _bf16_grouped(A,B,O,OFFS,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  A0:tl.constexpr,A1:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,
                  BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    e=tl.program_id(2);lo=0
    if e>0:lo=tl.load(OFFS+e-1)
    hi=tl.load(OFFS+e)
    if tl.program_id(0)*BM<hi-lo:
        r=lo+tl.program_id(0)*BM+tl.arange(0,BM)
        c=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for base in range(tl.cdiv(K,BK)):
            k=base*BK+kk
            a=tl.load(A+r[:,None]*A0+k[None,:]*A1,(r[:,None]<hi)&(k[None,:]<K),other=0.)
            b=tl.load(B+e*B0+k[:,None]*B1+c[None,:]*B2,(k[:,None]<K)&(c[None,:]<N),other=0.)
            acc=tl.dot(a,b,acc)
        tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<hi)&(c[None,:]<N))

@torch.library.custom_op('wr_mi300::grouped_bf16',mutates_args=())
def grouped_bf16(a:torch.Tensor,b:torch.Tensor,offs:torch.Tensor)->torch.Tensor:
    m,k=a.shape;e,bk,n=b.shape
    if k!=bk or a.dtype!=torch.bfloat16 or b.dtype!=torch.bfloat16:raise ValueError('Unexpected validation GEMM')
    out=torch.empty((m,n),device=a.device,dtype=a.dtype)
    _bf16_grouped[(tr.cdiv(m,32),tr.cdiv(n,64),e)](a,b,out,offs,m,n,k,*a.stride(),*b.stride(),BM=32,BN=64,BK=32,num_warps=4)
    return out

@grouped_bf16.register_fake
def _(a,b,offs):return a.new_empty((a.shape[0],b.shape[-1]))

def grouped_mm_validation(a,b,*,offs,**kwargs):
    if kwargs:raise ValueError('Unverified grouped GEMM options')
    return grouped_bf16(a,b,offs)
