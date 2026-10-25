"""Pointer-tiled MI300 port of record anvil_opt.gemm_xb_add.

Preserves BF16 rounding of X@B before the fused add and final BF16 store.
The original ANVIL maps, momentum, equalizer and weight update remain untouched.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _xb(X,B,O,M:tl.constexpr,N:tl.constexpr,ALPHA:tl.constexpr,
        BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=64):
    batch=tl.program_id(2)
    r=tl.program_id(0)*BM+tl.arange(0,BM);c=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for i in range(tl.cdiv(N,BK)):
        kk=i*BK+k
        a=tl.load(X+batch*M*N+r[:,None]*N+kk[None,:],(r[:,None]<M)&(kk[None,:]<N),other=0.)
        b=tl.load(B+batch*N*N+c[None,:]*N+kk[:,None],(c[None,:]<N)&(kk[:,None]<N),other=0.)
        acc=tl.dot(a,b,acc)
    index=batch*M*N+r[:,None]*N+c[None,:];mask=(r[:,None]<M)&(c[None,:]<N)
    original=tl.load(X+index,mask,other=0.).to(tl.float32)
    value=tl.fma(original,ALPHA,acc.to(tl.bfloat16).to(tl.float32))
    tl.store(O+index,value,mask)

def gemm_xb_add(x,b,out,alpha,**kwargs):
    batch,m,n=x.shape
    _xb[(tr.cdiv(m,32),tr.cdiv(n,64),batch)](x,b,out,m,n,alpha,BM=32,BN=64,BK=64,num_warps=4)
    return out

# Original pointer-based Gram kernels with MI300-sized launch tiles.
def gram_tall(x,out):
    from track_1_short.perf.kernels import polar_express as pe
    batch,m,k=x.shape
    pe.XTX_kernel[(batch*tr.cdiv(k,64)*tr.cdiv(k,64),)](A_ptr=x,C_ptr=out,M=m,K=k,
        a_stride_b=x.stride(0),a_stride_r=x.stride(1),a_stride_c=x.stride(2),
        c_stride_b=out.stride(0),c_stride_r=out.stride(1),c_stride_c=out.stride(2),
        BLOCK_SIZE_M=64,BLOCK_SIZE_N=64,BLOCK_SIZE_K=32,GROUP_SIZE_M=8,LOWER_UPPER=1,num_stages=1,num_warps=4)
    return out

def gram_wide(x,out):
    from track_1_short.perf.kernels import polar_express as pe
    batch,m,k=x.shape
    pe.XXT_kernel[(batch*tr.cdiv(m,64)*tr.cdiv(m,64),)](A_ptr=x,C_ptr=out,M=m,K=k,
        a_stride_b=x.stride(0),a_stride_r=x.stride(1),a_stride_c=x.stride(2),
        c_stride_b=out.stride(0),c_stride_r=out.stride(1),c_stride_c=out.stride(2),
        BLOCK_SIZE_M=64,BLOCK_SIZE_N=64,BLOCK_SIZE_K=32,GROUP_SIZE_M=8,LOWER_UPPER=1,num_stages=1,num_warps=4)
    return out

def poly_gram(a,alpha,beta,out):
    from track_1_short.perf.kernels import polar_express as pe
    batch,m,_=a.shape
    pe.ba_plus_cAA_kernel[(batch*tr.cdiv(m,64)*tr.cdiv(m,64),)](A_ptr=a,C_ptr=out,M=m,
        a_stride_b=a.stride(0),a_stride_r=a.stride(1),a_stride_c=a.stride(2),
        c_stride_b=out.stride(0),c_stride_r=out.stride(1),c_stride_c=out.stride(2),alpha=alpha,beta=beta,
        BLOCK_SIZE_M=64,BLOCK_SIZE_N=64,BLOCK_SIZE_K=32,GROUP_SIZE_M=8,LOWER_UPPER=1,num_stages=1,num_warps=4)
    return out
