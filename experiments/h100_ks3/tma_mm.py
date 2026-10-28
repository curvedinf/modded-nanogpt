"""Persistent Hopper TMA products, adapted from PR393's grouped_mm_kernel.

Unlike the upstream BF16-only output, preserve FP32/FP16 temporaries for
the KS3 geometry and its derivative. Weight operands use cached K-major
layouts; this wrapper performs no packing or tensor copies.
"""
import os
import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

TILE=os.environ.get('KS3_FP8_MM_TILE','128x128')
TILES={'64x128':(64,128,128,4,3),'128x128':(128,128,128,8,3),
       '128x256':(128,256,128,8,3)}
assert TILE in TILES


@tr.jit
def kernel(A,W,O,TABLE,SA,SW,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SMS:tl.constexpr):
    cols=tl.cdiv(N,BN);tiles=tl.cdiv(M,BM)*cols
    scale_a=tl.load(SA)
    for tile in tl.range(tl.program_id(0),tiles,SMS,flatten=True):
        mr=tile//cols;nc=tile%cols
        expert=tl.load(TABLE+mr*BM//128)
        scale=scale_a*tl.load(SW+expert)
        acc=tl.zeros((BM,BN),tl.float32)
        for k in range(tl.cdiv(K,BK)):
            a=A.load([mr*BM,k*BK]);w=W.load([expert*N+nc*BN,k*BK])
            acc=tl.dot(a,w.T,acc)
        acc*=scale
        split=tl.permute(tl.reshape(acc,(BM,2,BN//2)),(0,2,1))
        first,second=tl.split(split)
        O.store([mr*BM,nc*BN],first)
        O.store([mr*BM,nc*BN+BN//2],second)


def mm(a,w,table,sa,sw,dtype):
    m,k=a.shape;n=w.shape[-1];bm,bn,bk,nw,st=TILES[TILE]
    rows=w.transpose(1,2)
    assert a.is_contiguous() and rows.is_contiguous()
    assert m%bm==0 and n%bn==0 and k%bk==0
    rows=rows.view(-1,k)
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    ad=TensorDescriptor.from_tensor(a,[bm,bk]);wd=TensorDescriptor.from_tensor(rows,[bn,bk])
    od=TensorDescriptor.from_tensor(out,[bm,bn//2])
    sms=torch.cuda.get_device_properties(a.device).multi_processor_count
    kernel[(min(sms,tr.cdiv(m,bm)*tr.cdiv(n,bn)),)](ad,wd,od,table,sa,sw,m,n,k,bm,bn,bk,sms,num_warps=nw,num_stages=st)
    return out
