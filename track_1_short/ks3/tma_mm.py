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
from track_1_short.ks3.h100_kernels import _features

TILE='128x256'
TILES={'64x128':(64,128,128,4,3),'64x256':(64,256,128,4,3),'128x128':(128,128,128,8,3),
       '128x256':(128,256,128,8,3)}
assert TILE in TILES


@tr.jit
def kernel(A,W,O,TABLE,SA,SW,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SMS:tl.constexpr,
           G=None,P=None,H=None,HT=None,HS=None,HM=None,MODE:tl.constexpr=0,
           SRC=None,GATE=None,SCATTER=None,DLOG=None,ROUTER=None,ROWS:tl.constexpr=0,CUBIC_ONLY:tl.constexpr=False,FLATTEN:tl.constexpr=True,GS32:tl.constexpr=True):
    cols=tl.cdiv(N,BN);tiles=tl.cdiv(M,BM)*cols
    scale_a=tl.load(SA)
    for tile in tl.range(tl.program_id(0),tiles,SMS,flatten=FLATTEN):
        mr=tile//cols;nc=tile%cols
        expert=tl.load(TABLE+mr*BM//128)
        scale=scale_a*tl.load(SW+expert)
        acc=tl.zeros((BM,BN),tl.float32)
        for k in range(tl.cdiv(K,BK)):
            a=A.load([mr*BM,k*BK]);w=W.load([expert*N+nc*BN,k*BK])
            acc=tl.dot(a,w.T,acc)
        acc*=scale
        if MODE==4:
            rows=mr*BM+tl.arange(0,BM);columns=nc*BN+tl.arange(0,BN)
            source=tl.load(SRC+rows,rows<M,other=ROWS)
            combined=acc.to(tl.bfloat16).to(tl.float32)
            for e in tl.static_range(8):
                dl=tl.load(DLOG+source*8+e,(rows<M)&(source<ROWS),other=0.).to(tl.float32)
                router=tl.load(ROUTER+e*N+columns,columns<N,other=0.).to(tl.float32)
                combined+=dl[:,None]*router[None,:]
            tl.store(SCATTER+source[:,None].to(tl.int64)*N+columns[None,:],combined,
                     (rows[:,None]<M)&(source[:,None]<ROWS)&(columns[None,:]<N))
        else:
            split=tl.permute(tl.reshape(acc,(BM,2,BN//2)),(0,2,1))
            first,second=tl.split(split)
            O.store([mr*BM,nc*BN],first)
            O.store([mr*BM,nc*BN+BN//2],second)
            if MODE==2:
                rows=mr*BM+tl.arange(0,BM);columns=nc*BN+tl.arange(0,BN)
                source=tl.load(SRC+rows,rows<M,other=ROWS)
                gate=tl.load(GATE+source,(rows<M)&(source<ROWS),other=0.).to(tl.bfloat16)
                value=acc.to(tl.bfloat16)*gate[:,None]
                tl.store(SCATTER+source[:,None].to(tl.int64)*N+columns[None,:],value,
                         (rows[:,None]<M)&(source[:,None]<ROWS)&(columns[None,:]<N))
        if G is not None:
            columns=nc*BN+tl.arange(0,BN)
            h=_features(acc,G,expert,columns,N,BM,BN,P,True,CUBIC_ONLY,GS32)
            packed=tl.minimum(tl.maximum(h/tl.load(HS),-448.),448.).to(tl.float8e4nv)
            first,second=tl.split(tl.permute(tl.reshape(packed,(BM,2,BN//2)),(0,2,1)))
            H.store([mr*BM,nc*BN],first);H.store([mr*BM,nc*BN+BN//2],second)
            HT.store([nc*BN,mr*BM],tl.trans(first));HT.store([nc*BN+BN//2,mr*BM],tl.trans(second))
            tl.store(HM+tile,tl.max(tl.max(tl.abs(h),1),0))


def mm(a,w,table,sa,sw,dtype,g=None,prepared=None,h=None,ht=None,hs=None,
       mode=0,SRC=None,GATE=None,SCATTER=None,DLOG=None,ROUTER=None,ROWS=0,cubic_only=False,gs32=True):
    chosen=TILE
    m,k=a.shape;n=w.shape[-1];bm,bn,bk,nw,st=TILES[chosen]
    rows=w.transpose(1,2)
    assert a.is_contiguous() and rows.is_contiguous()
    assert m%bm==0 and n%bn==0 and k%bk==0
    rows=rows.view(-1,k)
    assert mode in (0,2,4) and (mode==0 or g is None)
    out=torch.empty((ROWS if mode==4 else m,n),device=a.device,dtype=dtype)
    ad=TensorDescriptor.from_tensor(a,[bm,bk]);wd=TensorDescriptor.from_tensor(rows,[bn,bk])
    od=None if mode==4 else TensorDescriptor.from_tensor(out,[bm,bn//2])
    if mode==4:SCATTER=out
    sms=torch.cuda.get_device_properties(a.device).multi_processor_count
    hd=htd=maxima=None
    if g is not None:
        hd=TensorDescriptor.from_tensor(h,[bm,bn//2]);htd=TensorDescriptor.from_tensor(ht,[bn//2,bm])
        maxima=torch.empty(tr.cdiv(m,bm)*tr.cdiv(n,bn),device=a.device)
    kernel[(min(sms,tr.cdiv(m,bm)*tr.cdiv(n,bn)),)](ad,wd,od,table,sa,sw,m,n,k,bm,bn,bk,sms,G=g,P=prepared,H=hd,HT=htd,HS=hs,HM=maxima,
        MODE=mode,SRC=SRC,GATE=GATE,SCATTER=SCATTER,DLOG=DLOG,ROUTER=ROUTER,ROWS=ROWS,CUBIC_ONLY=cubic_only,GS32=gs32,FLATTEN=True,num_warps=nw,num_stages=st)
    return out if g is None else (out,maxima)
