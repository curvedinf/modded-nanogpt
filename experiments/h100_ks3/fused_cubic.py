"""Experimental Hopper FP8 product fused with the rank-one cubic VJP.

The product's FP32 accumulator feeds the original straight-through cubic
derivative directly, avoiding a full FP32 V temporary. All geometry gradients
and both reusable FP8 dC layouts are produced inside the same kernel.
"""
import os
import torch,triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from experiments.h100_ks3.native_geometry import native_tanh


@tr.jit
def kernel(A,W,C,P,DC,DCT,PART,AMAX,TABLE,SRC,SA,SW,DS,
           M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,ROWS:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SMS:tl.constexpr,
           COMPONENTS:tl.constexpr,FLATTEN:tl.constexpr,MOMENTS:tl.constexpr):
    ntiles=tl.cdiv(N,BN);tiles=tl.cdiv(M,BM)*ntiles
    scale_a=tl.load(SA);scale_dc=tl.load(DS)
    for tile in tl.range(tl.program_id(0),tiles,SMS,flatten=FLATTEN):
        mr=tile//ntiles;nc=tile%ntiles
        expert=tl.load(TABLE+mr*BM//128)
        value=tl.zeros((BM,BN),tl.float32)
        for k in range(tl.cdiv(K,BK)):
            a=A.load([mr*BM,k*BK]);w=W.load([expert*N+nc*BN,k*BK])
            value=tl.dot(a,w.T,value)
        value*=scale_a*tl.load(SW+expert)
        rows=mr*BM+tl.arange(0,BM);cols=nc*BN+tl.arange(0,BN)
        real=tl.load(SRC+rows,rows<M,other=ROWS)<ROWS
        original=tl.load(C+rows[:,None]*N+cols[None,:])
        x=tl.minimum(tl.maximum(original,-65504.),65504.)
        pp=P+(expert*N+cols)*9
        ps=tl.load(pp+6)
        c0=tl.load(pp+0)*ps;c1=tl.load(pp+1)*ps
        c2=tl.load(pp+2)*ps;c3=tl.load(pp+3)*ps
        t=4.*native_tanh(x*.25)
        cubic=2*c0-2*c1+c2+c3;quadratic=-3*c0+3*c1-2*c2-c3
        spine=((cubic*t+quadratic)*t+c2)*t+c0
        dp=tl.where(real[:,None],value*2.*tl.maximum(x+spine,0.),0.)
        dm=dp*((3*cubic*t+2*quadratic)*t+c2)*(1.-t*t*.0625)
        dc=tl.where((original>=-65504.)&(original<=65504.),dp+dm,0.)
        out=tl.minimum(tl.maximum(dc/scale_dc,-57344.),57344.).to(tl.float8e5)
        first,second=tl.split(tl.permute(tl.reshape(out,(BM,2,BN//2)),(0,2,1)))
        DC.store([mr*BM,nc*BN],first);DC.store([mr*BM,nc*BN+BN//2],second)
        DCT.store([nc*BN,mr*BM],tl.trans(first));DCT.store([nc*BN+BN//2,mr*BM],tl.trans(second))
        tl.store(AMAX+tile,tl.max(tl.max(tl.abs(dc),1),0))
        off=(mr*N+cols)*COMPONENTS
        if MOMENTS:
            # Reduce four monomial moments before applying the Hermite basis.
            # Only column vectors remain live for the final basis transform.
            s0=tl.sum(dp,0)
            weighted=dp*t;s1=tl.sum(weighted,0)
            weighted=weighted*t;s2=tl.sum(weighted,0)
            weighted=weighted*t;s3=tl.sum(weighted,0)
            tl.store(PART+off,2*s3-3*s2+s0)
            tl.store(PART+off+1,-2*s3+3*s2)
            tl.store(PART+off+2,s3-2*s2+s1)
            tl.store(PART+off+3,s3-s2)
        else:
            t2=t*t;t3=t2*t
            tl.store(PART+off,tl.sum(dp*(2*t3-3*t2+1),0))
            tl.store(PART+off+1,tl.sum(dp*(-2*t3+3*t2),0))
            tl.store(PART+off+2,tl.sum(dp*(t3-2*t2+t),0))
            tl.store(PART+off+3,tl.sum(dp*(t3-t2),0))
        if COMPONENTS==8:
            tl.store(PART+off+4,0.);tl.store(PART+off+5,0.)
            tl.store(PART+off+6,0.);tl.store(PART+off+7,0.)


def run(a,w,c,prepared,table,src,sa,sw,ds,rows,compact=False):
    name=os.environ.get('KS3_FP8_CUBIC_VJP_TILE','64x128')
    bm,bn,nw={'64x64':(64,64,8),'64x128':(64,128,4),'128x128':(128,128,8),'64x256':(64,256,8)}[name]
    nw=int(os.environ.get('KS3_FP8_CUBIC_VJP_WARPS',nw))
    stages=int(os.environ.get('KS3_FP8_CUBIC_VJP_STAGES','3'))
    flatten=os.environ.get('KS3_FP8_CUBIC_VJP_FLATTEN','1')=='1'
    moments=os.environ.get('KS3_FP8_CUBIC_VJP_MOMENTS','0')=='1'
    if nw not in (4,8) or stages not in (1,2,3):raise ValueError('Invalid fused cubic launch')
    bk=128;m,k=a.shape;n=w.shape[-1];weight=w.transpose(1,2)
    assert a.is_contiguous() and weight.is_contiguous()
    assert m%bm==0 and n%bn==0 and k%bk==0
    dc=torch.empty(m,n,device=a.device,dtype=torch.float8_e5m2)
    dct=torch.empty(n,m,device=a.device,dtype=dc.dtype)
    part=torch.empty(m//bm,n,4 if compact else 8,device=a.device)
    maxima=torch.empty(m//bm*(n//bn),device=a.device)
    ad=TensorDescriptor.from_tensor(a,[bm,bk])
    wd=TensorDescriptor.from_tensor(weight.view(-1,k),[bn,bk])
    cd=TensorDescriptor.from_tensor(dc,[bm,bn//2])
    td=TensorDescriptor.from_tensor(dct,[bn//2,bm])
    sms=torch.cuda.get_device_properties(a.device).multi_processor_count
    kernel[(min(sms,m//bm*(n//bn)),)](ad,wd,c,prepared,cd,td,part,maxima,table,src,sa,sw,ds,
        m,n,k,rows,bm,bn,bk,sms,4 if compact else 8,flatten,moments,num_warps=nw,num_stages=stages)
    return dc,dct,part,maxima,bm
