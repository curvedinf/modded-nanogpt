"""H100 KS3 GEMMs and fused epilogues; GS32/BF16/FP16 recipe unchanged."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale, _spine_features
from experiments.h100_ks3.native_geometry import _spine_features as _native_spine_features

@tr.jit
def _features(x,G,e,cols,N:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,PREPARED=None,NATIVE_TANH:tl.constexpr=False):
    valid=cols<N;gp=G+(e*N+cols)*8
    if PREPARED is not None:
        pp=PREPARED+(e*N+cols)*9
        c0=tl.load(pp,valid,other=0.);c1=tl.load(pp+1,valid,other=0.)
        c2=tl.load(pp+2,valid,other=0.);c3=tl.load(pp+3,valid,other=0.)
        c4=tl.load(pp+4,valid,other=0.);c5=tl.load(pp+5,valid,other=0.)
        ps=tl.load(pp+6,valid,other=1.);scale=tl.load(pp+7,valid,other=1.);shift=tl.load(pp+8,valid,other=0.)
    else:
        c0=tl.load(gp,valid,other=0.);c1=tl.load(gp+1,valid,other=0.)
        c2=tl.load(gp+2,valid,other=0.);c3=tl.load(gp+3,valid,other=0.)
        c4=tl.load(gp+4,valid,other=0.);c5=tl.load(gp+5,valid,other=0.)
        amax=tl.maximum(tl.maximum(tl.abs(c0),tl.abs(c1)),tl.maximum(tl.maximum(tl.abs(c2),tl.abs(c3)),tl.maximum(tl.abs(c4),tl.abs(c5))))
        ps=_half_scale(amax)
        c0=tl.minimum(tl.maximum(libdevice.nearbyint(c0/ps),-127.),127.)
        c1=tl.minimum(tl.maximum(libdevice.nearbyint(c1/ps),-127.),127.)
        c2=tl.minimum(tl.maximum(libdevice.nearbyint(c2/ps),-127.),127.)
        c3=tl.minimum(tl.maximum(libdevice.nearbyint(c3/ps),-127.),127.)
        c4=tl.minimum(tl.maximum(libdevice.nearbyint(c4/ps),-127.),127.)
        c5=tl.minimum(tl.maximum(libdevice.nearbyint(c5/ps),-127.),127.)
        raw=tl.load(gp+6,valid,other=0.);rh=tl.load(gp+7,valid,other=0.)
        scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4
        shift=.5*libdevice.tanh(rh)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    if NATIVE_TANH:h=_native_spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    else:h=_spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    groups=tl.reshape(h,(BM,BN//32,32))
    hs=_half_scale(tl.max(tl.abs(groups),2))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(groups/hs[:,:,None]),-127.),127.)
    return tl.reshape(q*hs[:,:,None],(BM,BN))

@tr.jit
def gemm(A,W,O,TILES,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
         W0:tl.constexpr,W1:tl.constexpr,W2:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,MODE:tl.constexpr=0,
         G=None,H=None,SRC=None,GATE=None,SCATTER=None,ROWS:tl.constexpr=0,C=None,PART=None,PACK_INPUT:tl.constexpr=False,XPACK=None,DLOG=None,ROUTER=None,PREPARED=None,ASCALE=None,WSCALE=None,NATIVE_TANH:tl.constexpr=False):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);e=tl.load(TILES+tl.program_id(0)*BM//128)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        k=start*BK+kk
        if PACK_INPUT:
            source=tl.load(SRC+rows,rows<M,other=ROWS)
            a=tl.load(A+source[:,None].to(tl.int64)*K+k[None,:],(rows[:,None]<M)&(source[:,None]<ROWS)&(k[None,:]<K),other=0.).to(tl.float32)
            groups=tl.reshape(a,(BM,BK//32,32));scales=_half_scale(tl.max(tl.abs(groups),2))
            a=tl.reshape(tl.minimum(tl.maximum(libdevice.nearbyint(groups/scales[:,:,None]),-127.),127.)*scales[:,:,None],(BM,BK)).to(tl.bfloat16)
            if tl.program_id(1)==0:tl.store(XPACK+rows[:,None]*K+k[None,:],a,(rows[:,None]<M)&(k[None,:]<K))
        else:
            a=tl.load(A+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0.)
        w=tl.load(W+e*W0+k[:,None]*W1+cols[None,:]*W2,(k[:,None]<K)&(cols[None,:]<N),other=0.)
        acc=tl.dot(a,w,acc)
    if ASCALE is not None:acc=acc*tl.load(ASCALE)*tl.load(WSCALE+e)
    mask=(rows[:,None]<M)&(cols[None,:]<N)
    if MODE==3:
        _vjp_epilogue(acc,C,G,O,PART,SRC,rows,cols,mask,e,M,N,ROWS,BM)
    elif MODE==4:
        source=tl.load(SRC+rows,rows<M,other=ROWS)
        combined=acc.to(tl.bfloat16).to(tl.float32)
        for expert in tl.static_range(8):
            dl=tl.load(DLOG+source*8+expert,(rows<M)&(source<ROWS),other=0.).to(tl.float32)
            router=tl.load(ROUTER+expert*N+cols,cols<N,other=0.).to(tl.float32)
            combined+=dl[:,None]*router[None,:]
        tl.store(O+source[:,None].to(tl.int64)*N+cols[None,:],combined,mask&(source[:,None]<ROWS))
    else:
        tl.store(O+rows[:,None]*N+cols[None,:],acc,mask)
        if MODE==1:
            h=_features(acc,G,e,cols,N,BM,BN,PREPARED,NATIVE_TANH)
            tl.store(H+rows[:,None]*N+cols[None,:],h,mask)
        elif MODE==2:
            source=tl.load(SRC+rows,rows<M,other=ROWS)
            gate=tl.load(GATE+source,(rows<M)&(source<ROWS),other=0.).to(tl.bfloat16)
            value=acc.to(tl.bfloat16)*gate[:,None]
            tl.store(SCATTER+source[:,None].to(tl.int64)*N+cols[None,:],value,mask&(source[:,None]<ROWS))

@tr.jit
def dw_split(A,B,PART,OFFS,H:tl.constexpr,D:tl.constexpr,SPLIT:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    z=tl.program_id(2);e=z//SPLIT;s=z%SPLIT
    lo=0
    if e>0:lo=tl.load(OFFS+e-1)
    hi=tl.load(OFFS+e)
    chunks=tl.cdiv(hi-lo,BK*SPLIT)
    start=lo+s*chunks*BK;end=tl.minimum(hi,start+chunks*BK)
    h=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for base in range(start,end,BK):
        r=base+kk
        a=tl.load(A+r[None,:]*H+h[:,None],(r[None,:]<end)&(h[:,None]<H),other=0.)
        b=tl.load(B+r[:,None]*D+d[None,:],(r[:,None]<end)&(d[None,:]<D),other=0.)
        acc=tl.dot(a,b,acc)
    tl.store(PART+(e*SPLIT+s)*H*D+h[:,None]*D+d[None,:],acc,(h[:,None]<H)&(d[None,:]<D))

@tr.jit
def dw_reduce(P,O,SPLIT:tl.constexpr,SIZE:tl.constexpr,B:tl.constexpr=1024):
    e=tl.program_id(1);idx=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.full((B,),0,tl.float32)
    for s in range(SPLIT):acc+=tl.load(P+(e*SPLIT+s)*SIZE+idx,idx<SIZE,other=0.)
    tl.store(O+e*SIZE+idx,acc,idx<SIZE)

@tr.jit
def _vjp_epilogue(value,C,G,DC,PART,SRC,rows,cols,mask,e,M:tl.constexpr,N:tl.constexpr,ROWS:tl.constexpr,BM:tl.constexpr):
    real=tl.load(SRC+rows,rows<M,other=ROWS)<ROWS
    original=tl.load(C+rows[:,None]*N+cols[None,:],mask,other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    base=(e*(N)+cols)*(6+2)
    gp=G+base+0
    c0=tl.load(gp,cols<N,other=0.);c1=tl.load(gp+1,cols<N,other=0.);c2=tl.load(gp+2,cols<N,other=0.)
    c3=tl.load(gp+3,cols<N,other=0.);c4=tl.load(gp+4,cols<N,other=0.);c5=tl.load(gp+5,cols<N,other=0.)
    amax=tl.maximum(tl.maximum(tl.abs(c0),tl.abs(c1)),tl.maximum(tl.maximum(tl.abs(c2),tl.abs(c3)),tl.maximum(tl.abs(c4),tl.abs(c5))))
    ps=_half_scale(amax)
    c0=tl.minimum(tl.maximum(libdevice.nearbyint(c0/ps),-127.),127.)*ps
    c1=tl.minimum(tl.maximum(libdevice.nearbyint(c1/ps),-127.),127.)*ps
    c2=tl.minimum(tl.maximum(libdevice.nearbyint(c2/ps),-127.),127.)*ps
    c3=tl.minimum(tl.maximum(libdevice.nearbyint(c3/ps),-127.),127.)*ps
    c4=tl.minimum(tl.maximum(libdevice.nearbyint(c4/ps),-127.),127.)*ps
    c5=tl.minimum(tl.maximum(libdevice.nearbyint(c5/ps),-127.),127.)*ps
    rs=tl.load(G+base+6,cols<N,other=0.);rh=tl.load(G+base+6+1,cols<N,other=0.)
    scale=tl.where(rs>20.,rs,libdevice.log1p(libdevice.exp(rs)))+1e-4;shift=.5*libdevice.tanh(rh)
    t=4.*libdevice.tanh(x*.25);z=libdevice.tanh(scale[None,:]*(t+shift[None,:]))
    cubic=2*c0-2*c1+c2+c3;quadratic=-3*c0+3*c1-2*c2-c3
    spine=((cubic*t+quadratic)*t+c2)*t+c0;warped=((4*c5*z+2*c4)*z-3*c5)*z-c4
    dp=tl.where(mask&real[:,None],value*2.*tl.maximum(x+spine+warped,0.),0.)
    dz=((12*c5*z+4*c4)*z-3*c5)*(1.-z*z)
    dm=dp*((3*cubic*t+2*quadratic)*t+c2+dz*scale)*(1.-t*t*.0625)
    dc=tl.where((original>=-65504.)&(original<=65504.),dp+dm,0.)
    tl.store(DC+rows[:,None]*N+cols[None,:],dc,mask)
    off=(tl.program_id(0)*N+cols)*8;t2=t*t;t3=t2*t
    tl.store(PART+off,tl.sum(dp*(2*t3-3*t2+1),0),cols<N)
    tl.store(PART+off+1,tl.sum(dp*(-2*t3+3*t2),0),cols<N)
    tl.store(PART+off+2,tl.sum(dp*(t3-2*t2+t),0),cols<N)
    tl.store(PART+off+3,tl.sum(dp*(t3-t2),0),cols<N)
    tl.store(PART+off+4,tl.sum(dp*(2*z*z-1),0),cols<N)
    tl.store(PART+off+5,tl.sum(dp*(4*z*z*z-3*z),0),cols<N)
    tl.store(PART+off+6,tl.sum(dp*dz*(t+shift)*tl.sigmoid(rs),0),cols<N)
    tl.store(PART+off+7,tl.sum(dp*dz*scale*.5*(1.-libdevice.tanh(rh)*libdevice.tanh(rh)),0),cols<N)

@tr.jit
def gather_gate(G,Y,SRC,SCALE,GG,DS,ROWS:tl.constexpr,M:tl.constexpr,D:tl.constexpr=768,BM:tl.constexpr=4):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.arange(0,1024)
    source=tl.load(SRC+rows,rows<M,other=ROWS);valid=(rows<M)&(source<ROWS)
    grad=tl.load(G+source[:,None].to(tl.int64)*D+cols[None,:],valid[:,None]&(cols[None,:]<D),other=0.).to(tl.float32)
    y=tl.load(Y+rows[:,None]*D+cols[None,:],valid[:,None]&(cols[None,:]<D),other=0.).to(tl.float32)
    scale=tl.load(SCALE+source,valid,other=0.).to(tl.bfloat16).to(tl.float32)
    tl.store(GG+rows[:,None]*D+cols[None,:],grad*scale[:,None],(rows[:,None]<M)&(cols[None,:]<D))
    tl.store(DS+rows,tl.sum(grad*y,1),rows<M)
