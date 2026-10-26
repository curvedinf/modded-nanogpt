"""Fuse the tested gain's independent-feature producer with its consumers.

Retain the intermediate BF16 rounding, original geometry backward and all
optimizer/gradient storage boundaries. This changes kernel placement only.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale, _spine_features
from wr_reproduction import ks3_moe as core, ks3_pairwise_gain as gain

ORIGINAL_FEATURES = gain.features
ORIGINAL_VJP = gain.feature_vjp
EXECUTED = set()


@tr.jit
def _mul_rn(x,y):
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2",constraints="=v,v,v",
        args=[x,y],dtype=tl.float32,is_pure=True,pack=1)


@tr.jit
def _add_rn(x,y):
    return tl.inline_asm_elementwise("v_add_f32 $0, $1, $2",constraints="=v,v,v",
        args=[x,y],dtype=tl.float32,is_pure=True,pack=1)


@tr.jit
def _sub_rn(x,y):
    return tl.inline_asm_elementwise("v_sub_f32 $0, $1, $2",constraints="=v,v,v",
        args=[x,y],dtype=tl.float32,is_pure=True,pack=1)


@tr.jit
def _independent(C,G,TILES,M:tl.constexpr,N:tl.constexpr,BM:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    e=tl.load(TILES+tl.program_id(0)*BM//128);valid=cols<N
    gp=G+(e*N+cols)*8
    c0=tl.load(gp+0,valid,other=0.);c1=tl.load(gp+1,valid,other=0.)
    c2=tl.load(gp+2,valid,other=0.);c3=tl.load(gp+3,valid,other=0.)
    c4=tl.load(gp+4,valid,other=0.);c5=tl.load(gp+5,valid,other=0.)
    amax=tl.maximum(tl.maximum(tl.abs(c0),tl.abs(c1)),tl.maximum(
        tl.maximum(tl.abs(c2),tl.abs(c3)),tl.maximum(tl.abs(c4),tl.abs(c5))))
    ps=_half_scale(amax)
    c0=libdevice.nearbyint(c0/ps);c1=libdevice.nearbyint(c1/ps)
    c2=libdevice.nearbyint(c2/ps);c3=libdevice.nearbyint(c3/ps)
    c4=libdevice.nearbyint(c4/ps);c5=libdevice.nearbyint(c5/ps)
    c0=tl.minimum(tl.maximum(c0,-127.),127.);c1=tl.minimum(tl.maximum(c1,-127.),127.)
    c2=tl.minimum(tl.maximum(c2,-127.),127.);c3=tl.minimum(tl.maximum(c3,-127.),127.)
    c4=tl.minimum(tl.maximum(c4,-127.),127.);c5=tl.minimum(tl.maximum(c5,-127.),127.)
    raw=tl.load(gp+6,valid,other=0.);rh=tl.load(gp+7,valid,other=0.)
    scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4
    shift=.5*libdevice.tanh(rh)
    x=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    h=_spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    hs=_half_scale(tl.max(tl.abs(h),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.)
    # Preserve the original separate producer's BF16 storage boundary.
    return (q*hs[:,None]).to(tl.bfloat16).to(tl.float32)


@tr.jit
def _gain(C,RAW,TILES,M:tl.constexpr,N:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    other=tl.load(C+row[:,None]*N+(col[None,:]^1),mask,other=0.)
    u=libdevice.tanh(tl.minimum(tl.maximum(other,-65504.),65504.)*.25)
    a=libdevice.tanh(tl.load(RAW+expert*N+col,col<N,other=0.))
    # Preserve the original gain consumer's unfused elementary operations.
    factor=_add_rn(1.,_mul_rn(_mul_rn(BOUND,a[None,:]),u))
    return factor,a,u


@tr.jit
def _features(C,G,RAW,H,TILES,M:tl.constexpr,N:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    h=_independent(C,G,TILES,M,N,BM)
    factor,a,u=_gain(C,RAW,TILES,M,N,BOUND,BM)
    tl.store(H+row[:,None]*N+col[None,:],_mul_rn(h,factor),(row[:,None]<M)&(col[None,:]<N))


@tr.jit
def _prepare(V,C,G,RAW,BASE,WV,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,
             ROWS:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr=64):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    real=tl.load(SRC+row,row<M,other=ROWS)<ROWS
    h=_independent(C,G,TILES,M,N,BM)
    factor,a,u=_gain(C,RAW,TILES,M,N,BOUND,BM)
    v=tl.load(V+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(BASE+row[:,None]*N+col[None,:],h,mask)
    tl.store(WV+row[:,None]*N+col[None,:],_mul_rn(v,factor),mask)
    dp=_mul_rn(_mul_rn(v,h),BOUND)
    dp=_mul_rn(dp,_sub_rn(1.,_mul_rn(a[None,:],a[None,:])))
    dp=tl.where(mask&real[:,None],_mul_rn(dp,u),0.)
    tl.store(PART+tl.program_id(0)*N+col,tl.sum(dp,0),col<N)


def features(c,g,raw,tiles,bound,gs32=True):
    if not gs32 or core.KS3_RANK!=1:
        raise ValueError('Fused gain requires the frozen GS32 rank1 function')
    m,n=c.shape;bm=64 if core.BF16_TUNED_KERNELS else 16
    h=torch.empty_like(c,dtype=torch.bfloat16)
    _features[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,raw,h,tiles,m,n,bound,BM=bm,num_warps=4,enable_fp_fusion=False)
    EXECUTED.add('fused_features')
    return h


def feature_vjp(v,c,g,raw,tiles,src,offs,rows,bound):
    m,n=c.shape;e=g.shape[0]
    base=torch.empty_like(c,dtype=torch.bfloat16);wv=torch.empty_like(v)
    pair_part=torch.empty((tr.cdiv(m,64),n),device=c.device,dtype=torch.float32)
    _prepare[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,raw,base,wv,pair_part,
        tiles,src,m,n,rows,bound,num_warps=4,enable_fp_fusion=False)
    base_dc=torch.empty_like(c,dtype=torch.bfloat16)
    part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](
        wv,c,g,base_dc,part,tiles,src,m,n,rows,RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    dc=torch.empty_like(c,dtype=torch.bfloat16)
    gain._partner_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,raw,base,base_dc,dc,
        tiles,src,m,n,rows,bound,num_warps=4,enable_fp_fusion=False)
    dp=torch.empty_like(raw)
    gain.routed._reduce_pair[(e,tr.cdiv(n,32))](pair_part,dp,offs,n,pair_part.shape[0],
        tr.next_power_of_2(pair_part.shape[0]),num_warps=4)
    EXECUTED.add('fused_baseline_prepare')
    return dc,dg,dp


def install():
    if gain.features not in (ORIGINAL_FEATURES,features) or gain.feature_vjp not in (ORIGINAL_VJP,feature_vjp):
        raise RuntimeError('Another gain producer overlay is installed')
    gain.features=features;gain.feature_vjp=feature_vjp
