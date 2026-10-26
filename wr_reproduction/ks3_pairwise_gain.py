"""Zero-initialized adjacent-coordinate gain on unchanged independent features.

h_new = BF16(h_original * (1 + .5*tanh(raw_i)*tanh(c_(i xor1)/4))).
The factor lies in[.5,1.5]: a relative bound, not an absolute activation cap.
Reuse original geometry/projections, routed autograd, parameter registration,
and optimizer plumbing. The original BF16 coordinate-VJP boundary is retained.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from wr_reproduction import ks3_moe as core, ks3_pairwise as routed

KIND = 'bounded-multiplicative-adjacent-feature-gain-v1'
ORIGINAL_FEATURES = routed.features
ORIGINAL_FEATURE_VJP = routed.feature_vjp
ORIGINAL_ROUTED_FORWARD = routed.forward
forward = ORIGINAL_ROUTED_FORWARD
backward = routed.backward
RoutedGain = routed.RoutedPairwise


def reference(c, h, raw, bound=.5):
    partner = torch.arange(c.shape[-1], device=c.device) ^ 1
    return h * (1 + bound*raw.tanh()*torch.tanh(c.clamp(-65504.,65504.)[..., partner]/4))


def configure(settings):
    expected = dict(kind=KIND, bound=.5, partner_scale=4., lr_mul=.125,
        initial_raw_coefficient=0., placement='after-independent-GS32')
    if settings != expected:
        raise ValueError('Only the frozen bounded feature-gain recipe is covered')
    if routed.features not in (ORIGINAL_FEATURES, features) or routed.feature_vjp not in (ORIGINAL_FEATURE_VJP, feature_vjp):
        raise RuntimeError('Different pair feature overlay already installed')
    if routed.forward not in (ORIGINAL_ROUTED_FORWARD, _routed_forward):
        raise RuntimeError('Different routed forward overlay already installed')
    routed.features = features
    routed.feature_vjp = feature_vjp
    routed.forward = _routed_forward
    routed.configure(settings)
    return expected


def _routed_forward(*args, **kwargs):
    # Resolve dynamically so the sparse diagnostic wrapper sees actual calls.
    return forward(*args, **kwargs)


def baseline_features(c, g, tiles):
    m,n = c.shape; bm = 64 if core.BF16_TUNED_KERNELS else 16
    h = torch.empty_like(c, dtype=torch.bfloat16)
    core._geometry[(tr.cdiv(m,bm),tr.cdiv(n,32))](
        c,g,h,tiles,m,n,BM=bm,RANK=1,GS32=True,num_warps=4)
    return h


@tr.jit
def _gain_features(C,RAW,BASE,H,TILES,M:tl.constexpr,N:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    other=tl.load(C+row[:,None]*N+(col[None,:]^1),mask,other=0.)
    u=libdevice.tanh(tl.minimum(tl.maximum(other,-65504.),65504.)*.25)
    a=libdevice.tanh(tl.load(RAW+expert*N+col,col<N,other=0.))
    h=tl.load(BASE+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(H+row[:,None]*N+col[None,:],h*(1.+BOUND*a[None,:]*u),mask)


@tr.jit
def _prepare_vjp(V,C,RAW,BASE,WV,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,
                 ROWS:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr=64):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    real=tl.load(SRC+row,row<M,other=ROWS)<ROWS
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    other=tl.load(C+row[:,None]*N+(col[None,:]^1),mask,other=0.)
    u=libdevice.tanh(tl.minimum(tl.maximum(other,-65504.),65504.)*.25)
    a=libdevice.tanh(tl.load(RAW+expert*N+col,col<N,other=0.))
    v=tl.load(V+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    h=tl.load(BASE+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(WV+row[:,None]*N+col[None,:],v*(1.+BOUND*a[None,:]*u),mask)
    dp=tl.where(mask&real[:,None],v*h*BOUND*(1.-a[None,:]*a[None,:])*u,0.)
    tl.store(PART+tl.program_id(0)*N+col,tl.sum(dp,0),col<N)


@tr.jit
def _partner_vjp(V,C,RAW,BASE,BASE_DC,DC,TILES,SRC,M:tl.constexpr,N:tl.constexpr,
                 ROWS:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr=64):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    real=tl.load(SRC+row,row<M,other=ROWS)<ROWS
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    original=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    u=libdevice.tanh(tl.minimum(tl.maximum(original,-65504.),65504.)*.25)
    aj=libdevice.tanh(tl.load(RAW+expert*N+(col^1),col<N,other=0.))
    vj=tl.load(V+row[:,None]*N+(col[None,:]^1),mask,other=0.).to(tl.float32)
    hj=tl.load(BASE+row[:,None]*N+(col[None,:]^1),mask,other=0.).to(tl.float32)
    added=vj*hj*BOUND*aj[None,:]*.25*(1.-u*u)
    added=tl.where(mask&real[:,None]&(original>=-65504.)&(original<=65504.),added,0.)
    base=tl.load(BASE_DC+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(DC+row[:,None]*N+col[None,:],base+added,mask)


def features(c,g,raw,tiles,bound,gs32=True):
    if not gs32 or core.KS3_RANK!=1:
        raise ValueError('Feature gain requires original GS32 independent rank1 paths')
    m,n=c.shape;bm=64 if core.BF16_TUNED_KERNELS else 16
    base=baseline_features(c,g,tiles);h=torch.empty_like(base)
    _gain_features[(tr.cdiv(m,bm),tr.cdiv(n,32))](
        c,raw,base,h,tiles,m,n,bound,BM=bm,num_warps=4,enable_fp_fusion=False)
    return h


def feature_vjp(v,c,g,raw,tiles,src,offs,rows,bound):
    m,n=c.shape;e=g.shape[0];base=baseline_features(c,g,tiles)
    wv=torch.empty_like(v);pair_part=torch.empty((tr.cdiv(m,64),n),device=c.device,dtype=torch.float32)
    _prepare_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,raw,base,wv,pair_part,
        tiles,src,m,n,rows,bound,num_warps=4,enable_fp_fusion=False)
    base_dc=torch.empty_like(c,dtype=torch.bfloat16)
    part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](
        wv,c,g,base_dc,part,tiles,src,m,n,rows,RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    dc=torch.empty_like(c,dtype=torch.bfloat16)
    _partner_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,raw,base,base_dc,dc,
        tiles,src,m,n,rows,bound,num_warps=4,enable_fp_fusion=False)
    dp=torch.empty_like(raw)
    routed._reduce_pair[(e,tr.cdiv(n,32))](pair_part,dp,offs,n,pair_part.shape[0],
        tr.next_power_of_2(pair_part.shape[0]),num_warps=4)
    return dc,dg,dp


def install(base_install,settings,*args,**kwargs):
    configure(settings)
    return routed.install(base_install,settings,*args,**kwargs)
