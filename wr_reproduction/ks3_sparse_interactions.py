"""Three bounded cross-coordinate terms per original independent KS3 chart.

h_i = BF16(original_GS32_BF16_h_i + sum_k tanh(raw_ik)*u(c_i)*u(c_(i xor2^k))))
u(c)=tanh(clamp(c,-65504,65504)/4), k=0,1,2.
All independent rank-one pathways survive. Three zero coefficients per chart
connect groups of eight coordinates; no coordinate averaging or rescaling.
Each new term is bounded by one, so the added sum is bounded by three.
Original routed dispatch/projections and BF16 base coordinate VJP are reused.
"""
import math
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from wr_reproduction import ks3_moe as core, ks3_pairwise as routed

KIND = 'bounded-sparse-three-partner-interactions-v1'
ORIGINAL_FEATURES = routed.features
ORIGINAL_FEATURE_VJP = routed.feature_vjp
ORIGINAL_ROUTED_FORWARD = routed.forward
forward = ORIGINAL_ROUTED_FORWARD
backward = routed.backward
RoutedSparseInteractions = routed.RoutedPairwise


def reference(c, h, raw, bound=1.):
    u=(c.clamp(-65504.,65504.)/4.).tanh()
    indices=torch.arange(c.shape[-1],device=c.device)
    terms=[bound*raw[...,k].tanh()*u*u[...,indices^offset]
           for k,offset in enumerate((1,2,4))]
    return h+torch.stack(terms).sum(0)


def configure(settings):
    expected = dict(kind=KIND, bound=1., coordinate_scale=4., partners=[1,2,4], lr_mul=.125,
        initial_raw_coefficient=0., placement='after-independent-GS32')
    if settings != expected:
        raise ValueError('Only the frozen three-partner interaction recipe is covered')
    if routed.features not in (ORIGINAL_FEATURES, features) or routed.feature_vjp not in (ORIGINAL_FEATURE_VJP, feature_vjp):
        raise RuntimeError('Different routed feature overlay already installed')
    if routed.forward not in (ORIGINAL_ROUTED_FORWARD, _routed_forward):
        raise RuntimeError('Different routed forward overlay already installed')
    routed.features = features
    routed.feature_vjp = feature_vjp
    routed.forward = _routed_forward
    routed.configure(settings)
    return expected


def _routed_forward(*args, **kwargs):
    return forward(*args, **kwargs)


@tr.jit
def _basis(x):
    t = libdevice.tanh(tl.minimum(tl.maximum(x, -65504.), 65504.)/4.)
    dt = tl.where((x >= -65504.) & (x <= 65504.), (1.-t*t)/4., 0.)
    return t, dt


@tr.jit
def _add_features(C, RAW, BASE, H, TILES, M:tl.constexpr, N:tl.constexpr,
                  BOUND:tl.constexpr, BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    x=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    u,du=_basis(x)
    added=tl.full((BM,32),0.,tl.float32)
    for k in tl.static_range(3):
        partner=col^(1<<k)
        xj=tl.load(C+row[:,None]*N+partner[None,:],mask,other=0.)
        uj,duj=_basis(xj)
        a=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+k,col<N,other=0.))
        added+=BOUND*a[None,:]*u*uj
    base=tl.load(BASE+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(H+row[:,None]*N+col[None,:],base+added,mask)


@tr.jit
def _add_vjp(V,C,RAW,BASE_DC,DC,PART,TILES,SRC,
             M:tl.constexpr,N:tl.constexpr,ROWS:tl.constexpr,
             BOUND:tl.constexpr,BM:tl.constexpr=64):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    real=tl.load(SRC+row,row<M,other=ROWS)<ROWS
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    x=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    u,du=_basis(x)
    v=tl.load(V+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    added=tl.full((BM,32),0.,tl.float32)
    for k in tl.static_range(3):
        partner=col^(1<<k)
        xj=tl.load(C+row[:,None]*N+partner[None,:],mask,other=0.)
        uj,duj=_basis(xj)
        a=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+k,col<N,other=0.))
        aj=libdevice.tanh(tl.load(RAW+(expert*N+partner)*3+k,col<N,other=0.))
        vj=tl.load(V+row[:,None]*N+partner[None,:],mask,other=0.).to(tl.float32)
        added+=BOUND*(v*a[None,:]+vj*aj[None,:])*du*uj
        dp=tl.where(mask&real[:,None],v*BOUND*(1.-a[None,:]*a[None,:])*u*uj,0.)
        tl.store(PART+(tl.program_id(0)*N+col)*3+k,tl.sum(dp,0),col<N)
    base=tl.load(BASE_DC+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(DC+row[:,None]*N+col[None,:],base+tl.where(mask&real[:,None],added,0.),mask)


@tr.jit
def _reduce_interactions(P,O,OFFS,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BM:tl.constexpr=64):
    e=tl.program_id(0);col=tl.program_id(1)*32+tl.arange(0,32)
    lo=0
    if e>0:lo=tl.load(OFFS+e-1)//BM
    hi=tl.load(OFFS+e)//BM
    rows=tl.arange(0,B)
    for k in tl.static_range(3):
        v=tl.load(P+(rows[:,None]*N+col[None,:])*3+k,
                  (rows[:,None]>=lo)&(rows[:,None]<hi)&(rows[:,None]<T)&(col[None,:]<N),other=0.)
        tl.store(O+(e*N+col)*3+k,tl.sum(v,0),col<N)


def features(c, g, raw, tiles, bound, gs32=True):
    if not gs32 or core.KS3_RANK != 1:
        raise ValueError('Sparse interactions require original GS32 independent rank1 paths')
    m, n = c.shape
    if n%8 or raw.shape!=(*g.shape[:-1],3):raise ValueError('Expected three partners per chart in groups of eight')
    bm = 64 if core.BF16_TUNED_KERNELS else 16
    base = torch.empty_like(c, dtype=torch.bfloat16)
    core._geometry[(tr.cdiv(m, bm), tr.cdiv(n, 32))](
        c, g, base, tiles, m, n, BM=bm, RANK=1, GS32=True, num_warps=4)
    h = torch.empty_like(base)
    _add_features[(tr.cdiv(m, bm), tr.cdiv(n, 32))](
        c, raw, base, h, tiles, m, n, bound, BM=bm, num_warps=4)
    return h


def feature_vjp(v, c, g, raw, tiles, src, offs, rows, bound):
    m, n = c.shape
    e = g.shape[0]
    # Original storage boundary is essential for bitwise zero-init gradients.
    base_dc = torch.empty_like(c, dtype=torch.bfloat16)
    part = torch.empty((tr.cdiv(m, 64), n, 8), device=c.device, dtype=torch.float32)
    core._vjp[(tr.cdiv(m, 64), tr.cdiv(n, 32))](
        v, c, g, base_dc, part, tiles, src, m, n, rows, RANK=1, num_warps=4)
    dg = torch.empty_like(g)
    core._reduce[(e, n)](part, dg, offs, n, part.shape[0],
        tr.next_power_of_2(part.shape[0]), RANK=1, num_warps=4)
    dc = torch.empty_like(c, dtype=torch.bfloat16)
    raw_part = torch.empty((tr.cdiv(m, 64), n, 3), device=c.device, dtype=torch.float32)
    _add_vjp[(tr.cdiv(m, 64), tr.cdiv(n, 32))](
        v, c, raw, base_dc, dc, raw_part, tiles, src, m, n, rows, bound, num_warps=4)
    dp = torch.empty_like(raw)
    _reduce_interactions[(e, tr.cdiv(n, 32))](raw_part, dp, offs, n, raw_part.shape[0],
        tr.next_power_of_2(raw_part.shape[0]), num_warps=4)
    return dc, dg, dp


def install(base_install,settings,*args,**kwargs):
    configure(settings)
    bound=settings['bound'];lr=settings['lr_mul']
    if not math.isfinite(bound) or not 0<bound<=4 or not math.isfinite(lr) or lr<=0:
        raise ValueError('Invalid sparse interaction recipe')
    cls=base_install(*args,**kwargs)
    from track_1_short import training
    from track_1_short.model.moe import MOE_INDEX
    class SparseInteractionGPT(cls):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if self.ks3_subspace_rank!=1 or self.ks3_precision!='bf16':
                raise ValueError('Sparse interaction extension requires independent rank1/BF16 experts')
            self.ks3_interaction=torch.nn.Parameter(torch.zeros((*self.ks3_geometry.shape[:-1],3),device=self.device))
            self.ks3_interaction.label='ks3_interaction'
            self.ks3_interaction_bound=bound

        def _moe(self,layer,normed,weights,post):
            k=MOE_INDEX[layer];w1,w2,router=weights[k];rows=normed.reshape(-1,normed.shape[-1])
            if post is None:post=rows.new_ones(())
            if self.training:
                out,probs,counts=RoutedSparseInteractions.apply(rows,router,post,w1,w2,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.ks3_interaction[k],self.moe_bank_grad,k,bound,self.ks3_routed_training_gs32)
            else:
                from track_1_short.perf.kernels import moe_mlp as moe
                probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1);gate,expert=probs.max(-1)
                dest,src,offs,tiles,counts=moe.moe_dispatch(expert,8)
                y,_,_,_=forward(rows,src,tiles,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.ks3_interaction[k],bound)
                out=y[dest.long()]*gate[:,None].to(y.dtype)
            return out.reshape(normed.shape),8*(counts.float()/rows.shape[0]*probs.mean(0)).sum()

    original=training.AnvilAndAdam
    def optimizer(*a,**kw):
        kw['param_table']['ks3_interaction']=dict(optim='adam',comms='replicated',adam_betas=[.9,.99],lr_mul=lr,wd_mul=0.)
        kw['scatter_order'].append('ks3_interaction');kw['work_order'].append('ks3_interaction')
        return original(*a,**kw)
    training.AnvilAndAdam=optimizer
    return SparseInteractionGPT
