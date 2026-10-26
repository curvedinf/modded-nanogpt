"""Compact-support local residuals added to unchanged rank-one KS3 features.

Three fixed knots (-2, 0, 2), width 2. Each bounded zero-initialized raw
coefficient multiplies max(1-((c-k)/2)^2, 0)^2. The original geometry producer
and STE are called unchanged, including the original BF16 coordinate-gradient
storage boundary. The added residual follows independent GS32/BF16
packing and precedes the original BF16 output projection. No coordinate mixing.
"""
import math
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from wr_reproduction import ks3_moe as core

KIND = 'additive-compact-quartic-three-knots-v1'


def configure(settings):
    expected = dict(kind=KIND, knots=[-2.,0.,2.], width=2., bound_per_coefficient=1.,
                    bound=1., lr_mul=.125, initial_raw_coefficient=0.,
                    placement='after-independent-GS32')
    if settings != expected:
        raise ValueError('This gate covers exactly the frozen three-knot residual recipe')
    return expected


def reference(c, raw, bound=1.):
    knots=c.new_tensor([-2.,0.,2.])
    u=(c.clamp(-65504.,65504.)[...,None]-knots)/2
    return bound*(raw.tanh()*(1-u.square()).clamp_min(0).square()).sum(-1)


@tr.jit
def _basis(x):
    u0=(x+2.)*.5;u1=x*.5;u2=(x-2.)*.5
    s0=tl.maximum(1.-u0*u0,0.);s1=tl.maximum(1.-u1*u1,0.);s2=tl.maximum(1.-u2*u2,0.)
    return s0*s0,s1*s1,s2*s2,-2.*u0*s0,-2.*u1*s1,-2.*u2*s2


@tr.jit
def _add_features(C,RAW,BASE,H,TILES,M:tl.constexpr,N:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    x=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    p0=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+0,col<N,other=0.))
    p1=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+1,col<N,other=0.))
    p2=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+2,col<N,other=0.))
    b0,b1,b2,d0,d1,d2=_basis(x)
    term=BOUND*(p0[None,:]*b0+p1[None,:]*b1+p2[None,:]*b2)
    h=tl.load(BASE+row[:,None]*N+col[None,:],mask,other=0.).to(tl.float32)
    tl.store(H+row[:,None]*N+col[None,:],h+term,mask)


@tr.jit
def _add_vjp(V,C,RAW,BASE_DC,DC,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,
             ROWS:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr=64):
    row=tl.program_id(0)*BM+tl.arange(0,BM);col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    real=tl.load(SRC+row,row<M,other=ROWS)<ROWS
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    original=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    p0=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+0,col<N,other=0.))
    p1=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+1,col<N,other=0.))
    p2=libdevice.tanh(tl.load(RAW+(expert*N+col)*3+2,col<N,other=0.))
    b0,b1,b2,d0,d1,d2=_basis(x)
    v=tl.load(V+row[:,None]*N+col[None,:],mask,other=0.)
    v=tl.where(mask&real[:,None],v,0.)
    added=v*BOUND*(p0[None,:]*d0+p1[None,:]*d1+p2[None,:]*d2)
    added=tl.where((original>=-65504.)&(original<=65504.),added,0.)
    base=tl.load(BASE_DC+row[:,None]*N+col[None,:],mask,other=0.)
    tl.store(DC+row[:,None]*N+col[None,:],base+added,mask)
    a0=v*BOUND*(1.-p0[None,:]*p0[None,:])*b0
    a1=v*BOUND*(1.-p1[None,:]*p1[None,:])*b1
    a2=v*BOUND*(1.-p2[None,:]*p2[None,:])*b2
    offset=(tl.program_id(0)*N+col)*3
    tl.store(PART+offset+0,tl.sum(a0,0),col<N)
    tl.store(PART+offset+1,tl.sum(a1,0),col<N)
    tl.store(PART+offset+2,tl.sum(a2,0),col<N)


@tr.jit
def _reduce_local(P,O,OFFS,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BM:tl.constexpr=64):
    expert=tl.program_id(0);col=tl.program_id(1)*32+tl.arange(0,32)
    lo=0
    if expert>0:lo=tl.load(OFFS+expert-1)//BM
    hi=tl.load(OFFS+expert)//BM;row=lo+tl.arange(0,B)
    mask=(row[:,None]<hi)&(row[:,None]<T)&(col[None,:]<N)
    for k in tl.static_range(3):
        v=tl.load(P+(row[:,None]*N+col[None,:])*3+k,mask,other=0.)
        tl.store(O+(expert*N+col)*3+k,tl.sum(v,0),col<N)


def features(c,g,local,tiles,bound,gs32=True):
    m,n=c.shape;bm=64 if core.BF16_TUNED_KERNELS else 16
    baseline=torch.empty_like(c,dtype=torch.bfloat16)
    core._geometry[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,baseline,tiles,m,n,BM=bm,RANK=1,GS32=gs32,num_warps=4)
    h=torch.empty_like(baseline)
    _add_features[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,local,baseline,h,tiles,m,n,bound,BM=bm,num_warps=4)
    return h


def feature_vjp(v,c,g,local,tiles,src,offs,rows,bound):
    m,n=c.shape;e=g.shape[0]
    # Keep the original STE output dtype. Promoting this pointer to FP32 can
    # change compiler reassociation and BF16 rounding in the independent path,
    # even when every new coefficient is zero.
    base_dc=torch.empty_like(c,dtype=torch.bfloat16)
    part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,base_dc,part,tiles,src,m,n,rows,RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    dc=torch.empty_like(c,dtype=torch.bfloat16)
    local_part=torch.empty((tr.cdiv(m,64),n,3),device=c.device,dtype=torch.float32)
    _add_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,local,base_dc,dc,local_part,tiles,src,m,n,rows,bound,num_warps=4)
    dp=torch.empty_like(local)
    _reduce_local[(e,tr.cdiv(n,32))](local_part,dp,offs,n,local_part.shape[0],tr.next_power_of_2(local_part.shape[0]),num_warps=4)
    return dc,dg,dp


@torch.library.custom_op('wr_ks3_local::forward',mutates_args=())
def forward(rows:torch.Tensor,src:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,
            down:torch.Tensor,g:torch.Tensor,local:torch.Tensor,bound:float,gs32:bool=True
            )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if core.BF16_TUNED_KERNELS or not gs32:
        m,d=src.numel(),rows.shape[1]
        x=torch.empty((m,d),device=rows.device,dtype=torch.bfloat16)
        core._pack[(tr.cdiv(m,128),tr.cdiv(d,32))](rows,x,m,d,*rows.stride(),
            SRC=src,INPUT_ROWS=rows.shape[0],ROWS_PER_TILE=128,GS32=gs32,num_warps=4)
    else:
        x=core.pack(torch.where((src<rows.shape[0])[:,None],rows[src.clamp(0,rows.shape[0]-1).long()],0).contiguous())
    c=core.mm(x,up.transpose(1,2),tiles)
    h=features(c,g,local,tiles,bound,gs32)
    return core.mm(h,down.transpose(1,2),tiles,dtype=rows.dtype),x,c,h


@forward.register_fake
def _(rows,src,tiles,up,down,g,local,bound,gs32=True):
    m=src.numel();d=rows.shape[1];n=up.shape[1]
    return rows.new_empty((m,d)),rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((m,n))


@torch.library.custom_op('wr_ks3_local::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,
             gate:torch.Tensor,src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,
             up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,local:torch.Tensor,bound:float,
             bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    core.EXECUTED_MOE_BACKWARDS.add(layer)
    valid=(src<grad.shape[0])[:,None];idx=src.clamp(0,grad.shape[0]-1).long()
    gathered=grad[idx].float()
    gg=torch.where(valid,gathered*gate[idx,None].to(torch.bfloat16).float(),0).to(torch.bfloat16)
    ds=torch.where(valid,gathered*y.float(),0).sum(1)
    v=core.mm(gg,down,tiles);m,n=c.shape;d=x.shape[1];e=up.shape[0]
    dc,dg,dp=feature_vjp(v,c,g,local,tiles,src,offs,grad.shape[0],bound)
    if core.BF16_TUNED_KERNELS:
        from wr_reproduction.ks3_bf16_gemm import dw
        dw(h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs)
        dw(dc,x,bank[2*layer*e:(2*layer+1)*e],offs)
    else:
        core._dw[(tr.cdiv(n,32),tr.cdiv(d,64),e)](h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs,n,d,False,num_warps=4)
        core._dw[(tr.cdiv(n,32),tr.cdiv(d,64),e)](dc,x,bank[2*layer*e:(2*layer+1)*e],offs,n,d,False,num_warps=4)
    return core.mm(dc,up,tiles,dtype=grad.dtype),ds,dg,dp


@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,local,bound,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g),torch.empty_like(local)


class RoutedLocalized(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,router,post,w1,w2,up,down,g,local,bank,layer,bound,gs32=True):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
        gate,expert=probs.max(-1);dest,src,offs,tiles,counts=moe.moe_dispatch(expert,router.shape[0])
        scale=gate*post.float().reshape(())
        y,x,c,h=forward(rows,src,tiles,up,down,g,local,bound,gs32)
        ctx.save_for_backward(rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,x,c,h,scale,g,local)
        ctx.up,ctx.down,ctx.bank,ctx.layer,ctx.bound=up,down,bank,layer,bound
        ctx.mark_non_differentiable(counts)
        return y[dest.long()]*scale[:,None].to(y.dtype),probs,counts

    @staticmethod
    def backward(ctx,grad,dprobs,dcounts):
        from track_1_short.perf.kernels import moe_mlp as moe
        rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,x,c,h,scale,g,local=ctx.saved_tensors
        dxp,dsp,dg,dlocal=backward(grad.contiguous(),y,x,c,h,scale,src,offs,tiles,ctx.up,ctx.down,g,local,ctx.bound,ctx.bank,ctx.layer)
        ds=dsp[dest.long()];dp=torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp=dp.scatter_add(1,expert[:,None],(ds*post.float()).reshape(-1,1))
        dl=(probs*(dp-(probs*dp).sum(-1,keepdim=True))).to(torch.bfloat16)
        dx=moe.dx_combine(dxp,dest,dl.float(),router)
        return dx,dl.T@rows,(ds*gate).sum().to(post.dtype).reshape(post.shape),None,None,None,None,dg,dlocal,None,None,None,None


def install(base_install,settings,*args,**kwargs):
    configure(settings)
    bound=settings['bound'];lr=settings['lr_mul']
    if not math.isfinite(bound) or not 0<bound<=4 or not math.isfinite(lr) or lr<=0:
        raise ValueError('Invalid bounded local residual recipe')
    cls=base_install(*args,**kwargs)
    from track_1_short import training
    from track_1_short.model.moe import MOE_INDEX
    class LocalizedGPT(cls):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if self.ks3_subspace_rank!=1 or self.ks3_precision!='bf16':
                raise ValueError('Localized extension requires independent rank1/BF16 experts')
            self.ks3_local=torch.nn.Parameter(torch.zeros((*self.ks3_geometry.shape[:-1],3),device=self.device))
            self.ks3_local.label='ks3_local'
            self.ks3_local_bound=bound

        def _moe(self,layer,normed,weights,post):
            k=MOE_INDEX[layer];w1,w2,router=weights[k];rows=normed.reshape(-1,normed.shape[-1])
            if post is None:post=rows.new_ones(())
            if self.training:
                out,probs,counts=RoutedLocalized.apply(rows,router,post,w1,w2,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.ks3_local[k],self.moe_bank_grad,k,bound,self.ks3_routed_training_gs32)
            else:
                from track_1_short.perf.kernels import moe_mlp as moe
                probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1);gate,expert=probs.max(-1)
                dest,src,offs,tiles,counts=moe.moe_dispatch(expert,8)
                y,_,_,_=forward(rows,src,tiles,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.ks3_local[k],bound)
                out=y[dest.long()]*gate[:,None].to(y.dtype)
            return out.reshape(normed.shape),8*(counts.float()/rows.shape[0]*probs.mean(0)).sum()

    original=training.AnvilAndAdam
    def optimizer(*a,**kw):
        kw['param_table']['ks3_local']=dict(optim='adam',comms='replicated',adam_betas=[.9,.99],lr_mul=lr,wd_mul=0.)
        kw['scatter_order'].append('ks3_local');kw['work_order'].append('ks3_local')
        return original(*a,**kw)
    training.AnvilAndAdam=optimizer
    return LocalizedGPT
