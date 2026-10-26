"""Optional bounded interactions added to the unchanged rank-one KS3 features.

The original FP16 nonlinear equations and pair term share one producer. A
process-fixed placement can add the pair after independent GS32/BF16 packing.
The original FP32 STE runs into a lossless FP32 temporary before the pair VJP.
Shared kernels, routing, projection preparation and independent geometry stay
unchanged. This first implementation deliberately reuses the audited kernels.
"""
import math
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale,_spine_features
from wr_reproduction import ks3_moe as core

AFTER_GS32=False


def configure(coupling):
    """Set once before model/graph construction; never change during training."""
    global AFTER_GS32
    placement=coupling.get('placement','before-GS32')
    if placement not in ('before-GS32','after-independent-GS32'):
        raise ValueError('Unknown pair placement: '+placement)
    AFTER_GS32=placement=='after-independent-GS32'
    return placement


@tr.jit
def _geometry_pair(C, G, PAIR, H, TILES, M:tl.constexpr, N:tl.constexpr,
              BOUND:tl.constexpr, GS32:tl.constexpr, BM:tl.constexpr=64,
              AFTER_GS32:tl.constexpr=False):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    valid=col<N;base=(expert*N+col)*8;gp=G+base
    c0=tl.load(gp+0,valid,other=0.);c1=tl.load(gp+1,valid,other=0.)
    c2=tl.load(gp+2,valid,other=0.);c3=tl.load(gp+3,valid,other=0.)
    c4=tl.load(gp+4,valid,other=0.);c5=tl.load(gp+5,valid,other=0.)
    amax=tl.maximum(tl.maximum(tl.abs(c0),tl.abs(c1)),tl.maximum(tl.maximum(tl.abs(c2),tl.abs(c3)),tl.maximum(tl.abs(c4),tl.abs(c5))))
    ps=_half_scale(amax)
    c0=libdevice.nearbyint(c0/ps);c1=libdevice.nearbyint(c1/ps);c2=libdevice.nearbyint(c2/ps)
    c3=libdevice.nearbyint(c3/ps);c4=libdevice.nearbyint(c4/ps);c5=libdevice.nearbyint(c5/ps)
    c0=tl.minimum(tl.maximum(c0,-127.),127.);c1=tl.minimum(tl.maximum(c1,-127.),127.)
    c2=tl.minimum(tl.maximum(c2,-127.),127.);c3=tl.minimum(tl.maximum(c3,-127.),127.)
    c4=tl.minimum(tl.maximum(c4,-127.),127.);c5=tl.minimum(tl.maximum(c5,-127.),127.)
    rs=tl.load(gp+6,valid,other=0.);rh=tl.load(gp+7,valid,other=0.)
    scale=tl.where(rs>20.,rs,libdevice.log1p(libdevice.exp(rs)))+1e-4;shift=.5*libdevice.tanh(rh)
    x=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    xj=tl.load(C+row[:,None]*N+(col[None,:]^1),mask,other=0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    xj=tl.minimum(tl.maximum(xj,-65504.),65504.)
    raw=tl.load(PAIR+expert*N+col,col<N,other=0.)
    term=BOUND*libdevice.tanh(raw)[None,:]*libdevice.tanh(x*.25)*libdevice.tanh(xj*.25)
    h=_spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    if not AFTER_GS32:h=h+term
    if GS32:
        scale=_half_scale(tl.max(tl.abs(h),1))
        h=tl.minimum(tl.maximum(libdevice.nearbyint(h/scale[:,None]),-127.),127.)*scale[:,None]
    if AFTER_GS32:
        # Preserve the original independent GS32/BF16 value, then add the
        # bounded residual. The final producer still emits original BF16.
        h=h.to(tl.bfloat16).to(tl.float32)+term
    tl.store(H+row[:,None]*N+col[None,:],h,mask)


@tr.jit
def _add_vjp(V,C,PAIR,BASE_DC,DC,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,
             ROWS:tl.constexpr,BOUND:tl.constexpr,BM:tl.constexpr=64):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<M)&(col[None,:]<N)
    real=tl.load(SRC+row,row<M,other=ROWS)<ROWS
    expert=tl.load(TILES+tl.program_id(0)*BM//128)
    original=tl.load(C+row[:,None]*N+col[None,:],mask,other=0.)
    other=tl.load(C+row[:,None]*N+(col[None,:]^1),mask,other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    xj=tl.minimum(tl.maximum(other,-65504.),65504.)
    u=libdevice.tanh(x*.25);uj=libdevice.tanh(xj*.25)
    a=libdevice.tanh(tl.load(PAIR+expert*N+col,col<N,other=0.))
    aj=libdevice.tanh(tl.load(PAIR+expert*N+(col^1),col<N,other=0.))
    v=tl.load(V+row[:,None]*N+col[None,:],mask,other=0.)
    vj=tl.load(V+row[:,None]*N+(col[None,:]^1),mask,other=0.)
    added=(v*a[None,:]+vj*aj[None,:])*BOUND*.25*(1.-u*u)*uj
    added=tl.where(mask&real[:,None]&(original>=-65504.)&(original<=65504.),added,0.)
    base=tl.load(BASE_DC+row[:,None]*N+col[None,:],mask,other=0.)
    tl.store(DC+row[:,None]*N+col[None,:],base+added,mask)
    dp=tl.where(mask&real[:,None],v*BOUND*(1.-a[None,:]*a[None,:])*u*uj,0.)
    tl.store(PART+tl.program_id(0)*N+col,tl.sum(dp,0),col<N)


@tr.jit
def _reduce_pair(P,O,OFFS,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BM:tl.constexpr=64):
    expert=tl.program_id(0);col=tl.program_id(1)*32+tl.arange(0,32)
    lo=0
    if expert>0:lo=tl.load(OFFS+expert-1)//BM
    hi=tl.load(OFFS+expert)//BM
    row=lo+tl.arange(0,B)
    v=tl.load(P+row[:,None]*N+col[None,:],(row[:,None]<hi)&(row[:,None]<T)&(col[None,:]<N),other=0.)
    tl.store(O+expert*N+col,tl.sum(v,0),col<N)


def features(c,g,pair,tiles,bound,gs32=True):
    m,n=c.shape
    bm=64 if core.BF16_TUNED_KERNELS else 16
    h=torch.empty_like(c,dtype=torch.bfloat16)
    _geometry_pair[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,pair,h,tiles,m,n,bound,gs32,BM=bm,AFTER_GS32=AFTER_GS32,num_warps=4)
    return h


def feature_vjp(v,c,g,pair,tiles,src,offs,rows,bound):
    m,n=c.shape;e=g.shape[0]
    base_dc=torch.empty_like(c)
    part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,base_dc,part,tiles,src,m,n,rows,RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    dc=torch.empty_like(c,dtype=torch.bfloat16)
    pair_part=torch.empty((tr.cdiv(m,64),n),device=c.device,dtype=torch.float32)
    _add_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,pair,base_dc,dc,pair_part,tiles,src,m,n,rows,bound,num_warps=4)
    dp=torch.empty_like(pair)
    _reduce_pair[(e,tr.cdiv(n,32))](pair_part,dp,offs,n,pair_part.shape[0],tr.next_power_of_2(pair_part.shape[0]),num_warps=4)
    return dc,dg,dp


@torch.library.custom_op('wr_ks3_pair::forward',mutates_args=())
def forward(rows:torch.Tensor,src:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,
            down:torch.Tensor,g:torch.Tensor,pair:torch.Tensor,bound:float,gs32:bool=True
            )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if core.BF16_TUNED_KERNELS or not gs32:
        m,d=src.numel(),rows.shape[1]
        x=torch.empty((m,d),device=rows.device,dtype=torch.bfloat16)
        core._pack[(tr.cdiv(m,128),tr.cdiv(d,32))](rows,x,m,d,*rows.stride(),
            SRC=src,INPUT_ROWS=rows.shape[0],ROWS_PER_TILE=128,GS32=gs32,num_warps=4)
    else:
        x=core.pack(torch.where((src<rows.shape[0])[:,None],rows[src.clamp(0,rows.shape[0]-1).long()],0).contiguous())
    c=core.mm(x,up.transpose(1,2),tiles)
    h=features(c,g,pair,tiles,bound,gs32)
    return core.mm(h,down.transpose(1,2),tiles,dtype=rows.dtype),x,c,h


@forward.register_fake
def _(rows,src,tiles,up,down,g,pair,bound,gs32=True):
    m=src.numel();d=rows.shape[1];n=up.shape[1]
    return rows.new_empty((m,d)),rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((m,n))


@torch.library.custom_op('wr_ks3_pair::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,
             gate:torch.Tensor,src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,
             up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,pair:torch.Tensor,bound:float,
             bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    core.EXECUTED_MOE_BACKWARDS.add(layer)
    valid=(src<grad.shape[0])[:,None];idx=src.clamp(0,grad.shape[0]-1).long()
    gathered=grad[idx].float()
    gg=torch.where(valid,gathered*gate[idx,None].to(torch.bfloat16).float(),0).to(torch.bfloat16)
    ds=torch.where(valid,gathered*y.float(),0).sum(1)
    v=core.mm(gg,down,tiles);m,n=c.shape;d=x.shape[1];e=up.shape[0]
    dc,dg,dp=feature_vjp(v,c,g,pair,tiles,src,offs,grad.shape[0],bound)
    if core.BF16_TUNED_KERNELS:
        from wr_reproduction.ks3_bf16_gemm import dw
        dw(h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs)
        dw(dc,x,bank[2*layer*e:(2*layer+1)*e],offs)
    else:
        core._dw[(tr.cdiv(n,32),tr.cdiv(d,64),e)](h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs,n,d,False,num_warps=4)
        core._dw[(tr.cdiv(n,32),tr.cdiv(d,64),e)](dc,x,bank[2*layer*e:(2*layer+1)*e],offs,n,d,False,num_warps=4)
    return core.mm(dc,up,tiles,dtype=grad.dtype),ds,dg,dp


@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,pair,bound,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g),torch.empty_like(pair)


class RoutedPairwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,router,post,w1,w2,up,down,g,pair,bank,layer,bound,gs32=True):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
        gate,expert=probs.max(-1);dest,src,offs,tiles,counts=moe.moe_dispatch(expert,router.shape[0])
        scale=gate*post.float().reshape(())
        y,x,c,h=forward(rows,src,tiles,up,down,g,pair,bound,gs32)
        ctx.save_for_backward(rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,x,c,h,scale,g,pair)
        ctx.up,ctx.down,ctx.bank,ctx.layer,ctx.bound=up,down,bank,layer,bound
        ctx.mark_non_differentiable(counts)
        return y[dest.long()]*scale[:,None].to(y.dtype),probs,counts

    @staticmethod
    def backward(ctx,grad,dprobs,dcounts):
        from track_1_short.perf.kernels import moe_mlp as moe
        rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,x,c,h,scale,g,pair=ctx.saved_tensors
        dxp,dsp,dg,dpair=backward(grad.contiguous(),y,x,c,h,scale,src,offs,tiles,ctx.up,ctx.down,g,pair,ctx.bound,ctx.bank,ctx.layer)
        ds=dsp[dest.long()];dp=torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp=dp.scatter_add(1,expert[:,None],(ds*post.float()).reshape(-1,1))
        dl=(probs*(dp-(probs*dp).sum(-1,keepdim=True))).to(torch.bfloat16)
        dx=moe.dx_combine(dxp,dest,dl.float(),router)
        return dx,dl.T@rows,(ds*gate).sum().to(post.dtype).reshape(post.shape),None,None,None,None,dg,dpair,None,None,None,None


def install(base_install,coupling,*args,**kwargs):
    configure(coupling)
    bound=coupling['bound'];lr=coupling['lr_mul']
    if not math.isfinite(bound) or not 0<bound<=4 or not math.isfinite(lr) or lr<=0:
        raise ValueError('Invalid bounded pairwise recipe')
    cls=base_install(*args,**kwargs)
    from track_1_short import training
    from track_1_short.model.moe import MOE_INDEX
    class PairwiseGPT(cls):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if self.ks3_subspace_rank!=1 or self.ks3_precision!='bf16':
                raise ValueError('Pairwise extension requires independent rank1/BF16 experts')
            self.ks3_pair=torch.nn.Parameter(torch.zeros(self.ks3_geometry.shape[:-1],device=self.device))
            self.ks3_pair.label='ks3_pair'
            self.ks3_pair_bound=bound

        def _moe(self,layer,normed,weights,post):
            k=MOE_INDEX[layer];w1,w2,router=weights[k];rows=normed.reshape(-1,normed.shape[-1])
            if post is None:post=rows.new_ones(())
            if self.training:
                out,probs,counts=RoutedPairwise.apply(rows,router,post,w1,w2,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.ks3_pair[k],self.moe_bank_grad,k,bound,self.ks3_routed_training_gs32)
            else:
                from track_1_short.perf.kernels import moe_mlp as moe
                probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1);gate,expert=probs.max(-1)
                dest,src,offs,tiles,counts=moe.moe_dispatch(expert,8)
                y,_,_,_=forward(rows,src,tiles,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.ks3_pair[k],bound)
                out=y[dest.long()]*gate[:,None].to(y.dtype)
            return out.reshape(normed.shape),8*(counts.float()/rows.shape[0]*probs.mean(0)).sum()

    original=training.AnvilAndAdam
    def optimizer(*a,**kw):
        kw['param_table']['ks3_pair']=dict(optim='adam',comms='replicated',adam_betas=[.9,.99],lr_mul=lr,wd_mul=0.)
        kw['scatter_order'].append('ks3_pair');kw['work_order'].append('ks3_pair')
        return original(*a,**kw)
    training.AnvilAndAdam=optimizer
    return PairwiseGPT
