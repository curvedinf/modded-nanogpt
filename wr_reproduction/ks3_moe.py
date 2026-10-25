"""GS32/BF16 KS3 experts inside the immutable PR393 MoE.

Geometry and STE adapted from ks3_research/vendor/ks3_int8.py and
control393/ks3_dynamic.py (curvedinf/k-splanifolds, CC-BY-SA-4.0).
Device-side padded routing replaces Python expert slicing; equations retain
GS32 rounding, FP16 nonlinearities, BF16 products and FP32 accumulation.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale,_spine_features
from wr_reproduction.fused_fp8 import _native

BF16_TUNED_KERNELS=False
KS3_RANK=1
EXECUTED_MOE_BACKWARDS=set()  # CPU evidence from actual custom-op capture/eager execution.

@tr.jit
def _pack(A,O,R:tl.constexpr,C:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,
          SCALE=None,OT=None,AMAX=None,SRC=None,INPUT_ROWS:tl.constexpr=0,FP8:tl.constexpr=False,ROWS_PER_TILE:tl.constexpr=8,GS32:tl.constexpr=True,NATIVE_OUT=None,NATIVE_T=None):
    row=tl.program_id(0)*ROWS_PER_TILE+tl.arange(0,ROWS_PER_TILE)
    col=tl.program_id(1)*32+tl.arange(0,32)
    mask=(row[:,None]<R)&(col[None,:]<C)
    source_row=row
    if INPUT_ROWS:
        source_row=tl.load(SRC+row,row<R,other=INPUT_ROWS)
        mask=mask&(source_row[:,None]<INPUT_ROWS)
    a=tl.load(A+source_row[:,None]*A0+col[None,:]*A1,mask,other=0.).to(tl.float32)
    if GS32:
        s=_half_scale(tl.max(tl.abs(a),1))
        q=tl.minimum(tl.maximum(libdevice.nearbyint(a/s[:,None]),-127.),127.)
        values=q*s[:,None]
    else:
        # Opt-in dense BF16/FP8 training may retain unquantized projection values.
        values=a
    if FP8:
        mask=(row[:,None]<R)&(col[None,:]<C)
        out=tl.minimum(tl.maximum(values/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
        tl.store(OT+col[None,:]*R+row[:,None],out,mask)
        if AMAX is not None:tl.atomic_max(AMAX,tl.max(tl.max(tl.abs(values),1),0))
        if NATIVE_OUT is not None:
            native=_native(out.to(tl.uint8,bitcast=True),False,False)
            tl.store(NATIVE_OUT+row[:,None]*C+col[None,:],native,mask)
            if NATIVE_T is not None:tl.store(NATIVE_T+col[None,:]*R+row[:,None],native,mask)
    else:out=values
    if INPUT_ROWS:mask=(row[:,None]<R)&(col[None,:]<C)
    tl.store(O+row[:,None]*C+col[None,:],out,mask)

def pack(a):
    out=torch.empty(a.shape,device=a.device,dtype=torch.bfloat16)
    _pack[(tr.cdiv(a.shape[0],8),tr.cdiv(a.shape[1],32))](a,out,*a.shape,*a.stride())
    return out

@tr.jit
def _mm(A,W,O,TILES,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        W0:tl.constexpr,W1:tl.constexpr,W2:tl.constexpr,
        BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    r=tl.program_id(0)*BM+tl.arange(0,BM);c=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);e=tl.load(TILES+tl.program_id(0)*BM//128)
    acc=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(K,BK)):
        k=base*BK+kk
        a=tl.load(A+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),other=0.)
        w=tl.load(W+e*W0+k[:,None]*W1+c[None,:]*W2,(k[:,None]<K)&(c[None,:]<N),other=0.)
        acc=tl.dot(a,w,acc)
    tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<M)&(c[None,:]<N))

def mm(a,w,tiles,dtype=torch.float32):
    if BF16_TUNED_KERNELS:
        from wr_reproduction.ks3_bf16_gemm import run
        return run(a,w,tiles,dtype)
    m,k=a.shape;e,kw,n=w.shape
    assert k==kw and a.is_contiguous()
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    _mm[(tr.cdiv(m,32),tr.cdiv(n,64))](a,w,out,tiles,m,n,k,*w.stride(),num_warps=4)
    return out

@tr.jit
def _dw(A,B,O,OFFS,H:tl.constexpr,D:tl.constexpr,ADD:tl.constexpr,
        BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    e=tl.program_id(2);lo=0
    if e>0:lo=tl.load(OFFS+e-1)
    hi=tl.load(OFFS+e)
    h=tl.program_id(0)*BM+tl.arange(0,BM);d=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(lo,hi,BK):
        r=start+kk
        a=tl.load(A+r[None,:]*H+h[:,None],(r[None,:]<hi)&(h[:,None]<H),other=0.)
        b=tl.load(B+r[:,None]*D+d[None,:],(r[:,None]<hi)&(d[None,:]<D),other=0.)
        acc=tl.dot(a,b,acc)
    ptr=O+(e*H+h[:,None])*D+d[None,:];valid=(h[:,None]<H)&(d[None,:]<D)
    if ADD:acc+=tl.load(ptr,valid,other=0.).to(tl.float32)
    tl.store(ptr,acc,valid)

@tr.jit
def _chart_mean(x,RANK:tl.constexpr):
    if RANK==1:return x
    grouped=tl.reshape(x,(x.shape[0],32//RANK,RANK))
    mean=tl.sum(grouped,2)/RANK
    return tl.reshape(tl.broadcast_to(mean[:,:,None],(x.shape[0],32//RANK,RANK)),x.shape)


@tr.jit
def _geometry(C,G,H,TILES,M:tl.constexpr,N:tl.constexpr,HT=None,SCALE=None,AMAX=None,FP8:tl.constexpr=False,PARTIAL_AMAX:tl.constexpr=False,BM:tl.constexpr=16,GS32:tl.constexpr=True,RANK:tl.constexpr=1,NATIVE_OUT=None,NATIVE_T=None):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    e=tl.load(TILES+tl.program_id(0)*BM//128);valid=cols<N
    base=(e*(N//RANK)+cols//RANK)*(6*RANK+2)
    gp=G+base+(cols%RANK)*6
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
    raw=tl.load(G+base+6*RANK,valid,other=0.);rh=tl.load(G+base+6*RANK+1,valid,other=0.)
    scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4;shift=.5*libdevice.tanh(rh)
    x=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    h=_spine_features(x,_chart_mean(x,RANK),c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    if GS32:
        hs=_half_scale(tl.max(tl.abs(h),1));q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.)
        values=q*hs[:,None]
    else:
        # Opt-in dense BF16/FP8 training may retain unquantized projection values.
        values=h
    if FP8:
        out=tl.minimum(tl.maximum(values/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
        tl.store(HT+cols[None,:]*M+rows[:,None],out,(rows[:,None]<M)&valid[None,:])
        if NATIVE_OUT is not None:
            native=_native(out.to(tl.uint8,bitcast=True),False,False)
            tl.store(NATIVE_OUT+rows[:,None]*N+cols[None,:],native,(rows[:,None]<M)&valid[None,:])
            if NATIVE_T is not None:tl.store(NATIVE_T+cols[None,:]*M+rows[:,None],native,(rows[:,None]<M)&valid[None,:])
        if AMAX is not None:
            largest=tl.max(tl.max(tl.abs(values),1),0)
            if PARTIAL_AMAX:
                tl.store(AMAX+tl.program_id(0)*tl.cdiv(N,32)+tl.program_id(1),largest)
            else:tl.atomic_max(AMAX,largest)
    else:out=values
    tl.store(H+rows[:,None]*N+cols[None,:],out,(rows[:,None]<M)&valid[None,:])

@tr.jit
def _vjp(V,C,G,DC,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,ROWS:tl.constexpr,DCT=None,SCALE=None,AMAX=None,FP8:tl.constexpr=False,PARTIAL_AMAX:tl.constexpr=False,BM:tl.constexpr=64,RANK:tl.constexpr=1,NATIVE_OUT=None,NATIVE_T=None):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    e=tl.load(TILES+tl.program_id(0)*BM//128);mask=(rows[:,None]<M)&(cols[None,:]<N)
    real=tl.load(SRC+rows,rows<M,other=ROWS)<ROWS
    value=tl.load(V+rows[:,None]*N+cols[None,:],mask,other=0.)
    original=tl.load(C+rows[:,None]*N+cols[None,:],mask,other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    base=(e*(N//RANK)+cols//RANK)*(6*RANK+2)
    gp=G+base+(cols%RANK)*6
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
    rs=tl.load(G+base+6*RANK,cols<N,other=0.);rh=tl.load(G+base+6*RANK+1,cols<N,other=0.)
    scale=tl.where(rs>20.,rs,libdevice.log1p(libdevice.exp(rs)))+1e-4;shift=.5*libdevice.tanh(rh)
    t=4.*libdevice.tanh(_chart_mean(x,RANK)*.25);z=libdevice.tanh(scale[None,:]*(t+shift[None,:]))
    cubic=2*c0-2*c1+c2+c3;quadratic=-3*c0+3*c1-2*c2-c3
    spine=((cubic*t+quadratic)*t+c2)*t+c0;warped=((4*c5*z+2*c4)*z-3*c5)*z-c4
    dp=tl.where(mask&real[:,None],value*2.*tl.maximum(x+spine+warped,0.),0.)
    dz=((12*c5*z+4*c4)*z-3*c5)*(1.-z*z)
    dm=dp*((3*cubic*t+2*quadratic)*t+c2+dz*scale)*(1.-t*t*.0625)
    dc=tl.where((original>=-65504.)&(original<=65504.),dp+_chart_mean(dm,RANK),0.)
    if FP8:
        out=tl.minimum(tl.maximum(dc/tl.load(SCALE),-57344.),57344.).to(tl.float8e5)
        tl.store(DCT+cols[None,:]*M+rows[:,None],out,mask)
        if NATIVE_OUT is not None:
            native=_native(out.to(tl.uint8,bitcast=True),True,True)
            tl.store(NATIVE_OUT+rows[:,None]*N+cols[None,:],native,mask)
            if NATIVE_T is not None:tl.store(NATIVE_T+cols[None,:]*M+rows[:,None],native,mask)
        largest=tl.max(tl.max(tl.abs(dc),1),0)
        if PARTIAL_AMAX:
            tl.store(AMAX+tl.program_id(0)*tl.cdiv(N,32)+tl.program_id(1),largest)
        else:tl.atomic_max(AMAX,largest)
    else:out=dc
    tl.store(DC+rows[:,None]*N+cols[None,:],out,mask)
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
def _reduce(P,O,OFFS,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BM:tl.constexpr=64,RANK:tl.constexpr=1):
    e=tl.program_id(0);n=tl.program_id(1)
    lo=0
    if e>0:lo=tl.load(OFFS+e-1)//BM
    hi=tl.load(OFFS+e)//BM
    r=lo+tl.arange(0,B);j=tl.arange(0,8)
    v=tl.load(P+(r[:,None]*N+n)*8+j[None,:],(r[:,None]<hi)&(r[:,None]<T),other=0.)
    total=tl.sum(v,0)
    if RANK==1:
        tl.store(O+(e*N+n)*8+j,total)
    else:
        base=(e*(N//RANK)+n//RANK)*(6*RANK+2)
        tl.store(O+base+(n%RANK)*6+j,total,j<6)
        if n%RANK==0:
            # Polynomial coefficients are per coordinate; scale and shift are
            # shared by the chart, exactly as in the pinned upstream kernel.
            for coordinate in tl.static_range(1,RANK):
                extra=tl.load(P+(r[:,None]*N+n+coordinate)*8+j[None,:],
                              (r[:,None]<hi)&(r[:,None]<T),other=0.)
                total+=tl.sum(extra,0)
            tl.store(O+base+6*RANK+j-6,total,j>=6)

@torch.library.custom_op('wr_ks3::forward',mutates_args=())
def forward(rows:torch.Tensor,src:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if BF16_TUNED_KERNELS:
        m,d=src.numel(),rows.shape[1]
        x=torch.empty((m,d),device=rows.device,dtype=torch.bfloat16)
        _pack[(tr.cdiv(m,128),tr.cdiv(d,32))](rows,x,m,d,*rows.stride(),
            SRC=src,INPUT_ROWS=rows.shape[0],ROWS_PER_TILE=128,num_warps=4)
    else:
        x=pack(torch.where((src<rows.shape[0])[:,None],rows[src.clamp(0,rows.shape[0]-1).long()],0).contiguous())
    c=mm(x,up.transpose(1,2),tiles);h=torch.empty_like(c,dtype=torch.bfloat16)
    bm=64 if BF16_TUNED_KERNELS else 16
    _geometry[(tr.cdiv(c.shape[0],bm),tr.cdiv(c.shape[1],32))](c,g,h,tiles,*c.shape,BM=bm,RANK=KS3_RANK,num_warps=4)
    y=mm(h,down.transpose(1,2),tiles,dtype=rows.dtype)
    return y,x,c,h

@forward.register_fake
def _(rows,src,tiles,up,down,g):
    m=src.numel();d=rows.shape[1];n=up.shape[1]
    return rows.new_empty((m,d)),rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((m,n))

@torch.library.custom_op('wr_ks3::backward',mutates_args={'bank'})
def backward(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,gate:torch.Tensor,
             src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,
             bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if up.shape[0]==8:EXECUTED_MOE_BACKWARDS.add(layer)
    valid=(src<grad.shape[0])[:,None];idx=src.clamp(0,grad.shape[0]-1).long()
    gathered=grad[idx].float();gg=torch.where(valid,gathered*gate[idx,None].to(torch.bfloat16).float(),0).to(torch.bfloat16)
    ds=torch.where(valid,gathered*y.float(),0).sum(1)
    v=mm(gg,down,tiles);m,n=c.shape;d=x.shape[1];e=up.shape[0]
    dc=torch.empty_like(c,dtype=torch.bfloat16);part=torch.empty((tr.cdiv(m,64),n,8),device=c.device,dtype=torch.float32)
    _vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,dc,part,tiles,src,m,n,grad.shape[0],RANK=KS3_RANK,num_warps=4)
    dg=torch.empty_like(g)
    _reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=KS3_RANK,num_warps=4)
    if BF16_TUNED_KERNELS:
        from wr_reproduction.ks3_bf16_gemm import dw
        dw(h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs)
        dw(dc,x,bank[2*layer*e:(2*layer+1)*e],offs)
    else:
        _dw[(tr.cdiv(n,32),tr.cdiv(d,64),e)](h,gg,bank[(2*layer+1)*e:(2*layer+2)*e],offs,n,d,False,num_warps=4)
        _dw[(tr.cdiv(n,32),tr.cdiv(d,64),e)](dc,x,bank[2*layer*e:(2*layer+1)*e],offs,n,d,False,num_warps=4)
    dx=mm(dc,up,tiles,dtype=grad.dtype)
    return dx,ds,dg

@backward.register_fake
def _(grad,y,x,c,h,gate,src,offs,tiles,up,down,g,bank,layer):
    return x.new_empty(x.shape),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g)

class RoutedKS3(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,router,post,w1,w2,up,down,g,bank,layer):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
        gate,expert=probs.max(-1);dest,src,offs,tiles,counts=moe.moe_dispatch(expert,router.shape[0])
        pf=post.float().reshape(());scale=gate*pf
        y,x,c,h=forward(rows,src,tiles,up,down,g)
        ctx.save_for_backward(rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,x,c,h,scale,g)
        ctx.up=up;ctx.down=down;ctx.bank=bank;ctx.layer=layer
        ctx.mark_non_differentiable(counts)
        return y[dest.long()]*scale[:,None].to(y.dtype),probs,counts
    @staticmethod
    def backward(ctx,grad,dprobs,dcounts):
        from track_1_short.perf.kernels import moe_mlp as moe
        rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,x,c,h,scale,g=ctx.saved_tensors
        dxp,dsp,dg=backward(grad.contiguous(),y,x,c,h,scale,src,offs,tiles,ctx.up,ctx.down,g,ctx.bank,ctx.layer)
        ds=dsp[dest.long()];dp=torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp=dp.scatter_add(1,expert[:,None],(ds*post.float()).reshape(-1,1))
        dl=(probs*(dp-(probs*dp).sum(-1,keepdim=True))).to(torch.bfloat16)
        dx=moe.dx_combine(dxp,dest,dl.float(),router)
        return dx,dl.T@rows,(ds*gate).sum().to(post.dtype).reshape(post.shape),None,None,None,None,dg,None,None


def install(precision="bf16",geometry_config=None,training_gs32=True,charts=1024,batched_weight_pack=False,geometry_backward_rows=64,bf16_kernel_mode='legacy',projection_lr_multiplier=1.,subspace_rank=1,prepared_native_weights=False,prepared_native_activations=False,prepared_native_dual=False,prepared_native_gather=False):
    """Opt-in overlay; no immutable source or default control class is edited."""
    from track_1_short.model.gpt import GPT
    from track_1_short.model.moe import MOE_INDEX
    from track_1_short import training
    from track_1_short.model import gpt as gpt_module,moe as moe_module
    global KS3_RANK
    if subspace_rank not in (1,2,4):raise ValueError('Unsupported KS3 rank')
    if subspace_rank!=1 and (precision!='bf16' or charts*subspace_rank!=1024 or bf16_kernel_mode!='tuned'):
        raise ValueError('Higher rank requires1024 projection features and tuned BF16 training')
    if prepared_native_weights and (precision!='fp8' or charts not in (1024,1536) or subspace_rank!=1 or not batched_weight_pack):
        raise ValueError('Prepared native weights require1024/1536 rank-one FP8 with batched packing')
    if prepared_native_activations and not prepared_native_weights:raise ValueError('Prepared activations require prepared weights')
    if prepared_native_dual and not prepared_native_activations:raise ValueError('Prepared dual layouts require native activations')
    if prepared_native_gather and not prepared_native_dual:raise ValueError('Prepared gather requires native dual layouts')
    KS3_RANK=subspace_rank
    features=charts*subspace_rank
    if features not in (512,768,1024,1280,1536):raise ValueError('Unsupported KS3 projection width')
    if features==1280 and precision!='bf16':raise ValueError('1280-feature preparation covers BF16 only')
    if bf16_kernel_mode not in ('legacy','tuned'):raise ValueError('Unknown BF16 kernel mode')
    if bf16_kernel_mode=='tuned' and (precision!='bf16' or features not in (512,768,1024,1280,1536)):raise ValueError('Unsupported tuned BF16 width')
    if geometry_backward_rows not in (32,64,128):raise ValueError('Invalid geometry tile')
    if precision=='fp8':
        from wr_reproduction import ks3_moe_fp8
        ks3_moe_fp8.GEOMETRY_BACKWARD_ROWS=geometry_backward_rows
        ks3_moe_fp8.PREPARED_ACTIVATIONS=prepared_native_activations
        ks3_moe_fp8.PREPARED_DUAL=prepared_native_dual
        ks3_moe_fp8.PREPARED_GATHER=prepared_native_gather
    # Both immutable modules bind this constant. Only the opt-in KS3 launcher
    # invokes this overlay; the pinned source files are never edited.
    gpt_module.MOE_HIDDEN=moe_module.MOE_HIDDEN=features
    geometry_config=geometry_config or {}
    geometry_lr=geometry_config.get('lr_mul',.125)
    initial_scale=geometry_config.get('raw_scale',.75)
    initial_shift=geometry_config.get('raw_shift',0.)
    coefficients=geometry_config.get('coefficients',[0.]*6)
    if len(coefficients)!=6:raise ValueError('KS3 requires six geometry coefficients')
    class KS3GPT(GPT):
        def init_moe(self,model_dim,rank,grad_scale,bound):
            initial_rng=torch.get_rng_state() if features!=1024 else None
            super().init_moe(model_dim,rank,grad_scale,bound)
            if initial_rng is not None:
                from wr_reproduction.ks3_initialization import restore_original_moe_rng
                restore_original_moe_rng(initial_rng,self.moe_router,model_dim,bound)
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            global BF16_TUNED_KERNELS
            BF16_TUNED_KERNELS=precision=='bf16' and bf16_kernel_mode=='tuned'
            self.ks3_precision=precision
            self.ks3_training_gs32=training_gs32
            self.ks3_batched_weight_pack=batched_weight_pack
            self.ks3_prepared_native_weights=prepared_native_weights
            self.ks3_subspace_rank=subspace_rank
            g=torch.zeros((5,8,charts,6*subspace_rank+2),device=self.device)
            g[...,6*subspace_rank]=initial_scale;g[...,6*subspace_rank+1]=initial_shift
            for coordinate in range(subspace_rank):
                g[...,coordinate*6:(coordinate+1)*6]=torch.tensor(coefficients,device=self.device)
            self.ks3_geometry=torch.nn.Parameter(g);self.ks3_geometry.label='ks3_geometry'
            self.register_buffer('ks3_up',torch.empty((5,8,features,768),device=self.device,dtype=torch.bfloat16),persistent=False)
            self.register_buffer('ks3_down',torch.empty((5,8,768,features),device=self.device,dtype=torch.bfloat16),persistent=False)
            if precision=='fp8':
                from wr_reproduction.ks3_moe_fp8 import init_buffers
                init_buffers(self)
        def train(self,mode=True):
            result=super().train(mode)
            global BF16_TUNED_KERNELS
            # Validation retains the exact original BF16 kernel path.
            BF16_TUNED_KERNELS=bool(mode and precision=='bf16' and bf16_kernel_mode=='tuned')
            return result
        @torch.no_grad()
        def _quantize_moe(self,lo,hi,exact):
            if lo!=0 or hi!=80:raise ValueError('Single physical owner required')
            if precision=='fp8':
                from wr_reproduction.ks3_moe_fp8 import refresh_weights
                return refresh_weights(self,exact)
            self.prepare_ks3_eval()
        @torch.no_grad()
        def prepare_ks3_eval(self):
            if self.training and BF16_TUNED_KERNELS:
                from wr_reproduction.ks3_weight_pack import refresh_bf16
                return refresh_bf16(self)
            bank=self.moe_bank.view(5,2,8,features,768)
            for k in range(5):
                # GS32 groups follow each projection's input dimension.
                self.ks3_up[k].copy_(pack(bank[k,0].reshape(8*features,768)).view(8,features,768))
                for e in range(8):self.ks3_down[k,e].copy_(pack(bank[k,1,e].T))
        def build_moe_cols(self):pass
        def _moe(self,layer,normed,weights,post):
            k=MOE_INDEX[layer];w1,w2,router=weights[k];rows=normed.reshape(-1,normed.shape[-1])
            if post is None:post=rows.new_ones(())
            if self.training and precision=='fp8':
                from wr_reproduction.ks3_moe_fp8 import routed
                out,probs,counts=routed(self,k,rows,router,post,w1,w2)
            elif self.training:
                out,probs,counts=RoutedKS3.apply(rows,router,post,w1,w2,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k],self.moe_bank_grad,k)
            else:
                from track_1_short.perf.kernels import moe_mlp as moe
                probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1);gate,expert=probs.max(-1)
                dest,src,offs,tiles,counts=moe.moe_dispatch(expert,8)
                y,_,_,_=forward(rows,src,tiles,self.ks3_up[k],self.ks3_down[k],self.ks3_geometry[k])
                out=y[dest.long()]*gate[:,None].to(y.dtype)
            return out.reshape(normed.shape),8*(counts.float()/rows.shape[0]*probs.mean(0)).sum()
    original=training.AnvilAndAdam
    def optimizer(*args,**kwargs):
        from wr_reproduction.ks3_optimizer import configure
        configure(kwargs['param_table'],geometry_lr,projection_lr_multiplier)
        kwargs['scatter_order'].append('ks3_geometry');kwargs['work_order'].append('ks3_geometry')
        return original(*args,**kwargs)
    training.AnvilAndAdam=optimizer
    return KS3GPT
