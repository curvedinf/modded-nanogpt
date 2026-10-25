"""FP8-only product variant of the checked GS32/BF16 KS3 MoE block.

Weights are packed once per optimizer update; producers emit reusable dual
FP8 layouts. No standalone per-GEMM operand conversion. Original rank-local
MoE delayed activation/gradient scales are reused. Geometry remains FP16/FP32.
"""
import torch
import triton as tr
from wr_reproduction.ks3_moe import _pack,_geometry,_vjp,_reduce
from wr_reproduction.ks3_fp8_gemm import run,grouped_dw

GEOMETRY_BACKWARD_ROWS=64
PREPARED_ACTIVATIONS=False
PREPARED_DUAL=False
PREPARED_GATHER=False

def init_buffers(m):
    charts=m.ks3_geometry.shape[2]
    for name,shape in [('ks3_f8_up',(5,8,charts,768)),('ks3_f8_up_t',(5,8,768,charts)),
                       ('ks3_f8_down',(5,8,768,charts)),('ks3_f8_down_t',(5,8,charts,768))]:
        m.register_buffer(name,torch.empty(shape,device=m.device,dtype=torch.float8_e4m3fn),persistent=False)
        if getattr(m,'ks3_prepared_native_weights',False):
            m.register_buffer(name.replace('ks3_f8_','ks3_native_'),
                torch.empty(shape,device=m.device,dtype=torch.float8_e4m3fnuz),persistent=False)
    m.register_buffer('ks3_weight_scales',torch.ones((5,2,8),device=m.device),persistent=False)
    m.register_buffer('ks3_weight_amax',torch.ones((5,2,8),device=m.device),persistent=False)
    if getattr(m,'ks3_batched_weight_pack',False):
        m.register_buffer('ks3_pack_partial_amax',
            torch.empty((5,2,8,tr.cdiv(charts,128)*24),device=m.device),persistent=False)

@torch.no_grad()
def refresh_weights(m,exact):
    if getattr(m,'ks3_batched_weight_pack',False):
        from wr_reproduction.ks3_weight_pack import refresh
        return refresh(m,exact)
    bank=m.moe_bank.view(5,2,8,m.ks3_geometry.shape[2],768)
    for k in range(5):
        for projection in range(2):
            scale=m.ks3_weight_scales[k,projection];amax=m.ks3_weight_amax[k,projection]
            if exact:scale.copy_(bank[k,projection].float().abs().amax((1,2)).clamp_min(1e-12)/448.)
            else:scale.copy_(amax.clamp_min(1e-12)/448.)
            amax.zero_()
            for e in range(8):
                a=bank[k,projection,e] if projection==0 else bank[k,projection,e].T
                q=(m.ks3_f8_up if projection==0 else m.ks3_f8_down)[k,e]
                qt=(m.ks3_f8_up_t if projection==0 else m.ks3_f8_down_t)[k,e]
                _pack[(tr.cdiv(a.shape[0],8),tr.cdiv(a.shape[1],32))](a,q,*a.shape,*a.stride(),
                    SCALE=scale[e:e+1],OT=qt,AMAX=amax[e:e+1],FP8=True,GS32=getattr(m,'ks3_training_gs32',True))

@torch.library.custom_op('wr_ks3_fp8::forward',mutates_args={'pa'})
def forward(rows:torch.Tensor,src:torch.Tensor,tiles:torch.Tensor,up:torch.Tensor,down:torch.Tensor,
            us:torch.Tensor,ws:torch.Tensor,xs:torch.Tensor,ps:torch.Tensor,pa:torch.Tensor,
            g:torch.Tensor,gs32:bool=True)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    m,d=src.numel(),rows.shape[1];e,n,_=up.shape
    x=torch.empty((m,d),device=rows.device,dtype=torch.float8_e4m3fn);xt=torch.empty((d,m),device=rows.device,dtype=x.dtype)
    xn=torch.empty_like(x,dtype=torch.float8_e4m3fnuz) if PREPARED_ACTIVATIONS else None
    xnt=torch.empty_like(xt,dtype=torch.float8_e4m3fnuz) if PREPARED_DUAL else None
    _pack[(tr.cdiv(m,128),tr.cdiv(d,32))](rows,x,m,d,*rows.stride(),SCALE=xs,OT=xt,SRC=src,INPUT_ROWS=rows.shape[0],FP8=True,ROWS_PER_TILE=128,GS32=gs32,NATIVE_OUT=xn,NATIVE_T=xnt,num_warps=4)
    c=run(xn if xn is not None else x,up.reshape(e*n,d).T,us,xs,tiles=tiles,n=n)[0]
    h=torch.empty((m,n),device=rows.device,dtype=x.dtype);ht=torch.empty((n,m),device=rows.device,dtype=x.dtype)
    hn=torch.empty_like(h,dtype=torch.float8_e4m3fnuz) if PREPARED_ACTIVATIONS else None
    hnt=torch.empty_like(ht,dtype=torch.float8_e4m3fnuz) if PREPARED_DUAL else None
    partial=torch.empty((tr.cdiv(m,64)*tr.cdiv(n,32),),device=rows.device,dtype=torch.float32)
    _geometry[(tr.cdiv(m,64),tr.cdiv(n,32))](c,g,h,tiles,m,n,HT=ht,SCALE=ps,AMAX=partial,FP8=True,PARTIAL_AMAX=True,BM=64,GS32=gs32,NATIVE_OUT=hn,NATIVE_T=hnt,num_warps=4)
    # One partial per producer tile avoids tens of thousands of contending
    # atomics to one scale statistic. Preserve accumulation across forwards.
    torch.maximum(pa,partial.amax(),out=pa)
    y=run(hn if hn is not None else h,down.reshape(e*d,n).T,ws,ps,tiles=tiles,n=d,out_dtype=rows.dtype)[0]
    return y,c,xnt if xnt is not None else xt,hnt if hnt is not None else ht

@forward.register_fake
def _(rows,src,tiles,up,down,us,ws,xs,ps,pa,g,gs32=True):
    m,d,n=src.numel(),rows.shape[1],up.shape[1]
    dtype=torch.float8_e4m3fnuz if PREPARED_DUAL else torch.float8_e4m3fn
    return rows.new_empty((m,d)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((d,m),dtype=dtype),rows.new_empty((n,m),dtype=dtype)

@torch.library.custom_op('wr_ks3_fp8::backward',mutates_args={'bank','da'})
def backward(grad:torch.Tensor,y:torch.Tensor,c:torch.Tensor,xt:torch.Tensor,ht:torch.Tensor,gate:torch.Tensor,
             src:torch.Tensor,offs:torch.Tensor,tiles:torch.Tensor,upt:torch.Tensor,downt:torch.Tensor,
             us:torch.Tensor,ws:torch.Tensor,xs:torch.Tensor,gs:torch.Tensor,ps:torch.Tensor,ds:torch.Tensor,
             da:torch.Tensor,bank:torch.Tensor,flag:torch.Tensor,g:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    from track_1_short.perf.kernels.moe_mlp import grad_gather_quant
    if PREPARED_GATHER:
        from wr_reproduction.ks3_native_gather import gather
        original,original_t,dgate,gg,gt=gather(grad.contiguous(),y,src,gate.to(torch.bfloat16).float(),gs)
    else:
        gg,gt,dgate=grad_gather_quant(grad.contiguous(),y,src,gate.to(torch.bfloat16).float(),gs)
    m,n=c.shape;e,d,_=upt.shape
    v=run(gg,downt.reshape(e*n,d).T,ws,gs,tiles=tiles,n=n)[0]
    dc=torch.empty((m,n),device=c.device,dtype=torch.float8_e5m2);dct=torch.empty((n,m),device=c.device,dtype=dc.dtype)
    dcn=torch.empty_like(dc,dtype=torch.float8_e5m2fnuz) if PREPARED_ACTIVATIONS else None
    dcnt=torch.empty_like(dct,dtype=torch.float8_e5m2fnuz) if PREPARED_DUAL else None
    bm=GEOMETRY_BACKWARD_ROWS
    part=torch.empty((tr.cdiv(m,bm),n,8),device=c.device,dtype=torch.float32)
    maxima=torch.empty((tr.cdiv(m,bm)*tr.cdiv(n,32),),device=c.device,dtype=torch.float32)
    _vjp[(tr.cdiv(m,bm),tr.cdiv(n,32))](v,c,g,dc,part,tiles,src,m,n,grad.shape[0],
        DCT=dct,SCALE=ds,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=bm,NATIVE_OUT=dcn,NATIVE_T=dcnt,num_warps=4)
    torch.maximum(da,maxima.amax(),out=da)
    dg=torch.empty_like(g)
    _reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),BM=bm,num_warps=4)
    grouped_dw(ht,gt,offs,ps,gs,e,bank[(2*layer+1)*e:(2*layer+2)*e],flag)
    grouped_dw(dcnt if dcnt is not None else dct,xt,offs,ds,xs,e,bank[2*layer*e:(2*layer+1)*e],flag)
    dx=run(dcn if dcn is not None else dc,upt.reshape(e*d,n).T,us,ds,tiles=tiles,n=d,out_dtype=grad.dtype)[0]
    return dx,dgate,dg

@backward.register_fake
def _(grad,y,c,xt,ht,gate,src,offs,tiles,upt,downt,us,ws,xs,gs,ps,ds,da,bank,flag,g,layer):
    return grad.new_empty((c.shape[0],grad.shape[1])),c.new_empty((c.shape[0],)),torch.empty_like(g)

class Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,router,post,w1,w2,g,caches,scales,bank,flag,layer,gs32):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
        gate,expert=probs.max(-1);dest,src,offs,tiles,counts=moe.moe_dispatch(expert,8)
        scale=gate*post.float().reshape(())
        up,upt,down,downt=caches;us,ws,xs,gs,ps,pa,ds,da=scales
        y,c,xt,ht=forward(rows,src,tiles,up,down,us,ws,xs,ps,pa,g,gs32)
        ctx.save_for_backward(rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,c,xt,ht,scale,g)
        ctx.caches=caches;ctx.scales=scales;ctx.bank=bank;ctx.flag=flag;ctx.layer=layer
        ctx.mark_non_differentiable(counts)
        return y[dest.long()]*scale[:,None].to(y.dtype),probs,counts
    @staticmethod
    def backward(ctx,grad,dprobs,dcounts):
        from track_1_short.perf.kernels import moe_mlp as moe
        rows,router,post,probs,gate,expert,dest,src,offs,tiles,y,c,xt,ht,scale,g=ctx.saved_tensors
        up,upt,down,downt=ctx.caches;us,ws,xs,gs,ps,pa,ds,da=ctx.scales
        dxp,dsp,dg=backward(grad.contiguous(),y,c,xt,ht,scale,src,offs,tiles,upt,downt,us,ws,xs,gs,ps,ds,da,ctx.bank,ctx.flag,g,ctx.layer)
        dgate=dsp[dest.long()];dp=torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp=dp.scatter_add(1,expert[:,None],(dgate*post.float()).reshape(-1,1))
        dl=(probs*(dp-(probs*dp).sum(-1,keepdim=True))).to(torch.bfloat16)
        dx=moe.dx_combine(dxp,dest,dl.float(),router)
        return dx,dl.T@rows,(dgate*gate).sum().to(post.dtype).reshape(post.shape),None,None,dg,None,None,None,None,None,None

def routed(m,k,rows,router,post,w1,w2):
    prefix='ks3_native_' if getattr(m,'ks3_prepared_native_weights',False) else 'ks3_f8_'
    caches=tuple(getattr(m,prefix+name)[k] for name in ('up','up_t','down','down_t'))
    scales=(m.ks3_weight_scales[k,0],m.ks3_weight_scales[k,1],m._mlp_x_scale,m._mlp_grad_scale,
            m._moe_post_scales[k:k+1],m._moe_post_amax[k],m._moe_dpre_scales[k:k+1],m._moe_dpre_amax[k])
    return Routed.apply(rows,router,post,w1,w2,m.ks3_geometry[k],caches,scales,m.moe_bank_grad,m.moe_grad_accumulate,k,getattr(m,'ks3_training_gs32',True))
