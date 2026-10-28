"""Measured H100-only KS3 overlays. Shared PR393 operations stay unchanged."""
import os,json
from pathlib import Path
import torch,triton as tr
from experiments.h100_ks3.h100_kernels import gemm,gather_gate
from wr_reproduction import ks3_moe as core,ks3_bf16_gemm as matrix
from wr_reproduction.ks3_bf16_gather import gather

VARIANT='fused';PACK_INPUT=False;FUSE_VJP=False;FUSE_DX=False;SEPARATE=False;PREPARE=False;FAST_GATE=False;VJP_ROWS=64
FAST_TANH_FORWARD=False;FAST_TANH_BACKWARD=False
CACHE_DTYPE=torch.float32;V_DTYPE=torch.float32
CUBIC_ONLY=False
ORIGINAL_ROUTED=core.RoutedKS3

def tiles(m,n):return (128,128,64,8,3) if m<=32768 else (64,256,64,8,4)

def mm(a,w,table,dtype=torch.float32):
    m,k=a.shape;n=w.shape[-1];bm,bn,bk,nw,st=tiles(m,n)
    out=torch.empty(m,n,device=a.device,dtype=dtype)
    gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,w,out,table,m,n,k,*w.stride(),bm,bn,bk,num_warps=nw,num_stages=st)
    return out

def dw(a,b,out,offs):
    e,h,d=out.shape;bk=64 if a.shape[0]<=32768 else 128
    core._dw[(tr.cdiv(h,128),tr.cdiv(d,128),e)](a,b,out,offs,h,d,False,BM=128,BN=128,BK=bk,num_warps=8,num_stages=3)

@torch.library.custom_op('h100_ks3::forward',mutates_args=())
def forward(rows:torch.Tensor,src:torch.Tensor,table:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,scale:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    m=src.numel();n=up.shape[1];d=768;rank=(g.shape[-1]-2)//6
    if rank!=1 and not (PREPARE and SEPARATE and not FUSE_VJP):
        raise ValueError('Higher rank requires separate prepared geometry')
    x=torch.empty(m,d,device=rows.device,dtype=torch.bfloat16)
    if not PACK_INPUT:
        core._pack[(tr.cdiv(m,128),24)](rows,x,m,d,*rows.stride(),SRC=src,INPUT_ROWS=rows.shape[0],ROWS_PER_TILE=128,GS32=True,num_warps=4)
    prepared=torch.empty(8,n,9,device=rows.device)
    if PREPARE:
        from wr_reproduction.ks3_prepared_geometry import _prepare
        if rank==1:_prepare[(tr.cdiv(8*n,256),)](g,prepared,8*n)
        else:
            from experiments.h100_ks3.prepared_rank import prepare
            prepare[(tr.cdiv(8*n,256),)](g,prepared,8*n,rank)
    c=torch.empty(m,n,device=rows.device);h=torch.empty_like(c,dtype=torch.bfloat16)
    bm,bn,bk,nw,st=(64,128,64,4,4)
    w=up.transpose(1,2)
    if SEPARATE:
        c=mm(x,w,table,dtype=CACHE_DTYPE)
        from wr_reproduction.ks3_prepared_geometry import _geometry_prepared
        if FAST_TANH_FORWARD:
            from experiments.h100_ks3.native_geometry import _geometry_prepared
        extra={'CUBIC_ONLY':CUBIC_ONLY} if FAST_TANH_FORWARD else {}
        _geometry_prepared[(tr.cdiv(m,64),tr.cdiv(n,32))](c,g,h,table,m,n,BM=64,RANK=rank,num_warps=4,PREPARED=prepared,**extra)
    else:
        if 'f128' in VARIANT:bm,bn,bk,nw,st=128,128,64,8,3
        gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](rows if PACK_INPUT else x,w,c,table,m,n,d,*w.stride(),bm,bn,bk,1,G=g,H=h,SRC=src,ROWS=rows.shape[0],PACK_INPUT=PACK_INPUT,XPACK=x,PREPARED=prepared if PREPARE else None,NATIVE_TANH=FAST_TANH_FORWARD,num_warps=nw,num_stages=st)
    y=torch.empty(m,d,device=rows.device,dtype=torch.bfloat16);out=torch.empty_like(rows)
    bm,bn,bk,nw,st=tiles(m,d);w=down.transpose(1,2)
    gemm[(tr.cdiv(m,bm),tr.cdiv(d,bn))](h,w,y,table,m,d,n,*w.stride(),bm,bn,bk,2,SRC=src,GATE=scale,SCATTER=out,ROWS=rows.shape[0],num_warps=nw,num_stages=st)
    return out,y,x,c,h,prepared

@forward.register_fake
def _(rows,src,table,up,down,g,scale):
    m=src.numel();n=up.shape[1];return torch.empty_like(rows),rows.new_empty((m,768)),rows.new_empty((m,768)),rows.new_empty((m,n),dtype=CACHE_DTYPE),rows.new_empty((m,n)),g.new_empty((8,n,9))

@torch.library.custom_op('h100_ks3::backward_core',mutates_args={'bank'})
def backward_core(grad:torch.Tensor,y:torch.Tensor,x:torch.Tensor,c:torch.Tensor,h:torch.Tensor,scale:torch.Tensor,src:torch.Tensor,offs:torch.Tensor,table:torch.Tensor,up:torch.Tensor,down:torch.Tensor,g:torch.Tensor,prepared:torch.Tensor,bank:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    m,n=c.shape;rank=(g.shape[-1]-2)//6
    if FAST_GATE:
        gg=torch.empty(m,768,device=c.device,dtype=torch.bfloat16);ds=torch.empty(m,device=c.device)
        gather_gate[(tr.cdiv(m,4),)](grad,y,src,scale,gg,ds,grad.shape[0],m,num_warps=4,enable_fp_fusion=False)
    else:gg,ds=gather(grad,y,src,scale)
    dc=torch.empty_like(c,dtype=torch.bfloat16)
    bm=VJP_ROWS
    if FUSE_VJP:
        bm,bn,bk,nw,st=(64,64,128,4,3);part=torch.empty(tr.cdiv(m,bm),n,8,device=c.device)
        gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](gg,down,dc,table,m,n,768,*down.stride(),bm,bn,bk,3,G=g,SRC=src,ROWS=grad.shape[0],C=c,PART=part,num_warps=nw,num_stages=st)
    else:
        v=mm(gg,down,table,dtype=V_DTYPE);part=torch.empty(tr.cdiv(m,bm),n,8,device=c.device)
        if PREPARE:
            from wr_reproduction.ks3_prepared_geometry import _vjp_prepared
            if FAST_TANH_BACKWARD:
                from experiments.h100_ks3.native_geometry import _vjp_prepared
            extra={'CUBIC_ONLY':CUBIC_ONLY} if FAST_TANH_BACKWARD else {}
            _vjp_prepared[(tr.cdiv(m,bm),tr.cdiv(n,32))](v,c,g,dc,part,table,src,m,n,grad.shape[0],BM=bm,RANK=rank,num_warps=4,PREPARED=prepared,**extra)
        else:core._vjp[(tr.cdiv(m,bm),tr.cdiv(n,32))](v,c,g,dc,part,table,src,m,n,grad.shape[0],BM=bm,RANK=1,num_warps=4)
    dg=torch.empty_like(g);core._reduce[(8,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),BM=bm,RANK=rank,num_warps=4)
    dw(h,gg,bank[(2*layer+1)*8:(2*layer+2)*8],offs);dw(dc,x,bank[2*layer*8:(2*layer+1)*8],offs)
    if FUSE_DX:return dc,ds,dg
    return mm(dc,up,table,dtype=grad.dtype),ds,dg

@backward_core.register_fake
def _(grad,y,x,c,h,scale,src,offs,table,up,down,g,prepared,bank,layer):
    return (torch.empty_like(c,dtype=torch.bfloat16) if FUSE_DX else torch.empty_like(x)),x.new_empty((x.shape[0],),dtype=torch.float32),torch.empty_like(g)

@torch.library.custom_op('h100_ks3::dx',mutates_args=())
def dx(dc:torch.Tensor,up:torch.Tensor,table:torch.Tensor,src:torch.Tensor,dl:torch.Tensor,router:torch.Tensor)->torch.Tensor:
    m,k=dc.shape;n=768;bm,bn,bk,nw,st=tiles(m,n);out=torch.empty(dl.shape[0],n,device=dc.device,dtype=torch.bfloat16)
    gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](dc,up,out,table,m,n,k,*up.stride(),bm,bn,bk,4,SRC=src,ROWS=dl.shape[0],DLOG=dl,ROUTER=router,num_warps=nw,num_stages=st)
    return out
@dx.register_fake
def _(dc,up,table,src,dl,router):return dc.new_empty((dl.shape[0],768))

class Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,router,post,w1,w2,up,down,g,bank,layer,gs32=True):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
        gate,expert=probs.max(-1);dest,src,offs,table,counts=moe.moe_dispatch(expert,8)
        scale=gate*post.float().reshape(())
        out,y,x,c,h,prepared=forward(rows,src,table,up,down,g,scale)
        ctx.save_for_backward(rows,router,post,probs,gate,expert,dest,src,offs,table,y,x,c,h,scale,g,prepared)
        ctx.up=up;ctx.down=down;ctx.bank=bank;ctx.layer=layer;ctx.argument_count=len(ctx.needs_input_grad)
        ctx.mark_non_differentiable(counts)
        return out,probs,counts
    @staticmethod
    def backward(ctx,grad,dprobs,dcounts):
        from track_1_short.perf.kernels import moe_mlp as moe
        rows,router,post,probs,gate,expert,dest,src,offs,table,y,x,c,h,scale,g,prepared=ctx.saved_tensors
        dc,ds,dg=backward_core(grad.contiguous(),y,x,c,h,scale,src,offs,table,ctx.up,ctx.down,g,prepared,ctx.bank,ctx.layer)
        ds=ds[dest.long()];dp=torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp=dp.scatter_add(1,expert[:,None],(ds*post.float()).reshape(-1,1))
        dl=(probs*(dp-(probs*dp).sum(-1,keepdim=True))).to(torch.bfloat16)
        result=dx(dc,ctx.up,table,src,dl,router) if FUSE_DX else moe.dx_combine(dc,dest,dl.float(),router)
        return (result,dl.T@rows,(ds*gate).sum().to(post.dtype).reshape(post.shape),None,None,None,None,dg,None,None,None)[:ctx.argument_count]

def install(variant='fused'):
    global VARIANT,PACK_INPUT,FUSE_VJP,FUSE_DX,PREPARE,FAST_GATE,VJP_ROWS,SEPARATE,FAST_TANH_FORWARD,FAST_TANH_BACKWARD,CACHE_DTYPE,V_DTYPE,CUBIC_ONLY
    VARIANT=variant;PACK_INPUT='input' in variant;FUSE_VJP='vjp' in variant;FUSE_DX='dx' in variant
    SEPARATE='separate' in variant
    PREPARE='prepared' in variant;FAST_GATE='gate' in variant;VJP_ROWS=128 if '128' in variant else 64
    FAST_TANH_BACKWARD='tanh' in variant;FAST_TANH_FORWARD=FAST_TANH_BACKWARD and 'tanhb' not in variant
    CACHE_DTYPE=torch.float16 if 'fp16cache' in variant else torch.bfloat16 if 'bf16cache' in variant else torch.float32
    V_DTYPE=torch.float16 if 'fp16cache' in variant or 'fp16v' in variant else torch.bfloat16 if 'bf16cache' in variant or 'bf16v' in variant else torch.float32
    CUBIC_ONLY='cubic' in variant
    if CUBIC_ONLY and not (FAST_TANH_FORWARD and PREPARE and SEPARATE and not FUSE_VJP):
        raise ValueError('Cubic-only KS3 requires separate native prepared geometry')
    if CACHE_DTYPE!=torch.float32 and not SEPARATE:
        raise ValueError('Reduced preactivation cache requires separate geometry')
    if V_DTYPE!=torch.float32 and not (FAST_TANH_FORWARD and PREPARE and not FUSE_VJP):
        raise ValueError('Reduced VJP input cache requires native prepared geometry')
    if FAST_TANH_BACKWARD and not (PREPARE and not FUSE_VJP):
        raise ValueError('Native tanh variants require prepared geometry and separate VJP')
    matrix.run=mm;matrix.dw=dw
    if variant=='tiles':core.RoutedKS3=ORIGINAL_ROUTED
    elif variant=='gather':
        core.RoutedKS3=ORIGINAL_ROUTED
        from wr_reproduction.ks3_bf16_gather import install as use_gather
        use_gather()
    else:core.RoutedKS3=Routed
