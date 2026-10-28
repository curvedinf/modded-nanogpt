"""H100 KS3 FP8 products with fused producers and reusable dual layouts.

Geometry remains the checked FP16-forward/FP32-STE formulation. Evaluation
uses the preserved BF16 path. Upstream delayed activation/gradient scaling
and owner-sharded weight refresh remain on the original training clock.
"""
import os
import torch,triton as tr
from experiments.h100_ks3.h100_kernels import gemm
from experiments.h100_ks3.fp8_kernels import publish,layouts,weight_gradient,gather_gate,pack_input
from experiments.h100_ks3.native_geometry import _geometry_prepared,_vjp_prepared
from wr_reproduction.ks3_prepared_geometry import _prepare
from wr_reproduction import ks3_moe as core

DW_IMPL=os.environ.get('KS3_FP8_DW','legacy')
VJP_ROWS=int(os.environ.get('KS3_FP8_VJP_ROWS','128'))
VJP_WARPS=int(os.environ.get('KS3_FP8_VJP_WARPS','4'))
GATHER_IMPL=os.environ.get('KS3_FP8_GATHER','legacy')
TMA_STORES=os.environ.get('KS3_FP8_TMA_STORES','0')=='1'
V_DTYPE={'float32':torch.float32,'float16':torch.float16}[os.environ.get('KS3_FP8_V_DTYPE','float32')]
FUSE_GEOMETRY=os.environ.get('KS3_FP8_FUSE_GEOMETRY','0')=='1'
assert DW_IMPL in ('legacy','tma') and VJP_ROWS in (32,64,128)
assert VJP_WARPS in (4,8) and GATHER_IMPL in ('legacy','tiled')

def store_descriptors(row_major,column_major,rows,cols=32):
    if not TMA_STORES:return None,None
    from triton.tools.tensor_descriptor import TensorDescriptor
    assert row_major.shape[0]%rows==0 and row_major.shape[1]%cols==0
    return TensorDescriptor.from_tensor(row_major,[rows,cols]),TensorDescriptor.from_tensor(column_major,[cols,rows])

def tile(m):return (128,128,128,8,3) if m<=32768 else (64,256,128,8,4)
def mm(a,w,table,sa,sw,dtype=torch.float32,mode=0,**kw):
    m,k=a.shape;n=w.shape[-1];bm,bn,bk,nw,st=tile(m)
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,w,out,table,m,n,k,*w.stride(),bm,bn,bk,mode,ASCALE=sa,WSCALE=sw,num_warps=nw,num_stages=st,**kw)
    return out

def dw(a,b,out,offs,sa,sb,flag):
    e,h,d=out.shape
    if DW_IMPL=='tma':
        from track_1_short.perf.kernels.moe_mlp import _grouped_weight_grad_impl
        _grouped_weight_grad_impl(a,b,offs,sa,sb,e,out.view(e*h,d),flag)
        return
    weight_gradient[(tr.cdiv(h,128),tr.cdiv(d,128),e)](a,b,out,offs,sa,sb,flag,h,d,a.shape[1],num_warps=8,num_stages=3)

def gather(grad,y,src,scale,gg,gt,dgate,gs,rows,m):
    if GATHER_IMPL=='tiled':
        from experiments.h100_ks3.fp8_kernels import gather_gate_tiled
        gather_gate_tiled[(tr.cdiv(m,64),)](grad,y,src,scale,gg,gt,dgate,gs,rows,m,num_warps=4,enable_fp_fusion=False)
    else:
        gather_gate[(tr.cdiv(m,4),)](grad,y,src,scale,gg,gt,dgate,gs,rows,m,num_warps=4,enable_fp_fusion=False)

@torch.library.custom_op('h100_ks3_fp8::forward',mutates_args={'pa'})
def forward(rows:torch.Tensor,src:torch.Tensor,table:torch.Tensor,up:torch.Tensor,down:torch.Tensor,
            us:torch.Tensor,ws:torch.Tensor,xs:torch.Tensor,ps:torch.Tensor,pa:torch.Tensor,g:torch.Tensor,
            scale:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    m=src.numel();n=up.shape[1];d=768
    x=torch.empty(m,d,device=rows.device,dtype=torch.float8_e4m3fn);xt=torch.empty(d,m,device=rows.device,dtype=x.dtype)
    pack_input[(tr.cdiv(m,128),24)](rows,x,xt,src,xs,m,rows.shape[0],num_warps=4)
    prepared=torch.empty(8,n,9,device=rows.device);_prepare[(tr.cdiv(8*n,256),)](g,prepared,8*n)
    h=torch.empty(m,n,device=rows.device,dtype=x.dtype);ht=torch.empty(n,m,device=rows.device,dtype=x.dtype)
    if FUSE_GEOMETRY:
        bm,bn,bk,nw,st=64,128,128,4,4
        c=torch.empty(m,n,device=rows.device,dtype=torch.float32);weight=up.transpose(1,2)
        maxima=torch.empty(tr.cdiv(m,bm)*tr.cdiv(n,bn),device=rows.device)
        hd,htd=store_descriptors(h,ht,bm,bn)
        gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](x,weight,c,table,m,n,d,*weight.stride(),bm,bn,bk,1,G=g,H=h,PREPARED=prepared,ASCALE=xs,WSCALE=us,NATIVE_TANH=True,H_SCALE=ps,H_TRANSPOSE=ht,H_DESC=hd,H_TRANSPOSE_DESC=htd,H_AMAX=maxima,num_warps=nw,num_stages=st)
    else:
        c=mm(x,up.transpose(1,2),table,xs,us)
        maxima=torch.empty(tr.cdiv(m,64)*tr.cdiv(n,32),device=rows.device)
        hd,htd=store_descriptors(h,ht,64)
        _geometry_prepared[(tr.cdiv(m,64),tr.cdiv(n,32))](c,g,h,table,m,n,HT=ht,SCALE=ps,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=64,RANK=1,PREPARED=prepared,H_DESC=hd,HT_DESC=htd,num_warps=4)
    torch.maximum(pa,maxima.amax(),out=pa)
    out=torch.empty_like(rows)
    y=mm(h,down.transpose(1,2),table,ps,ws,dtype=rows.dtype,mode=2,SRC=src,GATE=scale,SCATTER=out,ROWS=rows.shape[0])
    return out,y,c,xt,ht,prepared

@forward.register_fake
def _(rows,src,table,up,down,us,ws,xs,ps,pa,g,scale):
    m=src.numel();n=up.shape[1]
    return torch.empty_like(rows),rows.new_empty((m,768)),rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((768,m),dtype=torch.float8_e4m3fn),rows.new_empty((n,m),dtype=torch.float8_e4m3fn),g.new_empty((8,n,9))

@torch.library.custom_op('h100_ks3_fp8::backward_core',mutates_args={'bank','da'})
def backward_core(grad:torch.Tensor,y:torch.Tensor,c:torch.Tensor,xt:torch.Tensor,ht:torch.Tensor,scale:torch.Tensor,
                  src:torch.Tensor,offs:torch.Tensor,table:torch.Tensor,down:torch.Tensor,g:torch.Tensor,prepared:torch.Tensor,
                  us:torch.Tensor,ws:torch.Tensor,xs:torch.Tensor,gs:torch.Tensor,ps:torch.Tensor,ds:torch.Tensor,da:torch.Tensor,
                  bank:torch.Tensor,flag:torch.Tensor,layer:int)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    m,n=c.shape
    gg=torch.empty(m,768,device=c.device,dtype=torch.float8_e5m2);gt=torch.empty(768,m,device=c.device,dtype=gg.dtype)
    dgate=torch.empty(m,device=c.device)
    gather(grad,y,src,scale,gg,gt,dgate,gs,grad.shape[0],m)
    v=mm(gg,down,table,gs,ws,dtype=V_DTYPE)
    dc=torch.empty(m,n,device=c.device,dtype=torch.float8_e5m2);dct=torch.empty(n,m,device=c.device,dtype=dc.dtype)
    part=torch.empty(tr.cdiv(m,VJP_ROWS),n,8,device=c.device);maxima=torch.empty(tr.cdiv(m,VJP_ROWS)*tr.cdiv(n,32),device=c.device)
    dcd,dctd=store_descriptors(dc,dct,VJP_ROWS)
    _vjp_prepared[(tr.cdiv(m,VJP_ROWS),tr.cdiv(n,32))](v,c,g,dc,part,table,src,m,n,grad.shape[0],DCT=dct,SCALE=ds,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=VJP_ROWS,RANK=1,PREPARED=prepared,DC_DESC=dcd,DCT_DESC=dctd,num_warps=VJP_WARPS)
    torch.maximum(da,maxima.amax(),out=da)
    dg=torch.empty_like(g);core._reduce[(8,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),BM=VJP_ROWS,RANK=1,num_warps=4)
    dw(ht,gt,bank[(2*layer+1)*8:(2*layer+2)*8],offs,ps,gs,flag)
    dw(dct,xt,bank[2*layer*8:(2*layer+1)*8],offs,ds,xs,flag)
    return dc,dgate,dg

@backward_core.register_fake
def _(grad,y,c,xt,ht,scale,src,offs,table,down,g,prepared,us,ws,xs,gs,ps,ds,da,bank,flag,layer):
    return c.new_empty(c.shape,dtype=torch.float8_e5m2),c.new_empty((c.shape[0],)),torch.empty_like(g)

@torch.library.custom_op('h100_ks3_fp8::dx',mutates_args=())
def dx(dc:torch.Tensor,up:torch.Tensor,table:torch.Tensor,src:torch.Tensor,dl:torch.Tensor,router:torch.Tensor,ds:torch.Tensor,us:torch.Tensor)->torch.Tensor:
    m,k=dc.shape;n=768;bm,bn,bk,nw,st=tile(m)
    out=torch.empty(dl.shape[0],n,device=dc.device,dtype=torch.bfloat16)
    gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](dc,up,out,table,m,n,k,*up.stride(),bm,bn,bk,4,SRC=src,ROWS=dl.shape[0],DLOG=dl,ROUTER=router,ASCALE=ds,WSCALE=us,num_warps=nw,num_stages=st)
    return out
@dx.register_fake
def _(dc,up,table,src,dl,router,ds,us):return dc.new_empty((dl.shape[0],768),dtype=torch.bfloat16)

class Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rows,router,post,w1,w2,g,caches,scales,bank,flag,layer):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
        gate,expert=probs.max(-1);dest,src,offs,table,counts=moe.moe_dispatch(expert,8)
        scale=gate*post.float().reshape(())
        up,upt,down,downt=caches;us,ws,xs,gs,ps,pa,ds,da=scales
        out,y,c,xt,ht,prepared=forward(rows,src,table,up,down,us,ws,xs,ps,pa,g,scale)
        ctx.save_for_backward(rows,router,post,probs,gate,expert,dest,src,offs,table,y,c,xt,ht,scale,g,prepared)
        ctx.caches=caches;ctx.scales=scales;ctx.bank=bank;ctx.flag=flag;ctx.layer=layer
        ctx.mark_non_differentiable(counts)
        return out,probs,counts
    @staticmethod
    def backward(ctx,grad,dprobs,dcounts):
        rows,router,post,probs,gate,expert,dest,src,offs,table,y,c,xt,ht,scale,g,prepared=ctx.saved_tensors
        up,upt,down,downt=ctx.caches;us,ws,xs,gs,ps,pa,ds,da=ctx.scales
        dc,dsp,dg=backward_core(grad.contiguous(),y,c,xt,ht,scale,src,offs,table,downt.transpose(1,2),g,prepared,us,ws,xs,gs,ps,ds,da,ctx.bank,ctx.flag,ctx.layer)
        ds_gate=dsp[dest.long()];dp=torch.zeros_like(probs) if dprobs is None else dprobs.float()
        dp=dp.scatter_add(1,expert[:,None],(ds_gate*post.float()).reshape(-1,1))
        dl=(probs*(dp-(probs*dp).sum(-1,keepdim=True))).to(torch.bfloat16)
        result=dx(dc,upt.transpose(1,2),table,src,dl,router,ds,us)
        return result,dl.T@rows,(ds_gate*gate).sum().to(post.dtype).reshape(post.shape),None,None,dg,None,None,None,None,None


def attach(base):
    import torch.distributed as dist
    from triton.backends.nvidia.compiler import get_ptxas
    if tuple(map(int,get_ptxas(90).version.split('.'))) < (13,3):
        raise RuntimeError('H100 FP8 dual stores require ptxas >=13.3; run setup_ptxas and set TRITON_PTXAS_PATH. See triton-lang/triton#11111.')
    from track_1_short.model.moe import MOE_INDEX
    from track_1_short.perf.cuda_graphs import fp8_refresh_graphs as refresh
    class FP8GPT(base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            from wr_reproduction.ks3_moe_fp8 import init_buffers
            init_buffers(self)
            n=self.ks3_up.shape[-2]
            self.register_buffer('ks3_fp8_transport',torch.empty((80,n*768),device=self.device,dtype=torch.float8_e4m3fn),persistent=False)
            self.register_buffer('ks3_fp8_partial',torch.ones((80,n*768//4096),device=self.device),persistent=False)
        @torch.no_grad()
        def _quantize_moe(self,lo,hi,exact):
            n=self.ks3_up.shape[-2];scales=self.ks3_weight_scales.view(-1)
            if exact:
                flat=self.moe_bank[lo:hi].reshape(hi-lo,-1)
                maximum=torch.maximum(flat.amax(1),-flat.amin(1)).float()
            else:maximum=self.ks3_fp8_partial[lo:hi].amax(1)
            scales[lo:hi].copy_(maximum.clamp_min(1e-12)/448.)
            publish[(n*768//4096,hi-lo)](self.moe_bank,self.ks3_fp8_transport,scales,self.ks3_fp8_partial,lo,n,num_warps=4)
            if lo==0 and hi==80:self.build_moe_cols()
        def gather_moe_f8(self):
            lo,hi=self._moe_lo,self._moe_hi;q=self.ks3_fp8_transport.view(torch.uint8);s=self.ks3_weight_scales.view(-1)
            return [('moe_bank',dist.all_gather_into_tensor(q,q[lo:hi],async_op=True).get_future()),('moe_scales',dist.all_gather_into_tensor(s,s[lo:hi],async_op=True).get_future())]
        @torch.no_grad()
        def build_moe_cols(self):
            n=self.ks3_up.shape[-2]
            layouts[(tr.cdiv(n*768,1024),80)](self.ks3_fp8_transport,self.ks3_f8_up,self.ks3_f8_up_t,self.ks3_f8_down,self.ks3_f8_down_t,n,num_warps=4)
        def _moe(self,layer,normed,weights,post):
            if not self.training:return super()._moe(layer,normed,weights,post)
            k=MOE_INDEX[layer];w1,w2,router=weights[k];rows=normed.reshape(-1,768)
            if post is None:post=rows.new_ones(())
            caches=(self.ks3_f8_up[k],self.ks3_f8_up_t[k],self.ks3_f8_down[k],self.ks3_f8_down_t[k])
            scales=(self.ks3_weight_scales[k,0],self.ks3_weight_scales[k,1],self._mlp_x_scale,self._mlp_grad_scale,self._moe_post_scales[k:k+1],self._moe_post_amax[k],self._moe_dpre_scales[k:k+1],self._moe_dpre_amax[k])
            out,probs,counts=Routed.apply(rows,router,post,w1,w2,self.ks3_geometry[k],caches,scales,self.moe_bank_grad,self.moe_grad_accumulate,k)
            return out.reshape(normed.shape),8*(counts.float()/rows.shape[0]*probs.mean(0)).sum()
    refresh.MOE_LOCAL_OUTPUTS=('ks3_fp8_transport','ks3_weight_scales','ks3_fp8_partial')
    refresh.MOE_COLS_OUTPUTS=('ks3_f8_up','ks3_f8_up_t','ks3_f8_down','ks3_f8_down_t')
    return FP8GPT
