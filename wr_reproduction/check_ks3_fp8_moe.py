"""Independent decoded-FP8 / upstream geometry oracle for the fused KS3 port."""
import sys,json,argparse,torch,triton as tr
import triton.language as tl
from pathlib import Path
from wr_reproduction.contracts import ROOT
sys.path.insert(0,str(ROOT/'pr393/upstream'))
from track_1_short.perf.kernels import moe_mlp as moe
from wr_reproduction.ks3_moe import _pack
from wr_reproduction.ks3_moe_fp8 import forward,backward
from ks3_research.vendor import ks3_int8 as original
from control393.ks3_dynamic import geometry_vjp,_reduce_geometry

def gs(a):
    r,n=a.shape;pad=torch.nn.functional.pad(a.float(),(0,(-n)%32)).reshape(r,-1,32)
    s=(pad.abs().amax(-1)/127).clamp(2**-24,65504).half().float()
    q=(pad/s[...,None]).round().clamp(-127,127)
    return (q*s[...,None]).reshape(r,-1)[:,:n],q.reshape(r,-1)[:,:n].to(torch.int8),s.half()

def quant(a,s,dtype):return (a/s).clamp(-torch.finfo(dtype).max,torch.finfo(dtype).max).to(dtype)

@tr.jit
def _raw_upstream_spine(C,P,PS,SCALE,SHIFT,H,M:tl.constexpr,N:tl.constexpr):
    """Pinned upstream FP16 nonlinearity, before either operand quantizer."""
    rows=tl.program_id(0)*16+tl.arange(0,16)
    cols=tl.program_id(1)*32+tl.arange(0,32);valid=cols<N
    x=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    c0=tl.load(P+cols*6+0,valid,other=0.).to(tl.float32)
    c1=tl.load(P+cols*6+1,valid,other=0.).to(tl.float32)
    c2=tl.load(P+cols*6+2,valid,other=0.).to(tl.float32)
    c3=tl.load(P+cols*6+3,valid,other=0.).to(tl.float32)
    c4=tl.load(P+cols*6+4,valid,other=0.).to(tl.float32)
    c5=tl.load(P+cols*6+5,valid,other=0.).to(tl.float32)
    ps=tl.load(PS+cols,valid,other=1.).to(tl.float32)
    scale=tl.load(SCALE+cols,valid,other=1.)
    shift=tl.load(SHIFT+cols,valid,other=0.)
    value=original._spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,P,valid,1)
    tl.store(H+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&valid[None,:])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-pack',action='store_true')
    parser.add_argument('--batched-weight-pack',action='store_true')
    parser.add_argument('--prepared-native-weights',action='store_true')
    parser.add_argument('--prepared-native-activations',action='store_true')
    parser.add_argument('--prepared-native-dual',action='store_true')
    parser.add_argument('--prepared-native-gather',action='store_true')
    parser.add_argument('--native-tile-policy',choices=['legacy','mi300-v2'],default='legacy')
    parser.add_argument('--geometry-backward-rows',type=int,choices=[32,64,128],default=64)
    parser.add_argument('--sweep-geometry-backward',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--charts',type=int,choices=[512,768,1024,1536],default=1024)
    args=parser.parse_args();gs32=not args.native_pack
    from wr_reproduction import ks3_moe_fp8,ks3_fp8_gemm
    ks3_fp8_gemm.TILE_MODE=args.native_tile_policy
    ks3_moe_fp8.GEOMETRY_BACKWARD_ROWS=args.geometry_backward_rows
    ks3_moe_fp8.PREPARED_ACTIVATIONS=args.prepared_native_activations
    ks3_moe_fp8.PREPARED_DUAL=args.prepared_native_dual
    ks3_moe_fp8.PREPARED_GATHER=args.prepared_native_gather
    torch.manual_seed(169);torch.set_num_threads(1);dev='cuda';E,N,D=8,args.charts,768
    up=(torch.randn(E,N,D,device=dev)*.025).bfloat16();down=(torch.randn(E,D,N,device=dev)*.02).bfloat16()
    g=torch.randn(E,N,8,device=dev)*.01;g[...,6]=.75
    f8=torch.float8_e4m3fn;e5=torch.float8_e5m2
    caches=[];scales=[]
    for w in (up,down):
        q=torch.empty_like(w,dtype=f8);qt=torch.empty(w.shape[0],w.shape[2],w.shape[1],device=dev,dtype=f8)
        s=w.float().abs().amax((1,2))/448
        for e in range(E):
            a=w[e];r,n=a.shape
            _pack[(tr.cdiv(r,8),tr.cdiv(n,32))](a,q[e],r,n,*a.stride(),SCALE=s[e:e+1],OT=qt[e],FP8=True,GS32=gs32)
            want=quant(gs(a)[0] if gs32 else a.float(),s[e],f8)
            assert torch.equal(q[e].view(torch.uint8),want.view(torch.uint8))
            assert torch.equal(qt[e].view(torch.uint8),q[e].T.contiguous().view(torch.uint8))
        caches.append((q,qt));scales.append(s)
    (u,ut),(w,wt)=caches;us,ws=scales
    actual=(u,ut,w,wt)
    if args.prepared_native_weights:
        from wr_reproduction.check_ks3_prepared_native import native_copy
        actual=tuple(native_copy(a) for a in actual)
    au,aut,aw,awt=actual
    scalar=lambda x:torch.tensor([x],device=dev,dtype=torch.float32)
    xs,ps,gscl,ds=map(scalar,(2**-4,.01,2**-6,.0001))
    lut=torch.zeros(256,device=dev,dtype=torch.int8);checks=[]
    for tokens in (1,257,4096):
        x=torch.randn(tokens,D,device=dev,dtype=torch.bfloat16);grad=torch.randn_like(x)*.02
        expert=torch.randint(E,(tokens,),device=dev)
        if tokens==257:expert.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,E)
        pa,da=scalar(0),scalar(0);flag=torch.zeros(1,device=dev,dtype=torch.int32)
        y,c,xt,ht=forward(x,src,tiles,au,aw,us,ws,xs,ps,pa,g,gs32)
        bank=torch.zeros(16,N,D,device=dev,dtype=torch.bfloat16)
        dx,dgate,dg=backward(grad,y,c,xt,ht,torch.ones(tokens,device=dev),src,offs,tiles,aut,awt,us,ws,xs,gscl,ps,ds,da,bank,flag,g,0)
        yr=torch.zeros_like(x);dxr=torch.zeros_like(x);br=torch.zeros_like(bank);gr=torch.zeros_like(g)
        for e in range(E):
            ids=(expert==e).nonzero().flatten();m=ids.numel()
            if not m:continue
            xx=quant(gs(x[ids])[0] if gs32 else x[ids].float(),xs,f8)
            coords=(xx.float()@u[e].float().T)*(xs*us[e])
            _,p,p_scale=gs(g[e,:,:6]);p=p.contiguous();p_scale=p_scale.flatten()
            scale=torch.nn.functional.softplus(g[e,:,6])+1e-4;shift=.5*g[e,:,7].tanh()
            h=torch.empty(m,N,device=dev,dtype=torch.int8);hs=torch.empty(m,N//32,device=dev,dtype=torch.float16)
            original._geometry_quant[(tr.cdiv(m,16),tr.cdiv(N,32))](coords,coords,p,p_scale,scale,shift,lut,h,hs,m,N,1,1,BM=16,BN=32,num_warps=4)
            if gs32:
                hq=quant(h.float()*hs.float().repeat_interleave(32,-1),ps,f8)
            else:
                raw=torch.empty(m,N,device=dev,dtype=torch.float32)
                _raw_upstream_spine[(tr.cdiv(m,16),tr.cdiv(N,32))](coords,p,p_scale,scale,shift,raw,m,N,num_warps=4)
                hq=quant(raw,ps,f8)
            yr[ids]=((hq.float()@w[e].float().T)*(ps*ws[e])).bfloat16()
            gg=quant(grad[ids].float(),gscl,e5)
            v=(gg.float()@wt[e].float().T)*(gscl*ws[e])
            part=torch.empty(tr.cdiv(m,64),N,8,device=dev)
            geometry_vjp[(tr.cdiv(m,64),tr.cdiv(N,32))](v,coords,p,p_scale,scale,shift,g[e,:,6].contiguous(),g[e,:,7].contiguous(),part,coords,v,1,m,N,BM=64,BN=32,num_warps=4)
            dc=quant(v,ds,e5)
            dxr[ids]=((dc.float()@u[e].float())*(ds*us[e])).bfloat16()
            br[e]=((dc.float().T@xx.float())*(ds*xs)).bfloat16()
            br[E+e]=((hq.float().T@gg.float())*(ps*gscl)).bfloat16()
            gr[e]=part.sum(0)
        errors={}
        for name,a,b in [('output',y[dest.long()],yr),('dx',dx[dest.long()],dxr),('bank',bank,br),('geometry',dg,gr)]:
            err=((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
            assert a.isfinite().all(),name
            assert err<.005,(tokens,name,err)
            errors[name]=err
        checks.append(dict(tokens=tokens,relative_rms=errors,post_amax=pa.item(),dpre_amax=da.item()))
    result=dict(event='kernel_correctness',passed=True,training_projection_gs32=gs32,
                coefficient_gs32=True,charts=N,geometry_backward_rows=args.geometry_backward_rows,
                checks=checks,full_convergence_verified=False)
    if args.prepared_native_weights:
        from wr_reproduction.check_ks3_prepared_native import check
        result['prepared_native_weights']=check(args.prepared_native_activations,args.prepared_native_dual,args.prepared_native_gather,gs32,args.charts)
    if args.batched_weight_pack:
        from wr_reproduction.check_ks3_weight_pack import check
        result['weight_packing']=check(N,gs32)
    if args.sweep_geometry_backward:
        from wr_reproduction.profile_ks3_geometry import sweep
        result['geometry_sweep']=sweep(N,gs32)
    if args.out:
        from datetime import datetime,timezone
        result['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as out:out.write(json.dumps(result)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
