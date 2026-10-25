"""Numerical and paired GPU-time comparison of geometry backward tile sizes."""
import statistics
import torch
import triton as tr
from wr_reproduction import ks3_moe as bf,ks3_moe_fp8 as fp


@torch.no_grad()
def sweep(charts=1024,gs32=True):
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(169);torch.set_num_threads(1)
    device='cuda';e,n,d=8,charts,768;f8=torch.float8_e4m3fn
    up=(torch.randn(e,n,d,device=device)*.025).bfloat16()
    down=(torch.randn(e,d,n,device=device)*.02).bfloat16()
    g=torch.randn(e,n,8,device=device)*.01;g[...,6]=.75
    caches=[];scales=[]
    for weight in (up,down):
        q=torch.empty_like(weight,dtype=f8)
        qt=torch.empty(weight.shape[0],weight.shape[2],weight.shape[1],device=device,dtype=f8)
        scale=weight.float().abs().amax((1,2))/448
        for expert in range(e):
            a=weight[expert];r,c=a.shape
            bf._pack[(tr.cdiv(r,8),tr.cdiv(c,32))](a,q[expert],r,c,*a.stride(),
                SCALE=scale[expert:expert+1],OT=qt[expert],FP8=True,GS32=gs32)
        caches.append((q,qt));scales.append(scale)
    (u,ut),(w,wt)=caches;us,ws=scales
    scalar=lambda v:torch.tensor([v],device=device,dtype=torch.float32)
    xs,ps,gs,ds=map(scalar,(2**-4,.01,2**-6,.0001))
    original=fp.GEOMETRY_BACKWARD_ROWS;cases=[]
    try:
        for tokens in (1,257,16384,49152):
            x=torch.randn(tokens,d,device=device,dtype=torch.bfloat16)
            grad=torch.randn_like(x)*.02;expert=torch.randint(e,(tokens,),device=device)
            if tokens==257:expert.zero_()  # empty experts and a partial routing tile
            dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
            pa,da=scalar(0),scalar(0)
            y,c,xt,ht=fp.forward(x,src,tiles,u,w,us,ws,xs,ps,pa,g,gs32)
            gate=torch.ones(tokens,device=device)
            bank=torch.zeros(16,n,d,device=device,dtype=torch.bfloat16)
            flag=torch.zeros(1,device=device,dtype=torch.int32)
            def backward():
                return fp.backward(grad,y,c,xt,ht,gate,src,offs,tiles,ut,wt,
                                   us,ws,xs,gs,ps,ds,da,bank,flag,g,0)
            fp.GEOMETRY_BACKWARD_ROWS=64
            reference=[v.clone() for v in backward()]+[bank.clone()]
            errors={};samples={str(tile):[] for tile in (32,64,128)}
            for tile in (32,64,128):
                fp.GEOMETRY_BACKWARD_ROWS=tile
                actual=list(backward())+[bank]
                errors[str(tile)]={}
                for name,a,b in zip(('dx','dgate','geometry','weight_gradient'),actual,reference):
                    assert a.isfinite().all(),(tokens,tile,name)
                    error=((a.float()-b.float()).square().mean().sqrt()/
                           b.float().square().mean().sqrt().clamp_min(1e-12)).item()
                    assert error<.0005,(tokens,tile,name,error)
                    errors[str(tile)][name]=error
            if tokens>=16384:
                for order in ((64,128,32),(32,64,128),(128,32,64)):
                    for tile in order:
                        fp.GEOMETRY_BACKWARD_ROWS=tile
                        samples[str(tile)].append(tr.testing.do_bench_cudagraph(backward,rep=100))
            cases.append(dict(tokens=tokens,relative_rms_vs_64=errors,backward_ms=samples,
                              median_ms={k:statistics.median(v) for k,v in samples.items() if v}))
    finally:fp.GEOMETRY_BACKWARD_ROWS=original
    return dict(cases=cases,scope='Expert backward only; no whole-update speed or convergence claim')
