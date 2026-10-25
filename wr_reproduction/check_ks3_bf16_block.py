"""Complete tuned-BF16 KS3 checks, exact producers, and paired core timing."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import torch
import triton as tr
from wr_reproduction.backend import install
from wr_reproduction import ks3_moe as bf
from wr_reproduction import ks3_bf16_gemm as gemm
from wr_reproduction.ks3_weight_pack import refresh_bf16


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--charts',type=int,choices=[256,512,768,1024,1280,1536],default=1024)
    parser.add_argument('--rank',type=int,choices=[1,2,4],default=1)
    parser.add_argument('--geometry-backward-rows',type=int,choices=[64],default=64)
    parser.add_argument('--tile-policy',choices=['mi300-v1','mi300-v2'],default='mi300-v1')
    args=parser.parse_args();install('pr393')
    if args.tile_policy=='mi300-v2' and (args.charts*args.rank!=1024 or args.rank not in (1,2)):
        raise ValueError('BF16-v2 block check requires1024 features with rank1/2')
    gemm.TILE_MODE=args.tile_policy
    from track_1_short.perf.kernels import moe_mlp as moe
    cls=bf.install('bf16',charts=args.charts,bf16_kernel_mode='tuned',subspace_rank=args.rank)
    stub=cls.__new__(cls);torch.nn.Module.__init__(stub)
    stub.train();assert bf.BF16_TUNED_KERNELS
    stub.eval();assert not bf.BF16_TUNED_KERNELS
    torch.manual_seed(169);torch.set_num_threads(1)
    rank=args.rank
    geometry_reference=None
    if rank!=1:
        from wr_reproduction.check_ks3_rank_geometry import check
        geometry_reference=check(rank,args.charts*rank)
    device='cuda';e,n,d=8,args.charts*rank,768
    m=torch.nn.Module();m.ks3_geometry=torch.zeros((5,e,args.charts,6*rank+2),device=device)
    m.moe_bank=(torch.randn(80,n,d,device=device)*.025).bfloat16()
    m.ks3_up=torch.empty(5,e,n,d,device=device,dtype=torch.bfloat16)
    m.ks3_down=torch.empty(5,e,d,n,device=device,dtype=torch.bfloat16)
    bank_view=m.moe_bank.view(5,2,e,n,d)
    def legacy_refresh():
        for layer in range(5):
            m.ks3_up[layer].copy_(bf.pack(bank_view[layer,0].reshape(e*n,d)).view(e,n,d))
            for expert in range(e):m.ks3_down[layer,expert].copy_(bf.pack(bank_view[layer,1,expert].T))
    legacy_refresh();wanted_up=m.ks3_up.clone();wanted_down=m.ks3_down.clone()
    refresh_bf16(m)
    assert torch.equal(m.ks3_up,wanted_up) and torch.equal(m.ks3_down,wanted_down)
    refresh_samples={name:[] for name in ('legacy','batched')}
    for order in (('legacy','batched'),('batched','legacy'),('legacy','batched')):
        for name in order:
            fn=legacy_refresh if name=='legacy' else lambda:refresh_bf16(m)
            refresh_samples[name].append(tr.testing.do_bench_cudagraph(fn,rep=50))
    up,down=m.ks3_up[0],m.ks3_down[0]
    g=torch.randn(e,args.charts,6*rank+2,device=device)*.01;g[...,6*rank]=.75
    checks=[]
    for tokens in (1,257,16384,49152):
        x=torch.randn(tokens,d,device=device,dtype=torch.bfloat16);grad=torch.randn_like(x)*.02
        expert=torch.randint(e,(tokens,),device=device)
        if tokens==257:expert.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
        gate=torch.ones(tokens,device=device)
        bank=torch.zeros(16,n,d,device=device,dtype=torch.bfloat16)
        def execute():
            y,xp,c,h=bf.forward(x,src,tiles,up,down,g)
            dx,ds,dg=bf.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,bank,0)
            return y,xp,dx,ds,dg
        bf.BF16_TUNED_KERNELS=False
        y,xp,dx,ds,dg=execute()
        expected=[y[dest.long()].clone(),dx[dest.long()].clone(),ds[dest.long()].clone(),dg.clone(),bank.clone()]
        expected_input=xp.clone()
        bf.BF16_TUNED_KERNELS=True
        gemm.TILE_MODE=args.tile_policy
        y,xp,dx,ds,dg=execute()
        assert torch.equal(xp,expected_input),'fused input producer changed GS32 values or padding'
        errors={}
        for name,a,b in zip(('output','dx','dgate','geometry','weight_gradient'),
                (y[dest.long()],dx[dest.long()],ds[dest.long()],dg,bank),expected):
            assert a.isfinite().all(),(tokens,name)
            error=((a.float()-b.float()).square().mean().sqrt()/
                   b.float().square().mean().sqrt().clamp_min(1e-12)).item()
            assert error<.005,(tokens,name,error)
            errors[name]=error
        labels=('v1','v2') if args.tile_policy=='mi300-v2' else ('legacy','tuned')
        samples={name:[] for name in labels}
        if tokens>=16384:
            for order in (labels,labels[::-1],labels):
                for name in order:
                    bf.BF16_TUNED_KERNELS=name!='legacy'
                    gemm.TILE_MODE='mi300-v2' if name=='v2' else 'mi300-v1'
                    samples[name].append(tr.testing.do_bench_cudagraph(execute,rep=100))
        checks.append(dict(tokens=tokens,input_pack_bitwise_equal=True,relative_rms=errors,
                           core_samples_ms=samples,median_ms={k:statistics.median(v) for k,v in samples.items() if v}))
        print(json.dumps(dict(event='block_check_progress',**checks[-1])),flush=True)
    bf.BF16_TUNED_KERNELS=False
    gemm.TILE_MODE='mi300-v1'
    result=dict(event='kernel_correctness',passed=True,precision='bf16',charts=args.charts,rank=rank,features=n,
                tile_policy=args.tile_policy,timing_baseline='current tuned v1' if args.tile_policy=='mi300-v2' else 'legacy',
                geometry_reference=geometry_reference,
                validation_uses_original_kernels=True,weight_packing_bitwise_equal=True,
                weight_refresh_ms={k:statistics.median(v) for k,v in refresh_samples.items()},checks=checks,
                scope='Complete expert forward/backward, excluding router and optimizer; no whole-update speed claim',
                full_convergence_verified=False,utc=datetime.now(timezone.utc).isoformat())
    with args.out.open('a') as out:out.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
