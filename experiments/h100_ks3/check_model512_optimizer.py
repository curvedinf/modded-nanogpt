"""Compare every 512-wide PR393 optimizer bank with the original cascade."""
import datetime
import json
import statistics
import sys
from pathlib import Path
import torch
import triton.testing

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3 import moe_optimizer as candidate


def shared(grad,velocity,momentum,split,fast_beta,fast_weight,margin):
    assert margin==1.05
    x=torch.empty_like(grad)
    candidate.fused.pre_fused(grad,velocity,x,momentum,fast_beta,fast_weight)
    return candidate.fused.polar(x,split,candidate.anvil.ANVIL_MAPS)


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(493)
    candidate.install()
    rows=[]
    for name,shape,margin,function in [
        ('qk',(4,256,512),1.05,shared),
        ('vo',(2,512,512),1.05,shared),
        ('dense',(2,2816,512),1.05,shared),
        ('ks3',(10,1536,512),1.1,candidate.cascade),
    ]:
        grad=torch.randn(shape,device='cuda',dtype=torch.bfloat16)*.01
        velocity=torch.randn(2,*shape,device='cuda')*.001
        args=[torch.tensor(v,device='cuda') for v in (.9,.85,.4385)]
        split=shape[1]>1024
        rv,nv=velocity.clone(),velocity.clone()
        ref=candidate.ORIGINAL(grad,rv,args[0],split,args[1],args[2],margin)
        out=function(grad,nv,args[0],split,args[1],args[2],margin)
        error=((out.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt().item()
        row=dict(bank=name,shape=shape,relative_rms=error,output_bitwise_equal=torch.equal(out,ref),
                 velocity_bitwise_equal=torch.equal(rv,nv))
        assert torch.isfinite(out).all() and error<.002,row
        assert row['velocity_bitwise_equal'],row
        for label,fn in [('original',candidate.ORIGINAL),('fused',function)]:
            v=velocity.clone()
            samples=[triton.testing.do_bench_cudagraph(
                lambda:fn(grad,v,args[0],split,args[1],args[2],margin),rep=40) for _ in range(3)]
            row[label]=dict(samples_ms=samples,median_ms=statistics.median(samples))
        rows.append(row)
        print(json.dumps(row),flush=True)
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=rows,passed=True)
    Path('research_results/wr_reproduction/h100-research/model512-optimizer-check.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
