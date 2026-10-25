"""Research-only maximum-reduction diagnostic; never a training reference."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import torch
import triton as tr
from wr_reproduction import fused_fp8
from wr_reproduction.check_shared_fp8_tiles import operands,oracle,check,F4,F5


def execute(a,b,sa,sb,mode,aux,amax,policy,stores=True):
    fused_fp8.AMAX_MODE=policy
    values=fused_fp8.run(a,b,sa,sb,mode=mode,post_scale=sa,dpre_scale=sb,
        amax=amax,aux=aux,emit_t=True,store_pre=mode==1 and stores,
        store_post=mode==1 and stores)
    return (*values,amax)


def exact(reference,actual):
    for left,right in zip(reference,actual):
        if left is None:
            assert right is None
        elif left.dtype in (F4,F5):
            assert torch.equal(left.view(torch.uint8),right.view(torch.uint8)),'FP8 output bytes changed'
        else:
            torch.testing.assert_close(left,right,rtol=0,atol=0,equal_nan=True)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tile-policy',choices=['mi300-v1','mi300-v2'],default='mi300-v2')
    args=parser.parse_args();torch.manual_seed(169);torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    assert fused_fp8.TILE_MODE=='legacy' and fused_fp8.AMAX_MODE=='atomic'
    fused_fp8.TILE_MODE=args.tile_policy
    def emit(**row):
        row['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    sa=torch.tensor(.1173,device='cuda');sb=torch.tensor(.0831,device='cuda')
    checks=[];results=[]
    try:
        # Tails, both layouts and four source-format combinations. In addition
        # to an independent FP32 oracle, require exact old-vs-new outputs.
        for mode in (1,2):
            for ad in (F4,F5):
                for bd in (F4,F5):
                    for layout in ('rc','cr'):
                        a,b=operands(33,65,131,ad,bd,layout)
                        aux=torch.randn(33,65,device='cuda').abs().to(F4) if mode==2 else None
                        baseline=execute(a,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'atomic')
                        candidate=execute(a,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'partial')
                        exact(baseline,candidate)
                        errors=check(candidate,oracle(a,b,sa,sb,F4 if mode==1 else F5,mode,aux),mode)
                        checks.append(dict(mode=mode,ad=str(ad),bd=str(bd),layout=layout,errors=errors))
                        emit(event='shared_fp8_amax_format',passed=True,**checks[-1])
        emit(event='shared_fp8_amax_oracle',passed=True,cases=len(checks),checks=checks)

        # A delayed-scale accumulator is shared by successive logical ranks.
        # Never reset its earlier maximum or change nonfinite-value behavior.
        for mode in (1,2):
            a,b=operands(1025,2816,768,F4 if mode==1 else F5,F4)
            aux=torch.rand(1025,2816,device='cuda').to(F4) if mode==2 else None
            for initial in (0.,1.,1.e10,float('inf'),float('nan')):
                old=torch.tensor(initial,device='cuda');new=old.clone()
                for multiplier in (1.,.25,2.,.5):
                    aa=(a.float()*multiplier).to(a.dtype)
                    exact(execute(aa,b,sa,sb,mode,aux,old,'atomic'),
                          execute(aa,b,sa,sb,mode,aux,new,'partial'))
            for bad in (float('nan'),float('inf'),float('-inf')):
                for all_rows in (False,True):
                    aa=a.float().clone()
                    if all_rows:aa.fill_(bad)
                    else:aa[0,0]=bad
                    aa=aa.to(a.dtype)
                    exact(execute(aa,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'atomic'),
                          execute(aa,b,sa,sb,mode,aux,torch.zeros((),device='cuda'),'partial'))
        emit(event='shared_fp8_amax_accumulation',passed=True,prior_maxima_preserved=True,
             successive_calls=4,nonfinite_inputs_checked=True,nonfinite_initial_amax_checked=True)

        # Real producer calls store the FP8 row/column layouts and amax only.
        # Include reduction launch and scratch allocation in captured timings.
        for m in (16384,32768,40960,49152):
            for mode in (1,2):
                n,k=2816,768
                a,b=operands(m,n,k,F4 if mode==1 else F5,F4)
                aux=torch.rand(m,n,device='cuda').to(F4) if mode==2 else None
                old=torch.zeros((),device='cuda');new=old.clone()
                baseline=lambda:execute(a,b,sa,sb,mode,aux,old,'atomic',stores=False)
                candidate=lambda:execute(a,b,sa,sb,mode,aux,new,'partial',stores=False)
                exact(baseline(),candidate())
                samples=[];base_samples=[]
                for order in ((baseline,candidate),(candidate,baseline),(baseline,candidate)):
                    for fn in order:
                        value=tr.testing.do_bench_cudagraph(fn,rep=50)
                        (samples if fn is candidate else base_samples).append(value)
                base=statistics.median(base_samples);value=statistics.median(samples)
                result=dict(m=m,n=n,k=k,mode=mode,baseline_ms=base,candidate_ms=value,
                    speedup=base/value,baseline_samples_ms=base_samples,candidate_samples_ms=samples,
                    outputs_bitwise_equal=True)
                results.append(result);emit(event='shared_fp8_amax_timing',**result)
                del a,b,aux,old,new
        fused_fp8.AMAX_MODE='partial'
        from wr_reproduction.check_mlp import main as check_mlp
        with torch.enable_grad():
            check_mlp(version='pr393',tokens=1024)
        artifact=args.out.parent/'shared-fp8-amax.json'
        artifact.write_text(json.dumps(dict(tile_policy=args.tile_policy,format_checks=checks,
            timing=results,all_outputs_bitwise_equal=True),indent=2,allow_nan=False)+'\n')
        from wr_reproduction.research import sha
        emit(event='kernel_correctness',passed=True,scope='Partial maximum reduction with unchanged FP8 producers',
             tile_policy=args.tile_policy,all_outputs_bitwise_equal=True,delayed_maximum_semantics_checked=True,
             source_format_pairs=4,original_dense_mlp_autograd_checked=True,
             full_convergence_verified=False,whole_update_speed_measured=False,
             artifact=str(artifact),artifact_sha256=sha(artifact),timing=results)
    finally:
        fused_fp8.TILE_MODE='legacy';fused_fp8.AMAX_MODE='atomic'


if __name__=='__main__':main()
