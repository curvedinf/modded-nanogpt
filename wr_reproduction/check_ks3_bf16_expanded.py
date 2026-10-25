"""Paired BF16 expert tile sweep versus the current tuned public dispatch."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import sys
import torch
import triton as tr
from wr_reproduction.contracts import ROOT
sys.path.insert(0,str(ROOT/'pr393/upstream'))
from track_1_short.perf.kernels import moe_mlp as moe
from wr_reproduction.ks3_moe import _mm,_dw
from wr_reproduction import ks3_bf16_gemm as tuned


# BM never exceeds128: an expert boundary can occur every128 routed rows.
# Wider output/reduction tiles and additional pipeline stages were not in
# the original sweep. No input formats or mathematical expressions change.
CONFIGS=[(64,256,64,8,1,32),(128,256,64,8,1,32),
         (64,128,256,4,1,16),(64,128,128,4,1,32),
         (128,128,64,8,2,32),(128,128,128,8,2,32),
         (64,128,128,4,2,16),(128,64,128,4,1,16)]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--charts',type=int,choices=[1024],default=1024)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(169)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';experts=8;n=args.charts;cases=[]
    def emit(**row):
        row['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def error(actual,reference):
        assert actual.isfinite().all()
        value=((actual.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12)).item()
        assert value<.0005,value
        return value
    for tokens in (16384,49152):
        expert=torch.randint(experts,(tokens,),device=device)
        _,src,offs,tiles,_=moe.moe_dispatch(expert,experts)
        m=src.numel();boundaries=[0]+offs.cpu().tolist();valid=src<tokens
        roles=[('input_projection',768,n,False,torch.float32),
               ('output_projection',n,768,False,torch.bfloat16),
               ('feature_gradient',768,n,True,torch.float32),
               ('input_gradient',n,768,True,torch.bfloat16),
               ('weight_gradient',n,768,None,torch.bfloat16)]
        for role,k,outdim,row_major,dtype in roles:
            a=torch.randn(m,k,device=device,dtype=torch.bfloat16)*.1;a[~valid]=0
            if row_major is None:
                b=torch.randn(m,outdim,device=device,dtype=torch.bfloat16)*.1;b[~valid]=0
                output=torch.empty(experts,k,outdim,device=device,dtype=dtype)
                reference=torch.stack([(a[lo:hi].float().T@b[lo:hi].float()).to(dtype)
                    for lo,hi in zip(boundaries,boundaries[1:])])
                def baseline():
                    tuned.dw(a,b,output,offs)
                    return output
            else:
                w=torch.randn(experts,k,outdim,device=device,dtype=torch.bfloat16)*.1
                if not row_major:w=w.transpose(1,2).contiguous().transpose(1,2)
                reference=torch.cat([(a[lo:hi].float()@w[index].float()).to(dtype)
                    for index,(lo,hi) in enumerate(zip(boundaries,boundaries[1:]))])
                def baseline():return tuned.run(a,w,tiles,dtype)[:boundaries[-1]]
            baseline_error=error(baseline(),reference);variants=[]
            for config in CONFIGS:
                bm,bn,bk,warps,stages,mfma=config
                options=dict(BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
                def candidate():
                    if row_major is None:
                        _dw[(tr.cdiv(k,bm),tr.cdiv(outdim,bn),experts)](a,b,output,offs,k,outdim,False,**options)
                        return output
                    out=torch.empty(m,outdim,device=device,dtype=dtype)
                    _mm[(tr.cdiv(m,bm),tr.cdiv(outdim,bn))](a,w,out,tiles,m,outdim,k,*w.stride(),**options)
                    return out[:boundaries[-1]]
                try:
                    numerical_error=error(candidate(),reference)
                    old=[];new=[]
                    for order in ((baseline,candidate),(candidate,baseline),(baseline,candidate)):
                        for fn in order:
                            value=tr.testing.do_bench_cudagraph(fn,rep=30)
                            (old if fn is baseline else new).append(value)
                    base_ms=statistics.median(old);new_ms=statistics.median(new)
                    variant=dict(config=config,relative_rms=numerical_error,paired_baseline_ms=base_ms,
                        median_ms=new_ms,paired_speedup=base_ms/new_ms,
                        baseline_samples_ms=old,candidate_samples_ms=new)
                except Exception as exc:
                    variant=dict(config=config,error=str(exc)[:800])
                variants.append(variant)
                emit(event='bf16_expanded_variant',tokens=tokens,role=role,**variant)
            successful=[v for v in variants if 'median_ms' in v]
            if not successful:raise RuntimeError('No correct candidate for '+role)
            best=max(successful,key=lambda x:x['paired_speedup'])
            case=dict(tokens=tokens,routed_capacity=m,used_rows=boundaries[-1],charts=n,role=role,
                k=k,n=outdim,row_major_weight=row_major,output_dtype=str(dtype),baseline_relative_rms=baseline_error,
                baseline='current tuned BF16 public dispatch',best=best,variants=variants)
            cases.append(case)
            emit(event='bf16_expanded_case',**{k:v for k,v in case.items() if k!='variants'})
            del a,reference
            if row_major is None:del b,output
            else:del w
    path=args.out.parent/'ks3-bf16-expanded-tiles.json'
    path.write_text(json.dumps(dict(configurations=CONFIGS,cases=cases),indent=2,allow_nan=False)+'\n')
    from wr_reproduction.research import sha
    emit(event='kernel_correctness',passed=True,precision='bf16',charts=n,case_count=len(cases),
        artifact=str(path),artifact_sha256=sha(path),baseline='current tuned BF16 public dispatch',
        paired_timing=True,reference='Independent FP32 products of the same BF16 operands with original output casts',
        scope='Expert matrix kernels only; excludes geometry/routing/optimizer and full updates',
        full_convergence_verified=False,whole_model_speed_measured=False,
        fastest=[dict(tokens=c['tokens'],role=c['role'],**c['best']) for c in cases])


if __name__=='__main__':main()
