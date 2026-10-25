"""BF16 hardware tile sweep against independent FP32 product references.

This is a kernel diagnostic, not a training trajectory or convergence result.
"""
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


CONFIGS=[('baseline',32,64,32,4,None,None)]
for bm,bn,bk,warps,mfma in ((64,128,64,4,16),(64,128,64,4,32),
        (64,128,128,4,16),(64,128,128,8,32),(128,128,64,8,32),
        (128,128,128,8,32),(64,64,64,4,16),(32,128,64,4,16)):
    CONFIGS.append((f'{bm}-{bn}-{bk}-w{warps}-m{mfma}',bm,bn,bk,warps,1,mfma))


def options(config):
    name,bm,bn,bk,warps,stages,mfma=config
    result=dict(BM=bm,BN=bn,BK=bk,num_warps=warps)
    if stages is not None:result['num_stages']=stages
    if mfma is not None:result['matrix_instr_nonkdim']=mfma
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--charts',type=int,choices=[512,768,1024,1536],default=1024)
    parser.add_argument('--geometry-backward-rows',type=int,choices=[64],default=64)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(169)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';e,n,d=8,args.charts,768;cases=[]
    def emit(row):
        row=dict(utc=datetime.now(timezone.utc).isoformat(),**row)
        with args.out.open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    for tokens in (16384,49152):
        expert=torch.randint(e,(tokens,),device=device)
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
        m=src.numel();boundaries=[0]+offs.cpu().tolist();valid=src<tokens
        roles=[('input_projection',768,n,False,torch.float32),
               ('output_projection',n,768,False,torch.bfloat16),
               ('feature_gradient',768,n,True,torch.float32),
               ('input_gradient',n,768,True,torch.bfloat16),
               ('weight_gradient',n,768,None,torch.bfloat16)]
        for role,k,outdim,row_major,dtype in roles:
            a=torch.randn(m,k,device=device,dtype=torch.bfloat16)*.1
            a[~valid]=0
            if row_major is None:
                b=torch.randn(m,outdim,device=device,dtype=torch.bfloat16)*.1;b[~valid]=0
                out=torch.empty(e,k,outdim,device=device,dtype=dtype)
                reference=torch.stack([(a[lo:hi].float().T@b[lo:hi].float()).to(dtype)
                    for lo,hi in zip(boundaries,boundaries[1:])])
            else:
                w=torch.randn(e,k,outdim,device=device,dtype=torch.bfloat16)*.1
                if not row_major:w=w.transpose(1,2).contiguous().transpose(1,2)
                out=torch.empty(m,outdim,device=device,dtype=dtype)
                reference=torch.cat([(a[lo:hi].float()@w[index].float()).to(dtype)
                    for index,(lo,hi) in enumerate(zip(boundaries,boundaries[1:]))])
            def invoke(config):
                _,bm,bn,*_=config
                if row_major is None:
                    _dw[(tr.cdiv(k,bm),tr.cdiv(outdim,bn),e)](a,b,out,offs,k,outdim,False,**options(config))
                else:
                    _mm[(tr.cdiv(m,bm),tr.cdiv(outdim,bn))](a,w,out,tiles,m,outdim,k,*w.stride(),**options(config))
            data={}
            for config in CONFIGS:
                invoke(config)
                # moe_dispatch reserves spare row capacity beyond offs[-1].
                # Only the expert ranges are consumed by gather and dW.
                actual=out if row_major is None else out[:boundaries[-1]]
                error=((actual.float()-reference.float()).square().mean().sqrt()/
                       reference.float().square().mean().sqrt().clamp_min(1e-12)).item()
                assert actual.isfinite().all(),(role,config)
                assert error<.0005,(tokens,role,config,error)
                data[config[0]]=dict(relative_rms=error,samples_ms=[],config=list(config[1:]))
            for order in (CONFIGS,CONFIGS[::-1],CONFIGS[3:]+CONFIGS[:3]):
                for config in order:
                    data[config[0]]['samples_ms'].append(tr.testing.do_bench_cudagraph(lambda:invoke(config),rep=50))
            for result in data.values():result['median_ms']=statistics.median(result['samples_ms'])
            winner=min(data,key=lambda name:data[name]['median_ms'])
            case=dict(tokens=tokens,role=role,candidates=data,winner=winner)
            cases.append(case)
            emit(dict(event='kernel_sweep_progress',tokens=tokens,role=role,winner=winner,
                      baseline_ms=data['baseline']['median_ms'],winner_ms=data[winner]['median_ms']))
    emit(dict(event='kernel_correctness',passed=True,precision='bf16',charts=n,cases=cases,
              reference='Independent FP32 products of the same BF16 operands; final output cast matches each role',
              scope='Matrix kernels only, including no geometry, routing, optimizer or full updates',
              full_convergence_verified=False))


if __name__=='__main__':main()
