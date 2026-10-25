"""Paired native-operand tile sweep with decoded oracles; no training references."""
import argparse
from datetime import datetime,timezone
import hashlib,json,statistics
from pathlib import Path
import torch
import triton as tr
from wr_reproduction.ks3_fp8_gemm import _run_tiled,_dw_tiled
from wr_reproduction.check_ks3_prepared_native import native_copy

CONFIGS=[(64,128,128,4,1,16),(128,128,128,8,1,32),(64,128,64,4,1,16),
         (128,128,64,8,1,32),(64,128,128,4,1,32),(64,128,128,4,2,32),
         (128,128,128,8,2,32),(64,256,128,8,1,32),(128,256,64,8,1,32),
         (128,256,128,8,1,32),(64,64,256,4,1,32),(64,128,256,8,1,32),
         (128,128,64,8,2,32),(64,256,64,8,2,32)]


def rms(actual,expected):
    assert actual.isfinite().all() and expected.isfinite().all()
    return ((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12)).item()


def sweep_case(execute,baseline,reference):
    rows=[]
    for config in CONFIGS:
        try:
            actual=execute(config);error=rms(actual,reference)
            assert error<1e-4,('relative_rms',error)
            samples={'baseline':[],'candidate':[]}
            for order in (('baseline','candidate'),('candidate','baseline'),('baseline','candidate')):
                pair={}
                for name in order:
                    selected=baseline if name=='baseline' else config
                    pair[name]=tr.testing.do_bench_cudagraph(lambda:execute(selected),rep=50)
                    samples[name].append(pair[name])
            ratios=[a/b for a,b in zip(samples['baseline'],samples['candidate'])]
            rows.append(dict(config=config,relative_rms=error,samples_ms=samples,
                             paired_speedup=statistics.median(ratios),median_ms=statistics.median(samples['candidate'])))
        except Exception as error:
            rows.append(dict(config=config,error_type=type(error).__name__,error=str(error)[:500]))
    valid=[x for x in rows if 'paired_speedup' in x]
    assert valid,'Every configuration failed'
    return dict(baseline_config=baseline,variants=rows,best=max(valid,key=lambda row:row['paired_speedup']))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.manual_seed(169);torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    device='cuda';e=8;f4=torch.float8_e4m3fn;f5=torch.float8_e5m2
    scalar=torch.ones(1,device=device);scales=torch.ones(e,device=device)
    flag=torch.zeros(1,device=device,dtype=torch.int32);cases=[]
    def emit(row):
        row=dict(row,utc=datetime.now(timezone.utc).isoformat())
        with args.out.open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    for m in (16384,49152):
        group=m//e
        tiles=torch.arange(e,device=device,dtype=torch.int32).repeat_interleave(group//128)
        offs=torch.arange(1,e+1,device=device,dtype=torch.int32)*group
        for role,k,n,dtype in (('input_projection',768,1024,f4),('output_projection',1024,768,f4),
                               ('feature_gradient',768,1024,f5),('input_gradient',1024,768,f5)):
            original_a=torch.randn(m,k,device=device).to(dtype)
            original_b=torch.randn(e*n,k,device=device).to(f4).T
            a=native_copy(original_a);b=native_copy(original_b.T).T
            baseline=(128,128,128,8,1,32) if n==768 else (64,128,128,4,1,16)
            # Projection row tiles must not span the128-row expert routing units.
            assert all(config[0]<=128 and 128%config[0]==0 for config in CONFIGS)
            output_dtype=torch.bfloat16 if n==768 else torch.float32
            def execute(config):return _run_tiled(a,b,scales,scalar,tiles,n,config,out_dtype=output_dtype)
            reference=execute(baseline).clone();oracle_errors=[]
            cols=torch.arange(16,device=device)*max(n//16,1)
            for expert in range(e):
                indices=torch.tensor([0,13,63,group-1],device=device)+expert*group
                expected=(original_a[indices].float()@original_b[:,expert*n+cols].float()).to(output_dtype)
                actual=reference[indices[:,None],cols[None,:]]
                error=rms(actual,expected);assert error<(1e-3 if output_dtype==torch.bfloat16 else 2e-5),error;oracle_errors.append(error)
            row=dict(role=role,m=m,k=k,n=n,output_dtype=str(output_dtype),a_format=str(dtype),b_format=str(f4),prepared_a=True,prepared_b=True,
                     independent_decoded_oracle_max_rms=max(oracle_errors),**sweep_case(execute,baseline,reference))
            cases.append(row);emit(dict(event='native_fp8_tile_case',**{k:v for k,v in row.items() if k!='variants'}))
        for role,ad,bd in (('output_weight_gradient',f4,f5),('input_weight_gradient',f5,f4)):
            original_a=torch.randn(1024,m,device=device).to(ad)
            original_b=torch.randn(768,m,device=device).to(bd)
            a=native_copy(original_a);b=native_copy(original_b)
            out=torch.empty((e,1024,768),device=device,dtype=torch.bfloat16)
            baseline=(64,128,64,4,1,16)
            def execute(config):
                _dw_tiled(a,b,offs,scalar,scalar,e,out,flag,config)
                return out
            reference=execute(baseline).clone();oracle_errors=[]
            hidx=torch.arange(16,device=device)*64;didx=torch.arange(16,device=device)*48
            for expert in range(e):
                sl=slice(expert*group,(expert+1)*group)
                expected=(original_a[hidx,sl].float()@original_b[didx,sl].float().T).bfloat16()
                actual=reference[expert,hidx[:,None],didx[None,:]]
                error=rms(actual,expected);assert error<1e-3,error;oracle_errors.append(error)
            row=dict(role=role,m=m,k=1024,n=768,output_dtype=str(torch.bfloat16),a_format=str(ad),b_format=str(bd),prepared_a=True,prepared_b=True,
                     independent_decoded_oracle_max_rms=max(oracle_errors),**sweep_case(execute,baseline,reference))
            cases.append(row);emit(dict(event='native_fp8_tile_case',**{k:v for k,v in row.items() if k!='variants'}))
    result=dict(passed=True,seed=169,cases=cases,scope='Primitive nativeFP8 tile sweep only; no model/convergence speed claim. Failed configs retained. All candidates must pass full block and whole-model gates before training.')
    artifact=args.out.parent/'native-fp8-tiles.json';artifact.write_text(json.dumps(result,indent=2)+'\n')
    emit(dict(event='kernel_correctness',passed=True,scope=result['scope'],case_count=len(cases),artifact=str(artifact),
              artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
              cases=[{k:v for k,v in row.items() if k!='variants'} for row in cases]))

if __name__=='__main__':main()
