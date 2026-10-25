"""Registered MI300 shared-FP8 tile sweep, preserving record bytes/epilogues."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import torch
import triton as tr
from wr_reproduction.fused_fp8 import _gemm,run

CONFIGS=[(64,128,64,4,1,16),(64,128,128,4,1,16),
         (128,128,64,8,1,32),(128,128,128,8,1,32),
         (64,256,128,8,1,32),(128,256,128,8,1,32),
         (64,64,128,4,1,32),(64,128,128,4,1,32)]
EXPANDED_CONFIGS=[(256,128,128,8,1,32),(256,256,64,8,1,32),
                  (256,256,128,8,1,32),(128,512,64,8,1,32),
                  (128,256,128,4,1,32),(128,256,128,8,2,32),
                  (128,128,128,8,2,32),(32,64,256,4,1,32),
                  (64,64,256,4,1,32),(64,128,256,4,1,32)]
F4=torch.float8_e4m3fn;F5=torch.float8_e5m2


def execute(a,b,sa,sb,config,out_dtype,mode=0,aux=None):
    m,k=a.shape;n=b.shape[1]
    ps=sa;ds=sb;amax=torch.zeros((),device=a.device)
    if config is None:
        values=run(a,b,sa,sb,out_dtype=out_dtype,mode=mode,post_scale=ps,dpre_scale=ds,
                   amax=amax,aux=aux,emit_t=bool(mode),store_pre=mode==1,store_post=mode==1)
        return (*values,amax)
    bm,bn,bk,warps,stages,mfma=config
    dtype=out_dtype if mode==0 else (F4 if mode==1 else F5)
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    trans=torch.empty((n,m),device=a.device,dtype=dtype) if mode else out
    pre=torch.empty((m,n),device=a.device,dtype=torch.bfloat16) if mode==1 else out
    post=torch.empty_like(pre) if mode==1 else out
    _gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a.view(torch.uint8),b.view(torch.uint8),out,trans,pre,post,
        aux if aux is not None else sa,sa,sb,ps,ds,amax,sa,m,n,k,*a.stride(),*b.stride(),
        a.dtype==F5,b.dtype==F5,mode,False,bool(mode),mode==1,mode==1,
        BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,
        enable_fp_fusion=False,matrix_instr_nonkdim=mfma)
    return out,trans if mode else None,pre if mode==1 else None,post if mode==1 else None,amax


def oracle(a,b,sa,sb,out_dtype,mode,aux):
    acc=(a.float()@b.float())*(sa*sb)
    if mode==1:
        pre=acc.bfloat16();post=pre.float().clamp_min(0).square().bfloat16()
        out=(post.float()*(1./sa)).clamp(max=448).to(F4)
        return out,pre,post,post.float().abs().amax()
    if mode==2:
        values=(2*acc*torch.sqrt(aux.float()*sa)).bfloat16()
        out=(values.float()*(1./sb)).clamp(-57344,57344).to(F5)
        return out,None,None,values.float().abs().amax()
    return acc.to(out_dtype),None,None,None


def check(values,reference,mode):
    out,trans,pre,post,amax=values;expected,epre,epost,eamax=reference
    errors={}
    for name,a,b in (('output',out,expected),('pre',pre,epre),('post',post,epost)):
        if b is None:continue
        af,bf=a.float(),b.float()
        assert af.isfinite().all() and bf.isfinite().all(),name
        rms=((af-bf).square().mean().sqrt()/bf.square().mean().sqrt().clamp_min(1e-12)).item()
        tolerance=.008 if a.dtype in (F4,F5) else (.002 if a.dtype==torch.bfloat16 else .0002)
        assert rms<tolerance,(name,rms,tolerance)
        errors[name]=rms
    if mode:
        assert torch.equal(out.view(torch.uint8).T.contiguous(),trans.view(torch.uint8)),'dual layouts differ'
        torch.testing.assert_close(amax,eamax,rtol=.008,atol=1e-6)
        errors['amax_relative']=(amax-eamax).abs().div(eamax.abs().clamp_min(1e-12)).item()
    return errors


def operands(m,n,k,ad,bd,layout='rc'):
    a=(torch.randn((m,k) if layout[0]=='r' else (k,m),device='cuda')*2).to(ad)
    b=(torch.randn((k,n) if layout[1]=='r' else (n,k),device='cuda')*2).to(bd)
    return a if layout[0]=='r' else a.T,b if layout[1]=='r' else b.T


CASES=[('attention_narrow',16384,384,768,F4,F4,torch.bfloat16,0),
       ('attention_wide',16384,2304,768,F4,F4,torch.bfloat16,0),
       ('attention_dx',16384,768,1536,F5,F4,torch.bfloat16,0),
       ('attention_dw',1536,768,16384,F5,F4,torch.bfloat16,0),
       ('dense_up_producer',16384,2816,768,F4,F4,F4,1),
       ('dense_dpre_producer',16384,2816,768,F5,F4,F5,2),
       ('dense_down',16384,768,2816,F4,F4,torch.bfloat16,0),
       ('dense_dw',2816,768,16384,F4,F5,torch.bfloat16,0),
       ('head_logits',16384,10240,768,F4,F4,F4,0),
       ('head_dx',16384,768,10240,F5,F4,torch.bfloat16,0),
       ('head_sampled_dw',10240,768,4096,F5,F4,torch.float32,0),
       ('large_dense_up',49152,2816,768,F4,F4,F4,1),
       ('large_dense_dw',2816,768,49152,F4,F5,torch.bfloat16,0),
       ('large_head_logits',49152,14336,768,F4,F4,F4,0),
       ('large_head_dx',49152,768,14336,F5,F4,torch.bfloat16,0)]
LATE_CASES=[('late_head_logits',49152,24576,768,F4,F4,F4,0),
            ('late_head_dx',49152,768,24576,F5,F4,torch.bfloat16,0),
            ('taper_head_logits',40960,50304,768,F4,F4,F4,0),
            ('taper_head_dx',40960,768,50304,F5,F4,torch.bfloat16,0),
            ('large_dense_dpre_producer',49152,2816,768,F5,F4,F5,2)]

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expanded',action='store_true',help='Broader tiles and late-stage shapes versus verified MI300-v1 dispatch')
    args=parser.parse_args();torch.manual_seed(169);torch.set_num_threads(1)
    configs=EXPANDED_CONFIGS if args.expanded else CONFIGS
    baseline_dispatch='mi300-v1' if args.expanded else 'legacy'
    from wr_reproduction import fused_fp8
    fused_fp8.TILE_MODE=baseline_dispatch
    torch.set_float32_matmul_precision('highest')
    sa=torch.tensor(.125,device='cuda');sb=torch.tensor(.125,device='cuda')
    def emit(**row):
        row['utc']=datetime.now(timezone.utc).isoformat()
        with args.out.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    eligible=[];format_checks=[]
    for config in configs:
        try:
            worst=0.
            # All four native format combinations, plus both nonstandard input
            # layouts. These also exercise row/column/reduction tail masks.
            for ad in (F4,F5):
                for bd in (F4,F5):
                    for layout in ('rc','cr'):
                        a,b=operands(33,65,131,ad,bd,layout)
                        error=check(execute(a,b,sa,sb,config,torch.float32),oracle(a,b,sa,sb,torch.float32,0,None),0)
                        worst=max(worst,error['output'])
            # Every stored byte, including negative zero and invalid encodings.
            for ad in (F4,F5):
                a=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8).view(ad).reshape(256,1)
                b=torch.ones((1,1),device='cuda',dtype=F4)
                actual=execute(a,b,sa,sb,config,torch.float32)[0].flatten()
                decoded=a.float().flatten();finite=decoded.isfinite()
                assert torch.equal(actual.isfinite(),finite)
                torch.testing.assert_close(actual[finite],decoded[finite]*(sa*sb),rtol=0,atol=0)
            eligible.append(config);format_checks.append(dict(config=config,passed=True,max_relative_rms=worst))
        except Exception as error:
            format_checks.append(dict(config=config,passed=False,error=str(error)[:800]))
        emit(event='shared_fp8_format_check',**format_checks[-1])
    if not eligible:raise RuntimeError('No tile passed four-format/layout/byte checks')
    # Shapes come from immutable PR393 dimensions and original logical batches.
    # The sampled-head dW uses original GROUP=4, hence reduction length4096.
    cases=CASES+LATE_CASES if args.expanded else CASES
    results=[]
    for name,m,n,k,ad,bd,dtype,mode in cases:
        a,b=operands(m,n,k,ad,bd)
        aux=torch.randn(m,n,device='cuda').abs().to(F4) if mode==2 else None
        reference=oracle(a,b,sa,sb,dtype,mode,aux)
        baseline=lambda:execute(a,b,sa,sb,None,dtype,mode,aux)
        baseline_errors=check(baseline(),reference,mode)
        baseline_ms=tr.testing.do_bench_cudagraph(baseline,rep=40)
        variants=[]
        for config in eligible:
            try:
                candidate=lambda:execute(a,b,sa,sb,config,dtype,mode,aux)
                errors=check(candidate(),reference,mode)
                if args.expanded:
                    samples=[];paired_baseline=[]
                    for order in ((baseline,candidate),(candidate,baseline),(baseline,candidate)):
                        for fn in order:
                            value=tr.testing.do_bench_cudagraph(fn,rep=30)
                            (samples if fn is candidate else paired_baseline).append(value)
                    median=statistics.median(samples);base_median=statistics.median(paired_baseline)
                    variants.append(dict(config=config,median_ms=median,samples_ms=samples,errors=errors,
                        paired_baseline_ms=base_median,paired_baseline_samples_ms=paired_baseline,
                        paired_speedup=base_median/median))
                else:
                    samples=[tr.testing.do_bench_cudagraph(candidate,rep=40) for _ in range(2)]
                    variants.append(dict(config=config,median_ms=statistics.median(samples),samples_ms=samples,errors=errors))
            except Exception as error:variants.append(dict(config=config,error=str(error)[:800]))
        valid=[r for r in variants if 'median_ms' in r]
        if not valid:raise RuntimeError('No valid tile for '+name)
        best=max(valid,key=lambda row:row['paired_speedup']) if args.expanded else min(valid,key=lambda row:row['median_ms'])
        result=dict(name=name,m=m,n=n,k=k,a_dtype=str(ad),b_dtype=str(bd),out_dtype=str(dtype),mode=mode,
                    layout='row-major A / column-major B',baseline_ms=baseline_ms,baseline_errors=baseline_errors,
                    best=best,variants=variants,baseline_dispatch=baseline_dispatch,
                    speedup=best['paired_speedup'] if args.expanded else baseline_ms/best['median_ms'])
        results.append(result)
        emit(event='shared_fp8_tile_case',**{k:v for k,v in result.items() if k!='variants'})
        del a,b,aux,reference
    path=args.out.parent/('shared-fp8-expanded-tile-sweep.json' if args.expanded else 'shared-fp8-tile-sweep.json')
    path.write_text(json.dumps(dict(baseline_dispatch=baseline_dispatch,paired_timing=args.expanded,
        format_checks=format_checks,cases=results),indent=2,allow_nan=False)+'\n')
    from wr_reproduction.research import sha
    emit(event='kernel_correctness',passed=True,scope='Shared native FP8 products, original fused producer epilogues and dual layouts; diagnostic tile sweep only',
         original_fn_e5_cache_encodings_preserved=True,all_stored_byte_patterns_checked=True,
         fp32_accumulation=True,full_convergence_verified=False,case_count=len(cases),
         baseline_dispatch=baseline_dispatch,paired_timing=args.expanded,
         passed_format_configurations=len(eligible),artifact=str(path),artifact_sha256=sha(path),
         fastest=[dict(name=r['name'],config=r['best']['config'],baseline_ms=r['best'].get('paired_baseline_ms',r['baseline_ms']),
                       initial_baseline_ms=r['baseline_ms'],best_ms=r['best']['median_ms'],speedup=r['speedup']) for r in results])


if __name__=='__main__':main()
