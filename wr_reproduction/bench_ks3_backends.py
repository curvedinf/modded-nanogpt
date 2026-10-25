"""Paired training-call microbenchmark: exact upstream KS3 INT8 vs run169 BLAS/BF16.

One expert block, no routing, attention, optimizer, or whole-model inference.
Both paths own the same FP32 parameter tensors and use BF16 inputs/FP16
nonlinearities. Compilation and warmup are excluded; parameter packing is timed.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from wr_reproduction.ks3_kernel import ks3_int8,SOURCE_URL,SOURCE_SHA256
from ks3_research.backward import ks3_blas_fused_unpack

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=25);parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--include-fp8',action='store_true')
    args=parser.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.manual_seed(393169)
    device='cuda';width,charts=768,1024
    down=(torch.randn(charts,width,device=device)*.02).requires_grad_()
    up=(torch.randn(width,charts,device=device)*.02).requires_grad_()
    coeff=(torch.randn(charts,1,6,device=device)*.01).requires_grad_()
    scale=torch.full((charts,),.75,device=device,requires_grad=True)
    shift=torch.zeros(charts,device=device,requires_grad=True)
    lut=(torch.tanh(torch.arange(-128,128,device=device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,hip=torch.version.hip,source=SOURCE_URL,source_sha256=SOURCE_SHA256,
        width=width,charts=charts,rank=1,master_dtype='float32',input_dtype='bfloat16',nonlinear='fp16',
        benchmark='One KS3 expert training forward and backward including per-call packing; no router/optimizer/attention.',
        inputs='Identical synthetic inputs and nonzero random projection/spine weights for all paths; not trained checkpoints.',
        forward_precision='Upstream: packed INT8 dots with INT32 accumulation. Alternatives: BLAS BF16 or reused native FP8 operands, FP32 outputs.',
        backward_precision='Upstream: packed operands dequantized inside FP32 backward GEMMs. Alternatives: BF16 or FP8 products with FP32 GEMM outputs.',
        timing='Alternating paired order after warmup; CUDA event times and synchronized host end-to-end wall time; compilation excluded.',
        repeats=args.repeats,warmup=args.warmup,cases=[])
    report['implementation_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path('ks3_research/backward.py'),Path('ks3_research/fp8_blas.py'),Path('ks3_research/unpack.py')] if p.exists()}
    for tokens in [2048,4096,12288]:
        x=torch.randn(tokens,width,device=device,dtype=torch.bfloat16,requires_grad=True)
        grad=torch.randn_like(x)*.01;params=(x,down,up,coeff,scale,shift)
        def upstream():return ks3_int8(*params,subspace_rank=1,nonlinear='fp16',tanh_lut=lut)
        def blas():return ks3_blas_fused_unpack(*params,subspace_rank=1,nonlinear='fp16',tanh_lut=lut,
                                              dynamic_rows=True,bf16_products=True)
        functions={'upstream_int8':upstream,'run169_blas_bf16':blas};samples={k:[] for k in functions}
        if args.include_fp8:
            functions['blas_fp8']=lambda:ks3_blas_fused_unpack(*params,subspace_rank=1,nonlinear='fp16',tanh_lut=lut,
                                                               dynamic_rows=True,fp8_products=True)
            samples['blas_fp8']=[]
        print(json.dumps(dict(event='warming',tokens=tokens)),flush=True)
        for i in range(args.warmup):
            for f in functions.values():
                y=f();g=torch.autograd.grad(y,params,grad)
                assert all(t.isfinite().all().item() for t in (y,*g))
                del y,g
        torch.cuda.synchronize()
        # Report numerical differences independently of timing; do not call approximate paths identical.
        outputs=[];grads=[]
        for f in functions.values():
            y=f();g=torch.autograd.grad(y,params,grad);outputs.append(y.detach());grads.append(g)
        def rms(a,b):return ((a.float()-b.float()).square().mean().sqrt()/a.float().square().mean().sqrt().clamp_min(1e-12)).item()
        differences={name:dict(output_relative_rms=rms(outputs[0],outputs[i]),
            gradient_relative_rms={label:rms(a,b) for label,a,b in zip(['input','down','up','spine','scale','shift'],grads[0],grads[i])})
            for i,name in enumerate(functions) if i}
        del outputs,grads,y,g
        for i in range(args.repeats):
            names=list(functions)
            if i%2:names.reverse()
            for name in names:
                torch.cuda.synchronize()
                start,mid,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                begun=time.perf_counter();start.record()
                y=functions[name]();mid.record()
                g=torch.autograd.grad(y,params,grad);end.record();end.synchronize()
                wall=(time.perf_counter()-begun)*1000
                samples[name].append(dict(forward_ms=start.elapsed_time(mid),backward_ms=mid.elapsed_time(end),
                                          total_gpu_ms=start.elapsed_time(end),total_wall_ms=wall))
                del y,g
        medians={name:{key:statistics.median(s[key] for s in data) for key in data[0]} for name,data in samples.items()}
        case=dict(tokens=tokens,medians=medians,samples=samples,numerical_differences=differences,
                  int8_over_blas_time={k:medians['upstream_int8'][k]/medians['run169_blas_bf16'][k] for k in medians['upstream_int8']})
        report['cases'].append(case);args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in case.items() if k!='samples'}),flush=True)
    report['finished_utc']=datetime.now(timezone.utc).isoformat();args.out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
