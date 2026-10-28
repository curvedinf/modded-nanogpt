"""Paired launch screen; unchanged upstream math, including accumulation."""
import datetime,json,os,statistics,sys
from pathlib import Path
import torch,triton.testing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.tma_dw import run
from track_1_short.perf.kernels.moe_mlp import _grouped_weight_grad_impl


def main():
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(431)
    h=int(os.environ.get('KS3_H100_WIDTH','1280'));d=512
    cases=[]
    for m in (16384,49152):
        a=(torch.randn(h,m,device='cuda')*.1).to(torch.float8_e4m3fn)
        b=(torch.randn(d,m,device='cuda')*.1).to(torch.float8_e5m2)
        # Include unequal routing and an empty expert, aligned as actual dispatch.
        offsets=torch.tensor([0,1,3,5,7,10,13,16],device='cuda',dtype=torch.int32)*(m//16)
        scale=torch.tensor([.1],device='cuda');flag=torch.zeros(1,device='cuda',dtype=torch.int32)
        out=torch.empty(8,h,d,device='cuda',dtype=torch.bfloat16)
        def original():_grouped_weight_grad_impl(a,b,offsets,scale,scale,8,out.view(8*h,d),flag)
        original();expected=out.clone()
        candidates=[('upstream',None)]+[(t,s) for t in ('64x128','64x256','128x128','128x256','256x128') for s in (3,4)]
        for tile,stages in candidates:
            fn=original if tile=='upstream' else lambda:run(a,b,out,offsets,scale,scale,flag,tile,stages)
            try:
                flag.zero_();fn()
                relative=((out.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                assert relative<.004,(tile,relative)
                old=out.clone();flag.fill_(1);fn()
                oracle=(expected.float()+old.float()).bfloat16()
                accumulated=((out.float()-oracle.float()).square().mean()/oracle.float().square().mean()).sqrt().item()
                assert accumulated<.004,(tile,accumulated)
                flag.zero_()
                samples=[triton.testing.do_bench_cudagraph(fn,rep=60) for _ in range(3)]
                row=dict(tokens=m,tile=tile,stages=stages,relative_rms=relative,accumulation_relative_rms=accumulated,ms=statistics.median(samples),samples_ms=samples,passed=True)
            except Exception as exc:
                row=dict(tokens=m,tile=tile,stages=stages,passed=False,error=repr(exc))
            cases.append(row);print(json.dumps(row),flush=True)
    Path(os.environ['KS3_DW_CHECK_OUTPUT']).write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),width=h,model_dim=d,cases=cases),indent=2)+'\n')


if __name__=='__main__':main()
