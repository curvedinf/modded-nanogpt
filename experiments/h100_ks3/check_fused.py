"""Paired complete routed-expert forward/backward ablations, including router."""
import sys,json,statistics,datetime,os
from pathlib import Path
import torch,triton as tr,triton.testing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.port import install
install()
from wr_reproduction import ks3_moe as core,ks3_bf16_gemm as matrix
from experiments.h100_ks3 import optimized as opt

def err(a,b):return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()

def main():
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(417)
    original=(matrix.run,matrix.dw,core.RoutedKS3,core.backward)
    core.BF16_TUNED_KERNELS=True
    width=int(os.environ.get('KS3_H100_WIDTH','1024'))
    cases=[]
    for t in [int(v) for v in os.environ.get('KS3_MICRO_TOKENS','1,257,16384,32768,40960,49152').split(',')]:
        rows=torch.randn(t,768,device='cuda',dtype=torch.bfloat16).requires_grad_()
        router=(torch.randn(8,768,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
        post=torch.tensor(.8,device='cuda',requires_grad=True)
        w1=torch.randn(8,width,768,device='cuda',dtype=torch.bfloat16)*.02
        w2=torch.randn_like(w1)*.02
        up=core.pack(w1.reshape(-1,768)).reshape_as(w1);down=torch.stack([core.pack(w.T) for w in w2])
        g=(torch.randn(8,width,8,device='cuda')*.01);g[...,6]=.75;g.requires_grad_()
        bank=torch.empty(16,width,768,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(rows)*.02;gp=torch.randn(t,8,device='cuda')*.002
        def run():
            y,p,counts=core.RoutedKS3.apply(rows,router,post,w1,w2,up,down,g,bank,0,True)
            grads=torch.autograd.grad((y,p),(rows,router,post,g),(grad,gp))
            return (y.detach(),*grads,bank)
        def setup(v):
            matrix.run,matrix.dw,core.RoutedKS3,core.backward=original
            if v!='original':opt.install(v)
        setup('original');ref=[v.detach().clone() for v in run()]
        variants={}
        for v in os.environ.get('KS3_MICRO_VARIANTS','original,tiles,gather,fused,fused-input,fused-dx,fused-input-dx,fused-vjp-dx').split(','):
            setup(v)
            try:
                actual=run();errors={n:err(a,b) for n,a,b in zip(['output','dx','router','post','geometry','weights'],actual,ref)}
                assert all(torch.isfinite(x).all() for x in actual)
                post_audit={'absolute_error':float((actual[3]-ref[3]).abs().max()),'reference':float(ref[3].flatten()[0])}
                # A single signed scalar reduction is more cancellation-sensitive
                # than the tensor RMS checks. Keep its separate audit visible.
                assert max(v for k,v in errors.items() if k!='post')<.003 and errors['post']<.01,(errors,post_audit)
                samples=[]
                if t>=16384:
                    for _ in range(3):samples.append(tr.testing.do_bench_cudagraph(run,rep=40))
                variants[v]={'passed':True,'relative_rms':errors,'post_audit':post_audit,'samples_ms':samples,'median_ms':statistics.median(samples) if samples else None}
            except Exception as ex:variants[v]={'passed':False,'exception':str(ex)[:1200]}
        cases.append({'tokens':t,'variants':variants})
        (Path('research_results/wr_reproduction/h100-ks3/micro')/os.environ.get('KS3_MICRO_OUTPUT','routed.json')).write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':cases},indent=2)+'\n')
        print(t,json.dumps({k:{x:y for x,y in v.items() if x!='relative_rms'} for k,v in variants.items()}),flush=True)
if __name__=='__main__':main()
