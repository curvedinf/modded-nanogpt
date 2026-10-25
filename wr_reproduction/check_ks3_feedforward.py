"""Independent dense KS3 output/input/weight/geometry/post-gradient oracles.

Run after the matched controls under the study's single-GPU lease. This is
correctness evidence, never a convergence result or a shared hardware sweep.
"""
import argparse
import json
import torch
from wr_reproduction import ks3_moe as core
from wr_reproduction.ks3_feedforward import DenseKS3
from ks3_research.backward import ks3_blas_fused_unpack

def error(a,b):
    assert torch.isfinite(a).all()
    return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--features',type=int,choices=(512,768,1024,1280,1536,2816),default=1024);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(172);core.BF16_TUNED_KERNELS=True
    checks=[]
    for tokens in (1,257,4096):
        d,n=768,a.features
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16).requires_grad_()
        u=(torch.randn(n,d,device='cuda')*.025).bfloat16().requires_grad_()
        w=(torch.randn(n,d,device='cuda')*.02).bfloat16().requires_grad_()
        geo=(torch.randn(n,8,device='cuda')*.01);geo[:,6]=.75;geo.requires_grad_()
        post=torch.tensor(.7,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        up=core.pack(u.detach());down=core.pack(w.detach().T)
        y=DenseKS3.apply(x,u,w,up,down,geo,0)*post
        grad=torch.randn_like(y)*.02
        actual=torch.autograd.grad(y,(x,u,w,geo,post),grad)
        xr=x.detach().requires_grad_();ur=u.detach().float().requires_grad_();dr=w.detach().T.float().contiguous().requires_grad_()
        coeff=geo.detach()[:,:6].contiguous().requires_grad_()
        scale=geo.detach()[:,6].contiguous().requires_grad_();shift=geo.detach()[:,7].contiguous().requires_grad_()
        pr=post.detach().requires_grad_()
        yr=ks3_blas_fused_unpack(xr,ur,dr,coeff,scale,shift,subspace_rank=1,nonlinear='fp16',dynamic_rows=True,bf16_products=True)*pr
        ref=torch.autograd.grad(yr,(xr,ur,dr,coeff,scale,shift,pr),grad)
        gr=torch.cat([ref[3],ref[4][:,None],ref[5][:,None]],1)
        values=dict(output=error(y,yr),input_gradient=error(actual[0],ref[0]),
                    up_gradient=error(actual[1],ref[1]),down_gradient=error(actual[2],ref[2].T),
                    geometry_gradient=error(actual[3],gr),post_gradient=error(actual[4],ref[6]))
        assert max(values.values())<.015,(tokens,values)
        checks.append(dict(tokens=tokens,relative_rms=values))
    result=dict(event='kernel_correctness',passed=True,checks=checks,features=a.features,
                oracle='Independent ks3_blas_fused_unpack autograd; all gradients, nonzero geometry and output post factor',
                shared_backend_modified=False,validation=False)
    open(a.out,'w').write(json.dumps(result)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
