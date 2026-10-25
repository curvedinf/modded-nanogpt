"""Device check of the exact user-selected kernel, without replacement GEMMs."""
import json
import torch
from wr_reproduction.ks3_kernel import ks3_int8,SOURCE_URL,SOURCE_SHA256

def main():
    torch.manual_seed(393);device='cuda';d,c=768,1024
    down=(torch.randn(c,d,device=device)*.02).requires_grad_()
    up=(torch.randn(d,c,device=device)*.02).requires_grad_()
    coeff=torch.zeros(c,1,6,device=device,requires_grad=True)
    scale=torch.full((c,),.75,device=device,requires_grad=True)
    shift=torch.zeros(c,device=device,requires_grad=True)
    checks=[]
    for n in [0,1,257,4096]:
        x=torch.randn(n,d,device=device,dtype=torch.bfloat16,requires_grad=True)
        y=ks3_int8(x,down,up,coeff,scale,shift,subspace_rank=1,nonlinear='fp16')
        grads=torch.autograd.grad(y,(x,down,up,coeff,scale,shift),torch.ones_like(y))
        assert y.shape==x.shape and y.dtype==x.dtype
        assert all(g.isfinite().all().item() for g in (y,*grads))
        assert all(g.dtype==torch.float32 for g in grads[1:])
        checks.append(dict(tokens=n,finite_output_and_all_gradients=True,output_dtype=str(y.dtype)))
    print(json.dumps(dict(passed=True,source=SOURCE_URL,sha256=SOURCE_SHA256,charts=c,width=d,
                         modified=False,checks=checks),indent=2))

if __name__=='__main__':main()
