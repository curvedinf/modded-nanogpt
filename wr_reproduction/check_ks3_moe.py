"""Compare padded KS3 experts against the previously benchmarked BF16 block."""
import json,sys,torch
from wr_reproduction.contracts import ROOT
sys.path.insert(0,str(ROOT/'pr393/upstream'))
from track_1_short.perf.kernels import moe_mlp as moe
from wr_reproduction.ks3_moe import forward,backward,pack
from ks3_research.backward import ks3_blas_fused_unpack

def main():
    torch.manual_seed(169);torch.set_num_threads(1);dev='cuda';E,N,D=8,1024,768
    up=(torch.randn(E,N,D,device=dev)*.025).bfloat16()
    down=(torch.randn(E,N,D,device=dev)*.02).bfloat16()
    g=torch.randn(E,N,8,device=dev)*.01;g[...,6]=.75
    pu=pack(up.reshape(-1,D)).reshape_as(up)
    pd=torch.stack([pack(w.T) for w in down])
    checks=[]
    for tokens in (1,257,4096):
        x=torch.randn(tokens,D,device=dev,dtype=torch.bfloat16)
        expert=torch.randint(E,(tokens,),device=dev)
        if tokens==257:expert.zero_() # all empty except one
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,E)
        yp,xp,c,h=forward(x,src,tiles,pu,pd,g)
        grad=torch.randn_like(x)*.02;gate=torch.ones(tokens,device=dev)
        bank=torch.zeros(16,N,D,device=dev,dtype=torch.bfloat16)
        dxp,ds,dg=backward(grad,yp,xp,c,h,gate,src,offs,tiles,pu,pd,g,bank,0)
        y=yp[dest.long()];dx=dxp[dest.long()]
        yr=torch.zeros_like(x);dxr=torch.zeros_like(x);br=torch.zeros_like(bank);gr=torch.zeros_like(g)
        for e in range(E):
            ids=(expert==e).nonzero().flatten()
            if not ids.numel():continue
            xx=x[ids].detach().requires_grad_();u=up[e].float().detach().requires_grad_();d=down[e].T.float().detach().requires_grad_()
            coeff=g[e,:,:6].contiguous().detach().requires_grad_();scale=g[e,:,6].contiguous().detach().requires_grad_();shift=g[e,:,7].contiguous().detach().requires_grad_()
            yy=ks3_blas_fused_unpack(xx,u,d,coeff,scale,shift,subspace_rank=1,nonlinear='fp16',dynamic_rows=True,bf16_products=True)
            grads=torch.autograd.grad(yy,(xx,u,d,coeff,scale,shift),grad[ids])
            yr[ids]=yy;dxr[ids]=grads[0];br[e]=grads[1];br[8+e]=grads[2].T
            gr[e,:,:6]=grads[3];gr[e,:,6]=grads[4];gr[e,:,7]=grads[5]
        errors={}
        for name,a,b in [('output',y,yr),('dx',dx,dxr),('bank',bank,br),('geometry',dg,gr)]:
            error=((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
            assert torch.isfinite(a).all(),name
            assert error<.015,(tokens,name,error)
            errors[name]=error
        checks.append(dict(tokens=tokens,relative_rms=errors,active_experts=int((counts>0).sum())))
    print(json.dumps(dict(passed=True,checks=checks),indent=2))
if __name__=='__main__':main()
