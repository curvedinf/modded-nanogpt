"""Verify original routed FP8 MoE on MI300, including its in-place bank gradients."""
import json
import sys
import torch
from wr_reproduction.contracts import ROOT,audit
from wr_reproduction.moe_kernels import install

def main():
    audit('pr393');sys.path.insert(0,str(ROOT/'pr393/upstream'))
    from track_1_short.perf.kernels import moe_mlp as m
    install(m)
    torch.manual_seed(393);dev='cuda';E,H,D=8,1024,768
    def scalar(v):return torch.tensor([v],device=dev,dtype=torch.float32)
    up=(torch.randn(E,H,D,device=dev)*.02).to(torch.bfloat16)
    down=(torch.randn(E,H,D,device=dev)*.02).to(torch.bfloat16)
    us=up.float().abs().amax((1,2))/448;ds=down.float().abs().amax((1,2))/448
    u8=(up.float()/us[:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    d8=(down.float()/ds[:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    checks=[]
    for tokens in [1,257,4096]:
        x=torch.randn(tokens,D,device=dev,dtype=torch.bfloat16,requires_grad=True)
        router=(torch.randn(E,D,device=dev)*.01).to(torch.bfloat16).requires_grad_()
        post=scalar(1.).requires_grad_();bank=torch.zeros(80,H,D,device=dev,dtype=torch.bfloat16)
        ps,dps=scalar(.01),scalar(.01);pa,dpa=scalar(0),scalar(0)
        y,probs,counts=m.RoutedFP8MoE.apply(x,router,post,up,down,
            u8.reshape(E*H,D),u8.transpose(1,2).contiguous().transpose(1,2),
            d8.reshape(E*H,D),d8.transpose(1,2).contiguous().transpose(1,2),
            us,ds,scalar(2**-4),scalar(2**-6),scalar(0),ps,pa,dps,dpa,bank,0,
            torch.zeros(1,device=dev,dtype=torch.int32),1)
        aux=.01*E*(counts.float()/tokens*probs.mean(0)).sum()
        grad=torch.randn_like(y)*.01
        actual=torch.autograd.grad((y,aux),(x,router,post),(grad,torch.ones_like(aux)))
        assert counts.sum().item()==tokens
        assert y.shape==x.shape and all(g.isfinite().all().item() for g in [y,*actual,bank])
        assert bank.abs().sum()>0
        # Independent decoded FP8 forward; original router/gate and BF16 epilogue rounding.
        probabilities=torch.nn.functional.linear(x,router).float().softmax(-1)
        gates,experts=probabilities.max(-1)
        ref=torch.empty_like(x);dx_expert=torch.zeros_like(x);ds_gate=torch.zeros(tokens,device=dev)
        bank_ref=torch.zeros_like(bank)
        for e in range(E):
            ids=(experts==e).nonzero().flatten()
            if not ids.numel():continue
            xx=(x.detach()[ids].float()/(2**-4)).clamp(-448,448).to(torch.float8_e4m3fn)
            pre=((xx.float()@u8[e].float().T)*(us[e]*(2**-4))).to(torch.bfloat16)
            pp=pre.clamp_min(0).square()
            pp8=(pp.float()/ps).clamp_max(448).to(torch.float8_e4m3fn)
            yy=((pp8.float()@d8[e].float())*(ps*ds[e])).to(torch.bfloat16)
            ref[ids]=yy*(gates[ids]*post).to(torch.bfloat16)[:,None]
            # Explicit record backward equations, independent of the port's grouped kernels.
            gg=(grad[ids].float()*(gates[ids]*post).float()[:,None]/(2**-6)).clamp(-57344,57344).to(torch.float8_e5m2)
            dp=(2.*((gg.float()@d8[e].float().T)*(ds[e]*(2**-6)))*torch.sqrt(pp8.float()*ps)).to(torch.bfloat16)
            dp8=(dp.float()/dps).clamp(-57344,57344).to(torch.float8_e5m2)
            bank_ref[e]=((dp8.float().T@xx.float())*(dps*(2**-4))).to(torch.bfloat16)
            bank_ref[E+e]=((pp8.float().T@gg.float())*(ps*(2**-6))).to(torch.bfloat16)
            dx_expert[ids]=((dp8.float()@u8[e].float())*(dps*us[e])).to(torch.bfloat16)
            ds_gate[ids]=(grad[ids].float()*yy.float()).sum(-1)
        rms=((y.float()-ref.float()).square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)).item()
        assert rms<.005,rms
        dp_router=(.01*E*counts.float()/tokens**2)[None,:].expand(tokens,E).clone()
        dp_router.scatter_add_(1,experts[:,None],(ds_gate*post).reshape(-1,1))
        dl=(probabilities*(dp_router-(probabilities*dp_router).sum(-1,keepdim=True))).to(torch.bfloat16)
        dx_ref=dx_expert.float()
        for e in range(E):dx_ref=dx_ref+dl[:,e].float()[:,None]*router[e].float()[None,:]
        refs=[dx_ref.to(torch.bfloat16),dl.T@x,(ds_gate*gates).sum().reshape(post.shape),bank_ref]
        grad_checks=[]
        for label,got,want in zip(['dx','router','post','bank'],[*actual,bank],refs):
            error=((got.float()-want.float()).square().mean().sqrt()/want.float().square().mean().sqrt().clamp_min(1e-12)).item()
            assert error<.005,(label,tokens,error)
            grad_checks.append(dict(tensor=label,relative_rms=error))
        checks.append(dict(tokens=tokens,forward_relative_rms=rms,finite_gradients=True,
                           gradients=grad_checks,active_experts=int((counts>0).sum()),post_amax=pa.item(),dpre_amax=dpa.item()))
    print(json.dumps(dict(passed=True,original_routing_and_autograd=True,expert_shape=[E,H,D],checks=checks),indent=2))

if __name__=='__main__':main()
