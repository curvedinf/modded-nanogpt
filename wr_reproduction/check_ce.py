"""Independent loss/autograd oracle for the record's FP8 CE and CPLM kernels."""
import json
import torch
from wr_reproduction.ce_kernel import run

def main():
    torch.manual_seed(393);checks=[]
    for vocab,weights,copy in [(10240,[1.,.5,.25],True),(18432,[1.],True),
                               (50304,[1.],True),(10240,[1.,.5,.25],False)]:
        n=16;device='cuda';gs=(.75/8)/448
        logits=(torch.randn(n,vocab,device=device)*3).to(torch.float8_e4m3fn)
        targets=torch.randint(vocab-1,(n,),device=device);targets[1]=targets[0]
        prefix=targets.clone();prefix[::3]=-1
        mtp=torch.tensor(weights,device=device);pw=torch.tensor([.25],device=device)
        pc=(torch.rand(n,device=device)*.2).requires_grad_();sink=torch.rand(n,device=device,requires_grad=True)
        cc=torch.tensor([vocab-1],device=device,dtype=torch.int32)
        actual=run(logits,targets,mtp,prefix,pw,gs,**(dict(p_copy=pc,a_sink=sink,copy_col=cc) if copy else {}))
        sf=(.5+.5*torch.tanh((logits.float()*(1./7.5)+5.*(1./7.5))*.5)).half().float()
        z=(23.*sf).detach().requires_grad_();lse=23.+torch.exp(z-23.).sum(-1).log()
        rows=torch.arange(n,device=device);loss=torch.zeros(n,device=device)
        for k,weight in enumerate(weights):
            if k==0 and copy:
                pl=torch.exp(z[rows,targets]-lse);al=torch.exp(z[:,vocab-1]-lse)
                mix=pl*(1.+al*sink/(1.-al).clamp_min(1e-6))+al*pc
                loss=loss-weight*(mix+1e-9).log()
            else:loss[:n-k]+=weight*(lse[:n-k]-z[rows[:n-k],targets[k:]])
        loss=loss+torch.where(prefix>=0,pw*(lse-z[rows,prefix.clamp_min(0)]),0)
        derivatives=torch.autograd.grad(loss.sum(),(z,pc,sink) if copy else (z,))
        gz=derivatives[0]
        # Source intentionally retains missing future targets' normalizer gradients at batch tail.
        prob=torch.exp(z.detach()-lse.detach()[:,None])
        for k,weight in enumerate(weights):
            if k:gz[n-k:]+=weight*prob[n-k:]
        g=(gz*((1./7.5*23.)*(1./gs))*sf*(1.-sf)).clamp(-57344,57344).to(torch.float8_e5m2)
        torch.testing.assert_close(actual[0],loss,atol=2e-5,rtol=2e-6)
        rms=((actual[1].float()-g.float()).square().mean().sqrt()/g.float().square().mean().sqrt()).item()
        assert rms<.003,rms
        if copy:
            for a,b in zip(actual[2:],derivatives[1:]):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
        checks.append(dict(vocab=vocab,mtp=weights,cplm=copy,loss_max_abs=(actual[0]-loss).abs().max().item(),gradient_relative_rms=rms))
    print(json.dumps(dict(passed=True,checks=checks),indent=2))

if __name__=='__main__':main()
