"""Independent tests for PR393 architecture semantics and ROCm adaptations."""
import unittest
import os
import torch
import torch.nn.functional as F
from control393.model import routed_mlp,norm_map
from control393.loss import Head,sampled_rows
from control393.copy_pointer import copy_sink_probs
from control393.sampling import CandidateBuilder
import numpy as np

class ControlTests(unittest.TestCase):
    def test_moe_dropless_output_and_all_gradients(self):
        torch.manual_seed(3)
        x=torch.randn(23,12,requires_grad=True)
        up=torch.randn(8,17,12,requires_grad=True)
        down=torch.randn(8,17,12,requires_grad=True)
        router=torch.randn(8,12,requires_grad=True)
        params=[x,up,down,router];out,aux=routed_mlp(*params)
        prob=F.softmax(x@router.T,-1);gate,choice=prob.max(-1)
        values=torch.stack([F.relu(x@up[e].T).square()@down[e] for e in range(8)],1)
        ref=values[torch.arange(len(x)),choice]*gate[:,None]
        counts=F.one_hot(choice,8).float().mean(0)
        refaux=8*(counts*prob.mean(0)).sum()
        torch.testing.assert_close(out,ref,rtol=1e-5,atol=1e-4)
        torch.testing.assert_close(aux,refaux)
        g=torch.randn_like(out)
        a=torch.autograd.grad((out*g).sum()+aux,params,retain_graph=True)
        b=torch.autograd.grad((ref*g).sum()+refaux,params)
        for i,j in zip(a,b):torch.testing.assert_close(i,j,rtol=2e-5,atol=1e-4)

    def test_sampled_head_matches_declared_estimator_and_exact_dx(self):
        torch.manual_seed(4)
        x=torch.randn(64,8,requires_grad=True);w=torch.randn(8,19,requires_grad=True)
        labels=torch.randint(0,18,(64,3));prefix=labels[:,0].clone();cc=18
        logits=Head.apply(x,w,labels,prefix,cc,True,8,0)
        g=torch.rand_like(logits)
        g.scatter_(1,labels,-torch.rand(64,3))
        g[:,cc]=torch.randn(64)
        dx,dw=torch.autograd.grad(logits,(x,w),g)
        ix=sampled_rows(64,8,0,x.device)
        pos=g[ix].clamp_min(0);pos[:,cc]=0
        neg=g.clamp_max(0);neg[:,cc]=0
        ref=4*x[ix].T@pos+x.T@neg
        ref[:,cc]=x.T@g[:,cc]
        torch.testing.assert_close(dx,g@w.T)
        torch.testing.assert_close(dw,ref,atol=1e-5,rtol=1e-5)

    def test_copy_candidate_and_normalization(self):
        b=CandidateBuilder(50304,18432,131072);b.reset(0,1)
        c,t,p=b.build(10240,np.array([1,2,3],dtype=np.int64),None)
        self.assertIn(50257,c);self.assertEqual(len(c),10240)
        self.assertEqual(norm_map().shape,(50304,))

    @unittest.skipUnless(bool(os.environ.get('KS3_REGISTERED_RUN_ID')) and torch.cuda.is_available(),
                         'GPU checks require a registered researchctl experiment')
    def test_pointer_forward_and_gradients_against_dense_document_mask(self):
        torch.manual_seed(17);device='cuda';n=2048;dim=128
        q=torch.randn(n,dim,device=device,dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True);sink=torch.randn(dim,device=device,requires_grad=True)
        tok=torch.randint(0,32,(n,),device=device);tgt=torch.randint(0,32,(n,),device=device)
        cu=torch.tensor([0,700,1100,n,n],device=device,dtype=torch.int32)
        p,a=copy_sink_probs(q,k,sink,tok,tgt,cu,2048)
        positions=torch.arange(n,device=device);doc=torch.searchsorted(cu.long(),positions,right=True)
        valid=(positions[None,:]<positions[:,None])&(doc[:,None]==doc[None,:])
        score=q.float()@k.float().T/dim**.5
        score=score.masked_fill(~valid,float('-inf'))
        s0=q.float()@sink/dim**.5
        prob=torch.softmax(torch.cat([score,s0[:,None]],1),1)
        refp=(prob[:,:n]*(tok[None,:]==tgt[:,None])).sum(1);refa=prob[:,-1]
        torch.testing.assert_close(p,refp,rtol=.015,atol=2e-4)
        torch.testing.assert_close(a,refa,rtol=.015,atol=2e-4)
        gp=torch.randn_like(p);ga=torch.randn_like(a)
        g=torch.autograd.grad((p*gp+a*ga).sum(),[q,k,sink],retain_graph=True)
        ref=torch.autograd.grad((refp*gp+refa*ga).sum(),[q,k,sink])
        for i,j in zip(g,ref):
            relative=((i.float()-j.float()).square().mean()/j.float().square().mean()).sqrt()
            self.assertLess(float(relative),.02)

if __name__=='__main__':unittest.main()
