"""Training-head objective/gradient equivalence; GPU checks run in registered pilots."""
from types import SimpleNamespace as NS
import unittest
import torch
import torch.nn.functional as F
from ks3_research.loss import shared_gather_loss
from ks3_research.model import KS3GPT
from mi300.model.gpt import GPT
from mi300.model.layers import CastedLinearT
from mi300.portable import training_loss


def error(actual,expected,dtype):
    a,b=actual.detach().float(),expected.detach().float()
    absolute=float((a-b).abs().max())
    relative=float((a-b).norm()/b.norm().clamp_min(1e-12))
    if dtype==torch.bfloat16:
        assert relative<.02,(absolute,relative)
        assert absolute<max(2e-5,.04*float(b.abs().max())),(absolute,relative)
    else:
        assert relative<1e-4,(absolute,relative)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    return dict(max_absolute=absolute,relative_rms=relative)


def check(device,dtype,sampled,mtp,slab_tokens,backend='shared_gather'):
    torch.manual_seed(91)
    n,dim,vocab=2*slab_tokens+1,128,257
    x=torch.randn(n,dim,device=device,dtype=dtype,requires_grad=True)
    w=(torch.randn(dim,vocab,device=device,dtype=dtype)*.03).requires_grad_()
    prefix_table=torch.randint(-1,vocab,(vocab,),device=device)
    weights=torch.tensor(mtp,device=device)
    prefix_weight=torch.tensor([.2 if len(mtp)>1 else 0.],device=device)
    sample=None
    if sampled:
        ids=torch.arange(0,vocab,2,device=device)
        pos=torch.full((vocab,),-1,device=device,dtype=torch.long)
        pos[ids]=torch.arange(ids.numel(),device=device)
        target_pos=torch.randint(ids.numel(),(n,),device=device)
        targets=ids[target_pos]
        prefix_targets=prefix_table[targets]
        prefix_pos=pos[prefix_targets.clamp_min(0)].masked_fill(prefix_targets<0,-1)
        sample=NS(vocab_pos=pos,target_pos=target_pos,prefix_pos=prefix_pos)
    else:
        targets=torch.randint(vocab,(n,),device=device)
        ids=torch.arange(vocab,device=device)
        target_pos=targets
        prefix_pos=prefix_table[targets]
    old=training_loss(x,w,targets,prefix_table,weights,prefix_weight,sample)
    new=shared_gather_loss(x,w,targets,prefix_table,weights,prefix_weight,sample,slab_tokens,
                           fused_softcap=backend=='fused_softcap')
    forward=error(new,old,dtype)
    upstream=torch.randn_like(old)/n
    old_grad=torch.autograd.grad(old,(x,w),upstream)
    new_grad=torch.autograd.grad(new,(x,w),upstream)
    gradients=[error(a,b,dtype) for a,b in zip(new_grad,old_grad)]
    if sampled:
        excluded=sample.vocab_pos<0
        assert torch.count_nonzero(new_grad[1][:,excluded])==0
    # Independent full-matrix reference verifies shifts across slab boundaries
    # and invalid final MTP / absent-prefix labels, not just the legacy wrapper.
    if dtype==torch.float32:
        logits=x@w[:,ids]
        logp=(23*torch.sigmoid((logits.float()+5)/7.5)).log_softmax(-1)
        expected=torch.zeros(n,device=device)
        for offset,weight in enumerate(mtp):
            rows=torch.arange(n-offset,device=device)
            expected[:n-offset]-=weight*logp[rows,target_pos[offset:]]
        good=prefix_pos>=0
        rows=torch.arange(n,device=device)[good]
        expected[good]-=prefix_weight[0]*logp[rows,prefix_pos[good]]
        torch.testing.assert_close(new,expected,atol=2e-5,rtol=2e-5)
    return dict(dtype=str(dtype),sampled=sampled,mtp=mtp,slab_tokens=slab_tokens,
                forward=forward,gradients=gradients)


def check_evaluation(device,backend='shared_gather'):
    # Exercise the actual override with only the loss-head fields constructed.
    model=KS3GPT.__new__(KS3GPT)
    torch.nn.Module.__init__(model)
    model.loss_backend,model.loss_slab_tokens=backend,4096
    model.lm_head=CastedLinearT(32,256,1.,1.,1.).to(device)
    torch.nn.init.normal_(model.lm_head.weight,std=.03)
    model.register_buffer('canon_mask',torch.randint(0,255,(256,32),device=device,dtype=torch.uint8))
    model.register_buffer('prefix_table',torch.arange(256,device=device))
    model.eval()
    x=torch.randn(1,67,32,device=device,dtype=torch.bfloat16)
    inputs=torch.randint(256,(67,),device=device)
    targets=torch.randint(256,(67,),device=device)
    weights=torch.ones(1,device=device)
    prefix=torch.zeros(1,device=device)
    with torch.no_grad():
        actual=model._loss(x,inputs,targets,weights,prefix,None)
        expected=GPT._loss(model,x,inputs,targets,weights,prefix,None)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    return True


def check_large_vocabulary(device,dtype,vocab):
    from ks3_research.softcap_ce import softcap_ce
    torch.manual_seed(93)
    # Strided rows, identical/absent/repeated labels, saturated logits, and
    # arbitrary signed upstream derivatives exercise the actual row kernels.
    n=17
    logits=(torch.randn(vocab,n,device=device,dtype=dtype)*3).T
    logits[0].zero_();logits[1,::2]=1000.;logits[1,1::2]=-1000.
    logits.requires_grad_()
    labels=torch.randint(vocab,(n,3),device=device)
    labels[2]=-1;labels[3,1]=labels[3,0];labels[-1,1:]=-1
    prefixes=torch.randint(vocab,(n,),device=device)
    prefixes[2]=-1;prefixes[3]=labels[3,0];prefixes[4]=-1
    mw=torch.tensor([1.,.3,.2],device=device);pw=torch.tensor([.15],device=device)
    logp=(23*torch.sigmoid((logits.float()+5)/7.5)).log_softmax(-1)
    old=torch.zeros(n,device=device)
    for i in range(3):
        y=labels[:,i]
        old-=mw[i]*logp.gather(1,y.clamp_min(0)[:,None]).squeeze(1)*(y>=0)
    old-=pw[0]*logp.gather(1,prefixes.clamp_min(0)[:,None]).squeeze(1)*(prefixes>=0)
    new=softcap_ce(logits,labels,prefixes,mw,pw)
    upstream=torch.randn(n,device=device)
    a=torch.autograd.grad(new,logits,upstream)[0]
    b=torch.autograd.grad(old,logits,upstream)[0]
    assert new[2]==0 and torch.count_nonzero(a[2])==0
    return dict(case='large_vocabulary_row',vocab=vocab,dtype=str(dtype),
                forward=error(new,old,dtype),gradients=[error(a,b,dtype)])


def run_device_checks(device,slab_tokens=4096,backend='shared_gather'):
    device=torch.device(device)
    dtypes=(torch.float32,torch.bfloat16) if device.type=='cuda' else (torch.float32,)
    checks=[check(device,dtype,sampled,mtp,slab_tokens,backend) for dtype in dtypes
            for sampled,mtp in ((False,[1.]),(False,[1.,.5,.25]),(True,[1.,.5,.25]))]
    if backend=='fused_softcap':
        checks.extend(check_large_vocabulary(device,dtype,vocab) for dtype in dtypes
                      for vocab in (10240,18432,24576,50304))
    return dict(checks=checks,backend=backend,evaluation_bitwise_equal=check_evaluation(device,backend),passed=True)


class LossHeadTests(unittest.TestCase):
    def test_objective_gradients_boundaries_sampling_and_evaluation(self):
        self.assertTrue(run_device_checks('cpu')['passed'])

    def test_fused_head_wrapper_and_objective(self):
        self.assertTrue(run_device_checks('cpu',4096,'fused_softcap')['passed'])

    def test_fused_head_requires_new_timing_pilot(self):
        from ks3_research.planning import load_config,plan,signature
        config=load_config('configs/ks3/ablation_b128k_fast_trial.json')
        old=signature(config)
        config['loss_backend']='fused_softcap'
        self.assertNotEqual(old,signature(config))
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(config,dict(signature=old))


if __name__=='__main__':
    unittest.main(verbosity=2)
