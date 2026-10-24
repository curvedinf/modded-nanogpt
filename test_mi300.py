"""Focused GPU correctness checks for the new portable attention and loss paths."""
import unittest
import torch
import torch.nn.functional as F
from mi300.portable import varlen_attention, qk_norm_rope, training_loss
from mi300.model.attention import Yarn
from mi300.sampled_softmax import SampledLoss
from mi300.config import TRAINING_STAGES
from mi300.schedule import TrainingSchedule

class PortableTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.device = 'cuda'

    def test_record_schedule(self):
        from track_1_short.config import TRAINING_STAGES as original
        for port, source in zip(TRAINING_STAGES, original):
            self.assertEqual(port.batch_size * 8, source.batch_size)
            for name in ('window_sizes','lr_mul','train_max_seq_len','mtp_weights_start','mtp_weights_end','prefix_weights','duration'):
                self.assertEqual(getattr(port,name),getattr(source,name))
        schedule = TrainingSchedule(TRAINING_STAGES,1122,20,torch.device('cpu'),0.8,4,20)
        self.assertEqual(schedule.total_steps,1194)
        self.assertEqual(schedule.boundaries,[(0,320),(320,681),(681,1107),(1107,1174),(1174,1194)])
        self.assertEqual(schedule.split_step,1175)

    def test_attention_outputs_and_gradients(self):
        # More than one query chunk, unequal head widths, two documents, and a local window.
        q = torch.randn(273,2,64,device=self.device,requires_grad=True)
        k = torch.randn_like(q,requires_grad=True)
        v = torch.randn(273,2,128,device=self.device,requires_grad=True)
        boundaries = torch.tensor([0,11,273,273],device=self.device,dtype=torch.int32)
        out = varlen_attention(q,k,v,boundaries,17,0.13)
        pos = torch.arange(273,device=self.device)
        docs = pos >= 11
        mask = (pos[:,None] >= pos[None,:]) & (pos[:,None]-pos[None,:] <= 17) & (docs[:,None]==docs[None,:])
        scores = torch.einsum('thd,shd->hts',q,k)*0.13
        ref = torch.einsum('hts,shd->thd',scores.masked_fill(~mask,float('-inf')).softmax(-1),v)
        torch.testing.assert_close(out,ref,atol=2e-5,rtol=2e-5)
        upstream = torch.randn_like(out)
        got = torch.autograd.grad((out*upstream).sum(),(q,k,v),retain_graph=True)
        want = torch.autograd.grad((ref*upstream).sum(),(q,k,v))
        for a,b in zip(got,want):
            torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)

    def test_rope_key_offset(self):
        x = torch.randn(16,12,128,device=self.device,requires_grad=True)
        yarn = Yarn(128,16,attn_scale=0.085,device=torch.device(self.device))
        q,k = qk_norm_rope(x,yarn.factor1,yarn.factor2,6,64,False,True)
        normalized = x.float() * (x.float().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).rsqrt()
        torch.testing.assert_close(k[1:,:,64:],normalized[:-1,6:,64:])
        grad, = torch.autograd.grad(q.square().sum()+k.square().sum(),x)
        self.assertTrue(torch.isfinite(grad).all())

    def test_paired_rope(self):
        x = torch.randn(16,12,64,device=self.device,requires_grad=True)
        yarn = Yarn(64,16,paired=True,attn_scale=0.13,device=torch.device(self.device))
        q,k = qk_norm_rope(x,yarn.factor1,yarn.factor2,6,64,True,False)
        self.assertEqual(q.shape,(32,3,64))
        # A rotation preserves each normalized head's squared norm, modulo BF16 rotary tables.
        torch.testing.assert_close(q.square().sum(-1),torch.full((32,3),64.,device=self.device),atol=.35,rtol=.01)
        grad, = torch.autograd.grad(q.sum()+k.sum(),x)
        self.assertTrue(torch.isfinite(grad).all())

    def loss_reference(self,sampled):
        n,d,v=1031,8,32  # crosses a CE slab boundary and exercises the MTP tail
        x=torch.randn(n,d,device=self.device,requires_grad=True)
        w=torch.randn(d,v,device=self.device,requires_grad=True)
        targets=torch.randint(0,16,(n,),device=self.device)
        prefix=torch.arange(v,device=self.device)//2
        prefix[0]=-1
        mw=torch.tensor([1.,.5,.25],device=self.device)
        pw=torch.tensor([.2],device=self.device)
        sample=None
        ids=torch.arange(v,device=self.device)
        if sampled:
            ids=torch.arange(24,device=self.device)
            positions=torch.full((v,),-1,device=self.device,dtype=torch.int32)
            positions[ids]=torch.arange(24,device=self.device,dtype=torch.int32)
            sample=SampledLoss(torch.empty(24,d,device=self.device),torch.empty(d,24,device=self.device),targets,prefix[targets],positions)
        got=training_loss(x,w,targets,prefix,mw,pw,sample)
        logp=F.log_softmax(23*torch.sigmoid((x@w[:,ids]+5)/7.5),-1)
        ref=torch.zeros(n,device=self.device)
        for i in range(3):
            ref[:n-i] -= mw[i]*logp[:n-i].gather(1,targets[i:,None]).squeeze(1)
        pref=prefix[targets]
        ref -= pw[0]*logp.gather(1,pref.clamp_min(0)[:,None]).squeeze(1)*(pref>=0)
        torch.testing.assert_close(got,ref,atol=2e-5,rtol=2e-5)
        a=torch.autograd.grad(got.sum(),(x,w),retain_graph=True)
        b=torch.autograd.grad(ref.sum(),(x,w))
        for actual,expected in zip(a,b):
            torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)

    def test_full_softmax_mtp_prefix(self):
        self.loss_reference(False)

    def test_sampled_softmax_mtp_prefix(self):
        self.loss_reference(True)

if __name__=='__main__':
    unittest.main(verbosity=2)
