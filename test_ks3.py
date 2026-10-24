"""Numerical KS3 checks and full-model / wall-clock experiment invariants."""
import copy
import unittest
import torch
import torch.nn.functional as F
from mi300.model.gpt import GPT
from ks3_research.model import KS3GPT, ACTIVE_SLOTS
from ks3_research.planning import load_config, plan, schedule_for, signature, stage_lengths
from ks3_research.launch import compare
from ks3_research.vendor.ks3_reference import KS3
from ks3_research.vendor.ks3_int8 import (KS3Int8, quantize, pack_weight, pack_spine, _project_features)


def unpack(codes, scales):
    return codes.float()*scales.float().repeat_interleave(32,-1)[..., :codes.shape[-1]]


def ste(value, quantized):
    return value+(quantized-value).detach()


def reference(model, inputs):
    """Explicit Hermite basis and autograd, independent of the fused VJP."""
    down, up, coeff = model.master_down, model.master_up, model.master_coefficients
    qx,sx = quantize(inputs.detach().reshape(-1,inputs.shape[-1]))
    qd,sd = pack_weight(down)
    qu,su = pack_weight(up)
    with torch.no_grad():
        pc,ps = pack_spine(coeff)
        scale = F.softplus(model.master_raw_scale)+1e-4
        shift = .5*torch.tanh(model.master_raw_shift)
        h,hs,_,_ = _project_features(qx,sx,qd,sd,pc,ps,scale,shift,model.tanh_lut,
                                     model.subspace_rank,model.nonlinear)
    x = ste(inputs,unpack(qx,sx).reshape_as(inputs))
    d,u = ste(down,unpack(qd,sd)),ste(up,unpack(qu,su))
    c = ste(coeff,(pc.float()*ps.float()[:,None]).reshape_as(coeff))
    coord = F.linear(x,d).clamp(-65504,65504)
    coord = coord.reshape(*inputs.shape[:-1],model.local_subspaces,model.subspace_rank)
    t = 4*torch.tanh(coord.mean(-1,keepdim=True)/4)
    z = torch.tanh((F.softplus(model.master_raw_scale)[:,None]+1e-4)
                   *(t+.5*torch.tanh(model.master_raw_shift)[:,None]))
    basis = torch.stack((2*t**3-3*t*t+1,-2*t**3+3*t*t,t**3-2*t*t+t,
                         t**3-t*t,2*z*z-1,4*z**3-3*z),-1)
    features = F.relu(coord+(basis*c.reshape(model.local_subspaces,model.subspace_rank,6)).sum(-1)).square()
    features = ste(features,unpack(h,hs).reshape_as(features))
    return F.linear(features.flatten(-2),u)


@unittest.skipUnless(torch.cuda.is_available(), 'ROCm GPU required')
class KS3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(4)

    def test_fused_forward_and_all_gradients(self):
        for rank in (1,3):
            with self.subTest(rank=rank):
                model = KS3Int8(KS3(65,37,33,rank).cuda(),nonlinear='fp16',trainable=True)
                x = (torch.randn(2,17,65,device='cuda')*.3).requires_grad_()
                actual,expected = model(x),reference(model,x)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                grad = torch.randn_like(actual)/actual.numel()
                values = (x,*model.parameters())
                a = torch.autograd.grad(actual,values,grad)
                b = torch.autograd.grad(expected,values,grad)
                for left,right in zip(a,b):
                    torch.testing.assert_close(left,right,atol=2e-6,rtol=5e-4)

    def test_full_width_topology_initialization_and_free_capacity(self):
        args = dict(vocab_size=257,num_layers=11,num_heads=6,head_dim=128,model_dim=768,
                    max_seq_len=64,ngram_dim=768,world_size=1,device=torch.device('cuda'))
        torch.manual_seed(0)
        mlp = GPT(**args).cuda()
        torch.manual_seed(0)
        ks3 = KS3GPT(**args,charts=3072).cuda()  # intentionally exceeds the original FFN budget
        original = dict(mlp.named_parameters())
        for name,value in ks3.named_parameters():
            if name in original and name != 'mlp_bank':
                torch.testing.assert_close(value,original[name],atol=0,rtol=0)
        ks3.cast_matrix_weights_bf16()
        self.assertEqual(ks3.mlp_bank.dtype,torch.float32)
        self.assertEqual(ks3.ffn_parameters(),11*3072*(2*768+8))
        self.assertGreater(ks3.ffn_parameters(),11*2*768*2816)
        self.assertEqual(len(ACTIVE_SLOTS),11)
        self.assertIn(11,ACTIVE_SLOTS)  # layer-8 parallel FFN
        self.assertNotIn(7,ACTIVE_SLOTS)
        fcs,projs = ks3._mlp_weights()
        for slot in ACTIVE_SLOTS:
            x = torch.randn(1,3,768,device='cuda',dtype=torch.bfloat16)
            result = ks3._mlp(x,fcs[slot],projs[slot],slot=slot)
            self.assertEqual(result.shape,x.shape)
            self.assertEqual(int(torch.count_nonzero(result)),0)
        self.assertTrue(all(p.requires_grad for p in ks3.parameters()))


class TimePlanningTests(unittest.TestCase):
    def test_comparison_requires_full_loss_and_actual_wall_time(self):
        config = dict(plan=dict(wall_budget_seconds=100.),control_wall_seconds=100.,control_val_loss=3.79)
        good = dict(full_validation=True,validation_tokens=10485760,val_loss=3.78)
        self.assertTrue(compare(good,config,0,99.)['exploratory_win'])
        self.assertFalse(compare(good,config,0,101.)['exploratory_win'])
        self.assertFalse(compare(good,config,1,99.)['exploratory_win'])
        self.assertFalse(compare(dict(good,validation_tokens=1024),config,0,5.)['exploratory_win'])
        self.assertFalse(compare(dict(good,val_loss=3.80),config,0,5.)['exploratory_win'])
        larger_budget = dict(config,plan=dict(wall_budget_seconds=200.))
        self.assertFalse(compare(good,larger_budget,0,150.)['exploratory_win'])

    def test_schedule_accounting_and_time_planning(self):
        config = load_config('configs/ks3/primary.json')
        for base in (648,1122,2140):
            actual = schedule_for(config,base,torch.device('cpu'))
            self.assertEqual(stage_lengths(config,base),[b-a for a,b in actual.boundaries])
        pilot = dict(signature=signature(config),setup_seconds=10.,validation_setup_seconds=5.,
                     validation_batch_seconds=[3.,2.,2.],
                     steps=[dict(stage=s,step_seconds=t) for s,t in enumerate((.3,.5,.8,.7,.3))
                            for _ in range(3)])
        fast = plan(config,pilot)
        slow_pilot = copy.deepcopy(pilot)
        for row in slow_pilot['steps']:
            row['step_seconds'] *= 1.7
        slow = plan(config,slow_pilot)
        self.assertGreater(fast['total_steps'],slow['total_steps'])
        self.assertNotEqual(fast['total_steps'],1194)
        for value in (fast,slow):
            self.assertEqual(value['total_steps']%4,0)
            self.assertLess(value['estimated_total_seconds'],value['schedule_planning_seconds'])
            self.assertEqual(value['termination'], 'full_validation_target_only')
            self.assertNotIn('wall_budget_seconds', value)
        wrong = dict(config,charts=config['charts']+1)
        with self.assertRaisesRegex(ValueError,'pilot'):
            plan(wrong,pilot)


if __name__ == '__main__':
    unittest.main(verbosity=2)
