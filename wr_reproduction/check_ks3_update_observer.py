"""CPU semantics checks for optimizer-zeroing and active-slot diagnostics."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from wr_reproduction.ks3_update_observer import UpdateObserver, specifications


class Model(torch.nn.Module):
    retained_slots=(0,1,2,3)
    def __init__(self):
        super().__init__()
        for name,shape in (('ks3_geometry',(5,8,32,8)),('ks3_dense_geometry',(5,64,8)),
                           ('moe_bank',(80,32,4)),('mlp_bank',(8,2,64,4))):
            self.register_parameter(name,torch.nn.Parameter(torch.randn(shape)))


class Manager:
    def __init__(self,model):
        self.optimizer=SimpleNamespace(param_cfgs={p:SimpleNamespace(optim='adam' if 'geometry' in n else 'anvil') for n,p in model.named_parameters()})
    def is_adam_step(self,step):return step%2==1
    def step_optimizers(self,step,*a,**kw):
        with torch.no_grad():
            for p,c in self.optimizer.param_cfgs.items():
                if c.optim=='adam' and not self.is_adam_step(step):continue
                p.add_(p.grad,alpha=-.01);p.grad=None
        return 'original-return'


def gradients(model,step):
    for name,p in model.named_parameters():
        p.grad=None if 'geometry' in name and step%2==0 else torch.full_like(p,.25 if 'geometry' in name else .5)


class Checks(unittest.TestCase):
    def test_active_slots_and_geometry_components(self):
        m=Model()
        for s in specifications(m):
            indices=s['indices'];self.assertLessEqual(indices.numel(),4096)
            if s['parameter']=='ks3_dense_geometry':self.assertTrue(torch.all(indices//(64*8)==4))
            if s['parameter']=='mlp_bank':self.assertTrue(torch.all(indices//(2*64*4)==4))
            if s['component']=='coefficients':self.assertEqual(set((indices%8).tolist()),set(range(6)))
            if s['component']=='raw_scale':self.assertEqual(set((indices%8).tolist()),{6})
            if s['component']=='raw_shift':self.assertEqual(set((indices%8).tolist()),{7})
            if s['parameter']=='moe_bank':
                expected=0 if s['component']=='up_projection' else 1
                self.assertTrue(torch.all((indices//(8*32*4))%2==expected))

    def test_even_odd_before_clear_parity_and_rng(self):
        torch.manual_seed(17);m=Model();plain=copy.deepcopy(m);manager=Manager(m);baseline=Manager(plain)
        rows=[];observer=UpdateObserver(m,manager,lambda **row:rows.append(row));observer.arm(2)
        for step in (2,3):
            gradients(m,step);gradients(plain,step)
            rng=torch.get_rng_state().clone();pointers=[p.data_ptr() for p in m.parameters()]
            self.assertEqual(manager.step_optimizers(step),'original-return');baseline.step_optimizers(step)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(pointers,[p.data_ptr() for p in m.parameters()])
            for a,b in zip(m.parameters(),plain.parameters()):self.assertTrue(torch.equal(a,b))
        self.assertEqual(observer.remaining,0)
        for key,value in rows[0]['measurements'].items():
            self.assertEqual(value['gradient_available'],'projection' in key)
        for key,value in rows[1]['measurements'].items():
            self.assertTrue(value['gradient_available'])
            self.assertAlmostEqual(value['gradient_sample_rms'],.5 if 'projection' in key else .25,places=6)
            self.assertGreater(value['update_rms'],0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        count=len(rows);gradients(m,4);manager.step_optimizers(4);self.assertEqual(len(rows),count)

    def test_missing_applied_gradient_is_error(self):
        m=Model();observer=UpdateObserver(m,Manager(m),lambda **kw:None)
        with self.assertRaisesRegex(RuntimeError,'Missing reduced gradient'):observer.capture(True)

    def test_extension_clock_adds_actual_cost_without_changing_elapsed(self):
        from wr_reproduction.probe_gradient_extension import ObservedRunner
        from wr_reproduction.train import Runner
        r=ObservedRunner.__new__(ObservedRunner)
        r.update_observer=SimpleNamespace(total_seconds=.25)
        rows=[]
        with patch.object(Runner,'log',lambda self,**kw:rows.append(kw)):
            r.log(event='time_limit',train_seconds=899.,cumulative_instrumentation_seconds=20.,probe_overhead_fraction=20/899.)
        self.assertEqual(rows[0]['train_seconds'],899.)
        self.assertEqual(rows[0]['frozen_v2_instrumentation_seconds'],20.)
        self.assertEqual(rows[0]['cumulative_instrumentation_seconds'],20.25)
        self.assertEqual(rows[0]['probe_overhead_fraction'],20.25/899.)


if __name__=='__main__':unittest.main()
