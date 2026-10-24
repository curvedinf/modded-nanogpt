"""Optimizer graph state/guard tests; actual replay runs only in registry pilots."""
from types import SimpleNamespace as NS
import statistics
import time
import unittest
import torch
from mi300.optim.anvil import AnvilBank
from ks3_research.optim import update_bank
from ks3_research.optimizer_graphs import BankGraphs,fingerprint,check_addresses,restore


def bank(device,dtype,shape,label):
    p=torch.randn(shape,device=device,dtype=dtype)*.03
    lane_shape=(*shape[:-1],1) if shape[-2]>=shape[-1] else (*shape[:-2],1,shape[-1])
    return AnvilBank(label=label,grad=torch.randn_like(p)*.01,p_slice=p,
        velocity=torch.randn((2,*shape),device=device)*.002,
        lane_energy=torch.rand(lane_shape,device=device)*.0001,
        mantissa=torch.zeros(shape,device=device,dtype=torch.uint16),
        momentum=torch.tensor(.95,device=device),eff_wd=torch.tensor(.01,device=device),
        fast_beta=torch.tensor(.9,device=device),fast_weight=torch.tensor(1.,device=device),
        eff_lr=torch.full((shape[0],1,1),.02,device=device),beta2=.9,
        split_baddbmm=shape[-2]>1024,red_dim=-1 if shape[-2]>=shape[-1] else -2)


def copy_bank(value):
    return AnvilBank(**{name:(getattr(value,name).clone() if isinstance(getattr(value,name),torch.Tensor)
                             else getattr(value,name)) for name in value.__slots__})


def run_device_checks(device,update=update_bank,specs=None,benchmark=False):
    torch.manual_seed(107)
    if specs is None:
        specs=(('qk',torch.bfloat16,(2,128,768)),('vo',torch.bfloat16,(2,768,128)),
               ('ks3',torch.float32,(2,1536,768)))
    banks={name:bank(device,dtype,shape,name) for name,dtype,shape in specs}
    reference={name:copy_bank(b) for name,b in banks.items()}
    opt=NS(banks=banks,work_order=list(banks),live_bank_state=lambda name:banks[name].mutated())
    graph=BankGraphs(opt,update)
    for name,b in banks.items():
        for field,tensor in b.mutated().items():
            torch.testing.assert_close(tensor,getattr(reference[name],field),atol=0,rtol=0)
        torch.testing.assert_close(b.grad,reference[name].grad,atol=0,rtol=0)
    # Replay observes changing gradients, learning rates, decay, and twin rails.
    for step in range(6):
        for name,b in banks.items():
            expected=reference[name]
            gradient=torch.randn_like(b.grad)*(.01+step*.001)
            for target in (b,expected):
                target.grad.copy_(gradient)
                target.momentum.fill_(.92-step*.01)
                target.fast_beta.fill_(.85 if step>=3 else .95)
                target.fast_weight.fill_(.4385 if step>=3 else 1.)
                target.eff_wd.fill_(.003+step*.001)
                target.eff_lr.fill_(.02/(step+1))
                target.eff_lr[1::2].mul_(2.)
                target.eff_lr[0].zero_()  # Frozen matrix retains parameter/mantissa.
            frozen=b.p_slice[0].clone()
            update(expected)
            graph.update(b)
            torch.cuda.synchronize()
            for field,tensor in b.mutated().items():
                torch.testing.assert_close(tensor,getattr(expected,field),atol=0,rtol=0)
            torch.testing.assert_close(b.p_slice[0],frozen,atol=0,rtol=0)
            torch.testing.assert_close(b.grad,gradient,atol=0,rtol=0)
    timings=None
    if benchmark:
        samples={'eager':[],'graphs':[]}
        for repetition in range(9):
            for mode in (('eager','graphs') if repetition%2==0 else ('graphs','eager')):
                torch.cuda.synchronize();begun=time.perf_counter()
                if mode=='eager':
                    for b in reference.values():update(b)
                else:
                    for b in banks.values():graph.update(b)
                torch.cuda.synchronize();duration=time.perf_counter()-begun
                if repetition>=3:samples[mode].append(duration)
        for name,b in banks.items():
            for field,tensor in b.mutated().items():
                torch.testing.assert_close(tensor,getattr(reference[name],field),atol=0,rtol=0)
        medians={key:statistics.median(values) for key,values in samples.items()}
        timings=dict(samples_seconds=samples,median_seconds=medians,
                     speedup=medians['eager']/medians['graphs'])
    return dict(passed=True,bank_specs=[dict(label=name,dtype=str(dtype),shape=shape) for name,dtype,shape in specs],
                state_after_capture_bitwise_equal=True,replay_states_bitwise_equal=True,
                dynamic_scalars=True,gradients_preserved=True,paired_benchmark=timings,**graph.stats())


class OptimizerGraphCPUTests(unittest.TestCase):
    def test_capture_state_restoration_preserves_addresses(self):
        b=bank('cpu',torch.float32,(2,16,8),'cpu')
        original=fingerprint(b)
        snapshot={name:t.clone() for name,t in b.mutated().items()}
        for tensor in b.mutated().values():
            tensor.zero_()
        restore(b.mutated(),snapshot)
        self.assertEqual(fingerprint(b),original)
        for name,t in b.mutated().items():
            torch.testing.assert_close(t,snapshot[name],atol=0,rtol=0)

    def test_rebound_buffers_and_live_parameters_are_rejected(self):
        b=bank('cpu',torch.float32,(2,16,8),'cpu')
        original=fingerprint(b)
        check_addresses(b,original,b.mutated())
        live=dict(b.mutated(),p_slice=b.p_slice.clone())
        with self.assertRaisesRegex(RuntimeError,'live state changed'):
            check_addresses(b,original,live)
        b.grad=b.grad.clone()
        with self.assertRaisesRegex(RuntimeError,'buffers changed'):
            check_addresses(b,original,b.mutated())

    def test_graph_backend_requires_new_pilot(self):
        from ks3_research.planning import load_config,plan,signature
        c=load_config('configs/ks3/ablation_fused_softcap_trial.json');old=signature(c)
        c['optimizer_backend']='bank_graphs'
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(c,dict(signature=old))


if __name__=='__main__':
    unittest.main(verbosity=2)
