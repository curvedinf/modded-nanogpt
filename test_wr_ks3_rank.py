"""CPU architectural checks for the opt-in shared higher-rank KS3 geometry."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from wr_reproduction.check_ks3_rank_geometry import surrogate
from wr_reproduction.geometry_diagnostics import collect
from wr_reproduction.research import ROOT,read
from wr_reproduction.research_controller import validate_recipe


class RankTests(unittest.TestCase):
    def test_zero_geometry_preserves_initial_function_and_input_gradient(self):
        torch.manual_seed(169)
        coords=torch.randn(7,32)
        outputs=[];grads=[]
        for rank in (1,2,4):
            x=coords.clone().requires_grad_()
            g=torch.zeros(32//rank,6*rank+2);g[:,6*rank]=.75
            y=surrogate(x,g,rank);y.sum().backward()
            outputs.append(y);grads.append(x.grad)
        for i in (1,2):
            self.assertTrue(torch.equal(outputs[0],outputs[i]))
            self.assertTrue(torch.equal(grads[0],grads[i]))
        self.assertTrue(torch.equal(outputs[0],coords.relu().square()))

    def test_higher_ranks_couple_only_coordinates_in_the_same_chart(self):
        for rank in (2,4):
            with self.subTest(rank=rank):
                g=torch.zeros(8//rank,6*rank+2);g[:,6*rank]=.75
                g[0,1:4]=.05
                x=torch.tensor([[1.,-1.,2.,-2.,3.,-3.,4.,-4.]],requires_grad=True)
                y=surrogate(x,g,rank);y[0,0].backward()
                self.assertTrue((x.grad[0,1:rank]>0).all())
                self.assertEqual(x.grad[0,rank:].abs().sum().item(),0)
                changed=x.detach().clone();changed[0,1]=0
                self.assertGreater(surrogate(changed,g,rank)[0,0].item(),y[0,0].item())

    def test_higher_rank_terminal_diagnostics_keep_coordinate_and_chart_counts(self):
        for rank in (2,4):
            g=torch.zeros(1,2,3,6*rank+2)
            g[...,0]=.1;g[...,6]=-.2;g[...,6*rank]=.8;g[...,6*rank+1]=.02
            recipe=dict(rank=rank,geometry=dict(coefficients=[0]*6,raw_scale=.75,raw_shift=0.))
            with self.subTest(rank=rank),tempfile.TemporaryDirectory() as directory:
                result=collect(SimpleNamespace(ks3_geometry=g),recipe,Path(directory))
                self.assertTrue(result['all_finite'])
                self.assertEqual(result['shape'],[1,2,3,6*rank+2])
                self.assertEqual(result['components']['c0']['total_count'],6*rank)
                self.assertEqual(result['components']['raw_scale']['total_count'],6)
                self.assertAlmostEqual(result['components']['c0']['delta_rms'],(.05/rank)**.5,places=7)
                self.assertAlmostEqual(result['components']['raw_scale']['delta_rms'],.05,places=7)
                self.assertEqual(result['components']['c1']['delta_rms'],0)

    def test_only_implemented_rank_shapes_and_precision_are_admitted(self):
        recipe=read(ROOT/'configs/wr_reproduction/research_bf16_tuned.json')
        validate_recipe(dict(recipe,charts=512,rank=2))
        recipe.update(charts=256,rank=4)
        validate_recipe(recipe)
        for change in (dict(rank=3),dict(charts=768),dict(precision='fp8'),dict(bf16_kernel_mode='legacy')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))


if __name__=='__main__':unittest.main()
