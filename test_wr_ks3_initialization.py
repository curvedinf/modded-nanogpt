"""CPU comparison against the actual pinned PR393 init_moe source body."""
import ast
from pathlib import Path
import types
import unittest
import torch

from wr_reproduction.ks3_initialization import restore_original_moe_rng


class InitializationTests(unittest.TestCase):
    def test_resized_experts_preserve_router_and_all_later_rng_draws(self):
        torch.set_num_threads(1)
        source=Path(__file__).parent/'wr_reproduction/pr393/upstream/track_1_short/model/moe.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MoEMixin')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_moe')
        code=compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec')
        namespace=dict(torch=torch,nn=torch.nn,MOE_LAYERS=(1,3,5,6,8),MOE_EXPERTS=8,
                       MOE_HIDDEN=1024,quantize_weights_dual_ntiles=lambda h,d:1)
        exec(code,namespace)
        def construct(width):
            namespace['MOE_HIDDEN']=width
            model=torch.nn.Module();model.world_size=8;model.device=torch.device('cpu')
            namespace['init_moe'](model,16,0,.01,.05)
            return model
        torch.manual_seed(168)
        original=construct(1024)
        expected_state=torch.get_rng_state().clone()
        expected_later=torch.randn(1000)
        for width in (512,768,1280,1536):
            with self.subTest(charts=width):
                torch.manual_seed(168)
                before=torch.get_rng_state().clone()
                resized=construct(width)
                self.assertFalse(torch.equal(original.moe_router,resized.moe_router))
                restore_original_moe_rng(before,resized.moe_router,16,.05)
                self.assertTrue(torch.equal(original.moe_router,resized.moe_router))
                self.assertTrue(torch.equal(expected_state,torch.get_rng_state()))
                self.assertTrue(torch.equal(expected_later,torch.randn(1000)))
                self.assertEqual(resized.moe_bank.shape,(80,width,16))
                self.assertEqual(resized.moe_bank.c_proj_matrices,original.moe_bank.c_proj_matrices)


if __name__=='__main__':unittest.main()
