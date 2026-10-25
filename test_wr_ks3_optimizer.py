"""KS3 LR override checked through the actual pinned ANVIL config builder."""
import ast
import copy
from dataclasses import asdict,dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from wr_reproduction.ks3_optimizer import configure


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).parent/'wr_reproduction/pr393/upstream/track_1_short/optim/anvil.py'
        tree=ast.parse(path.read_text())
        config=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ParamConfig')
        optim=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnvilAndAdam')
        builder=next(n for n in optim.body if isinstance(n,ast.FunctionDef) and n.name=='_build_param_cfg')
        namespace=dict(__name__=__name__,dataclass=dataclass,nn=torch.nn)
        exec(compile(ast.Module(body=[config,builder],type_ignores=[]),str(path),'exec'),namespace)
        cls.build=staticmethod(namespace['_build_param_cfg'])

    def config(self,multiplier,world=1,rank=0):
        table={'moe_bank':dict(optim='anvil',comms='sharded',adam_betas=None),
               'mlp_bank':dict(optim='anvil',comms='sharded',adam_betas=None),
               'moe_router':dict(optim='adam',comms='replicated',adam_betas=[.9,.99],lr_mul=1.,wd_mul=0.)}
        before=copy.deepcopy(table);original_moe=table['moe_bank']
        configure(table,.125,multiplier)
        self.assertEqual(table['mlp_bank'],before['mlp_bank'])
        self.assertEqual(table['moe_router'],before['moe_router'])
        self.assertEqual(original_moe,before['moe_bank'])
        owner=SimpleNamespace(param_table=table,world_size=world,rank=rank,param_cfgs={},
            anvil_defaults=dict(lr=.02,weight_decay=.1,momentum=.95,beta2=.95),
            adam_defaults=dict(lr=.003,weight_decay=.1,eps=1e-8))
        result={}
        for label in ('moe_bank','mlp_bank','moe_router','ks3_geometry'):
            param=torch.nn.Parameter(torch.empty(80,32,16))
            param.reshape=(80,32,16)
            if label=='moe_bank':
                param.c_proj_matrices=frozenset(k*16+8+e for k in range(5) for e in range(8))
            self.build(owner,param,label);result[label]=asdict(owner.param_cfgs[param])
        return result

    def test_actual_anvil_scales_only_expert_steps_and_preserves_schedule_decay(self):
        for world,rank in ((1,0),(8,0),(8,3),(8,7)):
            baseline=self.config(1.,world,rank)
            for multiplier in (.5,2.):
                with self.subTest(world=world,rank=rank,multiplier=multiplier):
                    result=self.config(multiplier,world,rank)
                    self.assertAlmostEqual(result['moe_bank']['lr_mul'],baseline['moe_bank']['lr_mul']*multiplier)
                    result['moe_bank']['lr_mul']=baseline['moe_bank']['lr_mul']
                    self.assertEqual(result,baseline)
                    self.assertIn(2.,result['moe_bank']['per_matrix_lr_mul'])

    def test_default_preserves_original_moe_entry_and_geometry_settings(self):
        result=self.config(1.)
        self.assertAlmostEqual(result['moe_bank']['lr_mul'],2**.5)
        geometry=result['ks3_geometry']
        self.assertEqual(geometry['lr_mul'],.125)
        self.assertEqual(geometry['adam_betas'],(.9,.99))
        self.assertEqual(geometry['wd_mul'],0.)

    def test_invalid_multiplier_rejected(self):
        for multiplier in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):configure({},.125,multiplier)


if __name__=='__main__':unittest.main()
