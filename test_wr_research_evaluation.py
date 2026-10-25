"""Raw accepted-control evaluation with the actual pinned YaRN implementation."""
import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
from wr_reproduction.research import Metadata, point
from wr_reproduction.research_evaluation import AcceptedEvaluation, WINDOWS


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root=Path(__file__).parent/'wr_reproduction/pr393'
        namespace=dict(__name__=__name__,torch=torch,nn=torch.nn,math=math,
                       dataclass=dataclass,SampledLoss=object,BLOCK_SIZE=128)
        for filename,owner,name in [('model/attention.py',None,'Yarn'),
                                    ('model/gpt.py',None,'ForwardScheduleConfig'),
                                    ('training.py','TrainingManager','advance_schedule')]:
            path=root/'upstream/track_1_short'/filename
            body=ast.parse(path.read_text()).body
            if owner:body=next(n.body for n in body if isinstance(n,ast.ClassDef) and n.name==owner)
            node=next(n for n in body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        cls.Yarn=namespace['Yarn'];cls.Config=namespace['ForwardScheduleConfig']
        cls.advance=staticmethod(namespace['advance_schedule'])
        cls.contract=json.loads((root/'contract.json').read_text())
        cls.metadata=Metadata(data=[dict(name='test',sha256='test',bytes=1)])

    def manager(self):
        yarns=[self.Yarn(64,1024,attn_scale=.12,device='cpu'),
               self.Yarn(64,1024,paired=True,attn_scale=.12,device='cpu'),
               self.Yarn(128,1024,attn_scale=.12,device='cpu')]
        for yarn in yarns:yarn.rebuild_rows=128
        model=torch.nn.Module()
        model.yarns=torch.nn.ModuleList(yarns)
        model.yarn,model.yarn_paired_head,model.yarn_wide=yarns
        model.weight=torch.nn.Parameter(torch.arange(20,dtype=torch.float32))
        stages=[SimpleNamespace(**r) for r in self.contract['stages']]
        manager=SimpleNamespace(model=model,ws_short=1,ws_long=3,train_max_seq_len=896,
            schedule=SimpleNamespace(lookup=lambda step:(stages[self.metadata.stage(step)],0),
                                     mtp_weights=[torch.ones(1)]*1054,
                                     prefix_weights=[torch.zeros(())]*1054))
        manager.advance_schedule=MethodType(self.advance,manager)
        manager.get_forward_args=lambda:self.Config(manager.mtp_weights,manager.prefix_weight,
            manager.ws_short*128,manager.ws_long*128,min(manager.train_max_seq_len,2560),None)
        manager.advance_schedule(0)
        return manager

    def test_late_stage_restores_exact_accepted_tables_and_leaves_raw_weights(self):
        manager=self.manager();baseline=self.manager();policy=AcceptedEvaluation()
        weights=manager.model.weight.detach().clone()
        addresses=[(y.factor1.data_ptr(),y.factor2.data_ptr()) for y in manager.model.yarns]
        for step in (0,280,595,974,1033,1053):
            manager.advance_schedule(step);policy.observe(manager,step)
        for step in (0,280,595):baseline.advance_schedule(step)
        for yarn in baseline.model.yarns:yarn.ensure_full()
        self.assertFalse(torch.equal(manager.model.yarn.angular_freq,baseline.model.yarn.angular_freq))
        original=manager.get_forward_args();cfg,report=policy.prepare(manager)
        self.assertEqual((cfg.ws_short,cfg.ws_long),WINDOWS)
        self.assertEqual((manager.ws_short,manager.ws_long),(6,13))
        self.assertIs(cfg.mtp_weights,original.mtp_weights)
        self.assertIs(cfg.prefix_weight,original.prefix_weight)
        self.assertEqual(cfg.train_max_seq_len,original.train_max_seq_len)
        self.assertTrue(report['restored']);self.assertTrue(report['compatible'])
        self.assertEqual(report['yarn_snapshot_step'],595)
        for a,b,address in zip(manager.model.yarns,baseline.model.yarns,addresses,strict=True):
            self.assertTrue(torch.equal(a.angular_freq,b.angular_freq))
            self.assertEqual(a.attn_scale,b.attn_scale)
            self.assertTrue(torch.equal(a.factor1,b.factor1))
            self.assertTrue(torch.equal(a.factor2,b.factor2))
            self.assertEqual((a.factor1.data_ptr(),a.factor2.data_ptr()),address)
        self.assertTrue(torch.equal(weights,manager.model.weight))

    def test_early_stages_are_not_relabelled_as_accepted(self):
        manager=self.manager();policy=AcceptedEvaluation()
        for step in (0,280):
            manager.advance_schedule(step);policy.observe(manager,step)
            original=manager.get_forward_args();cfg,report=policy.prepare(manager)
            self.assertEqual((cfg.ws_short,cfg.ws_long),(original.ws_short,original.ws_long))
            self.assertFalse(report['compatible']);self.assertFalse(report['restored'])
            sig=self.metadata.validation(step)
            sig['attention_windows']=[cfg.ws_short,cfg.ws_long]
            self.assertIsNone(point(dict(event='validation',final=False,val_tokens=10485760,
                loss=3.,train_seconds=100.,evaluation_signature=sig),'test'))

    def test_accepted_stages_use_existing_state_and_capture_once(self):
        manager=self.manager();policy=AcceptedEvaluation()
        for step in (0,280,595):
            manager.advance_schedule(step);policy.observe(manager,step)
        snapshot=policy.snapshot
        cfg,report=policy.prepare(manager)
        self.assertFalse(report['restored']);self.assertTrue(report['compatible'])
        digest=report['yarn_snapshot_sha256']
        manager.advance_schedule(974);policy.observe(manager,974)
        self.assertIs(policy.snapshot,snapshot)
        cfg,report=policy.prepare(manager)
        self.assertFalse(report['restored']);self.assertEqual(report['yarn_snapshot_sha256'],digest)
        self.assertEqual((cfg.ws_short,cfg.ws_long),WINDOWS)

    def test_late_stage_requires_observed_continuous_state(self):
        manager=self.manager()
        for step in (280,595,1033):manager.advance_schedule(step)
        with self.assertRaisesRegex(RuntimeError,'not captured'):AcceptedEvaluation().prepare(manager)


if __name__=='__main__':unittest.main()
