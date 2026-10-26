"""CPU regression tests for held-out time matching, recovery and continuous accounting."""
import tempfile
import unittest
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from wr_reproduction.probe_study import (decision,point,Screen,initial_protocol,signature,digest,atomic,append,paired,DEFINITION)

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.sig=signature(initial_protocol(),[dict(sha256='text-a')],'backend-a')
    def row(self,loss=4.1,wall=60.):
        return dict(event='probe',measurement=DEFINITION,loss=loss,checkpoint_elapsed_seconds=wall,
                    probe_signature=self.sig,signature_hash=digest(self.sig),chunk_losses=[loss]*8)
    def p(self,loss=4.1,wall=60.,run='candidate'):
        return point(self.row(loss,wall),run)
    def test_strict_gate_and_latch(self):
        ref=self.p(4.8,run='mlp')
        self.assertFalse(decision(self.p(5.),[ref],False)['abort'])
        self.assertTrue(decision(self.p(4.99),[ref],False)['abort'])
        self.assertTrue(decision(self.p(5.01),[ref],True)['abort'])
    def test_first_inclusive_one_sided(self):
        ref=self.p(3.4,run='mlp')
        self.assertTrue(decision(self.p(3.41),[ref],False)['abort'])
        self.assertFalse(decision(self.p(3.40999),[ref],False)['abort'])
        self.assertFalse(decision(self.p(3.),[ref],False)['abort'])
    def test_current_best_and_latest_per_run(self):
        refs=[self.p(3.4,run='mlp'),self.p(3.39,run='ks3')]
        self.assertEqual(decision(self.p(3.4),refs,False)['reference_run_id'],'ks3')
        refs=[self.p(3.,59.,'mlp'),self.p(3.5,60.,'mlp')]
        self.assertFalse(decision(self.p(3.4),refs,False)['abort'])
    def test_signature_future_and_stale_excluded(self):
        for wall in (60.00001,56.9999):
            self.assertEqual(decision(self.p(),[self.p(3.,wall,'mlp')],True)['reason'],'no_comparable_nonfuture_probe')
        for key in ('backend_hash','validation_data','offset','attention_windows'):
            p=self.p(3.,run='mlp');p['signature']=dict(self.sig,**{key:'incompatible'})
            self.assertFalse(decision(self.p(),[p],True)['abort'])
        self.assertTrue(decision(self.p(),[self.p(3.,57.,'mlp')],True)['abort'])
    def test_historical_training_and_forged_hash_excluded(self):
        self.assertIsNone(point(dict(self.row(),event='train'),'old'))
        self.assertIsNone(point(dict(self.row(),signature_hash='wrong'),'old'))
        self.assertIsNone(point(dict(self.row(),measurement='training_objective_proxy_not_full_validation'),'old'))
    def test_exempt_stop_nonfinite(self):
        for exempt in (False,True):
            self.assertTrue(decision(self.p(float('nan')),[],False,exempt=exempt)['abort'])
            self.assertTrue(decision(self.p(),[],False,stop=True,exempt=exempt)['abort'])
        self.assertFalse(decision(self.p(),[self.p(3.,run='mlp')],False,exempt=True)['abort'])
    def test_registry_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);directory=state/'runs'/'candidate';old=state/'runs'/'old';other=state/'runs'/'mlp'
            config=dict(study='probe-study',run_id='candidate',stop_files=[str(state/'STOP')],probe_protocol=initial_protocol())
            atomic(directory/'run.json',config);append(directory/'events.jsonl',self.row(4.9))
            atomic(old/'run.json',dict(study='matched-study',continuous_from_initialization=True,kind='control'))
            atomic(other/'run.json',dict(study='probe-study',continuous_from_initialization=True,kind='control'))
            append(old/'events.jsonl',self.row(1.));append(other/'events.jsonl',self.row(4.))
            for rid in ('old','mlp'):append(state/'registry.jsonl',dict(event='registered',run_id=rid))
            atomic(state/'reference-index.json',dict(observations=[self.p(0.,run='historical')]))
            # An interrupted last registry line must not hide prior complete registrations.
            with (state/'registry.jsonl').open('a') as f:f.write('{"interrupted":')
            screen=Screen(directory,state);self.assertTrue(screen.latched)
            self.assertEqual([p['run_id'] for p in screen.references()],['mlp'])
            self.assertTrue(screen.observe(self.row(4.1))['abort'])
            (state/'STOP').touch();self.assertEqual(screen.observe(self.row(3.))['reason'],'explicit_study_stop')
    def test_clock_includes_setup_scoring_restoration(self):
        from wr_reproduction.probe_evaluation import FixedProbe
        probe=FixedProbe.__new__(FixedProbe);probe.reserved=False;probe.signature=self.sig;probe.signature_hash=digest(self.sig);probe.cumulative=7.;probe.diagnostics=[]
        probe.score=lambda:[4.,4.2]
        with patch('torch.cuda.synchronize'),patch('wr_reproduction.probe_evaluation.time.perf_counter',side_effect=[110.,110.5,112.]):
            row=probe.observe(8,100.)
        self.assertEqual(row['checkpoint_elapsed_seconds'],10.5)
        self.assertEqual(row['probe_completion_seconds'],12.)
        self.assertEqual(row['cumulative_probe_seconds'],9.)
        self.assertTrue(row['setup_and_restore_included'])
    def test_paired_uncertainty_not_repeats(self):
        self.assertFalse(paired([1,2,3,4],[1,1,1,1])['resolves_001'])
        self.assertTrue(paired([1.001,2.001,3.001,4.001],[1,2,3,4])['resolves_001'])
    def test_cohort_environment_retains_root_stop_and_isolates_registry(self):
        from wr_reproduction.probe_study import ROOT_STATE,ROOT_RESULTS,MANAGER
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state';results=root/'results';public=root/'public'
            env=dict(os.environ,WR_PROBE_STATE=str(state),WR_PROBE_RESULTS=str(results),
                     WR_PROBE_PUBLIC_STATE=str(public),WR_PROBE_FENCE=str(root/'fence.json'),
                     WR_PROBE_PROTOCOL=str(root/'protocol.json'),WR_PROBE_VERIFY_SEEDS='369,370,371')
            script='from wr_reproduction.probe_study import *; import json; publish("latest-run.json",{"status":"test"}); print(json.dumps({"environment":environment(),"stops":[str(p) for p in STOP_FILES],"args":systemd_environment()}))'
            data=json.loads(subprocess.check_output([sys.executable,'-c',script],env=env,text=True))
            self.assertEqual(data['environment']['WR_PROBE_STATE'],str(state))
            self.assertIn('--setenv=WR_PROBE_FENCE='+str(root/'fence.json'),data['args'])
            for directory in (ROOT_STATE,ROOT_RESULTS,MANAGER,state,results):
                self.assertIn(str(directory/'STOP'),data['stops'])
            self.assertEqual(json.loads((state/'latest-run.json').read_text()),json.loads((public/'latest-run.json').read_text()))
            self.assertFalse((public/'registry.jsonl').exists())
            self.assertFalse((state/'registry.jsonl').exists())
    def test_calibration_precision_and_budget_are_separate_gates(self):
        from wr_reproduction.probe_precision_report import report
        protocol=dict(initial_protocol(),tokens=1048576,interval_seconds=120.)
        sig=signature(protocol,[dict(sha256='text-a')],'calibration')
        def result(architecture):
            values=[3.+.1*(i%4) for i in range(128)]
            if architecture!='mlp':values=[v-.02+(.0001 if i%2 else -.0001) for i,v in enumerate(values)]
            return dict(run_id=architecture,calibration=dict(passed=True,mode='precision',architecture=architecture,
                state_restoration=dict(passed=True),pending_state_restoration=dict(passed=True),
                continuation_parity=dict(passed=True),oracle_max_abs=0.,probe_signature=sig,signature_hash=digest(sig),
                chunk_losses=values,prefix_elapsed_seconds=120.5,cold_probe_seconds=4.,warm_probe_seconds=2.,pending_probe_seconds=2.))
        mlp,ks3=result('mlp'),result('ks3_experts')
        self.assertEqual(report(mlp,ks3,protocol)['status'],'passed')
        with self.assertRaises(AssertionError):report(mlp,ks3,dict(protocol,attention_windows=[128,384]))
        self.assertEqual(report(mlp,ks3,dict(protocol,reserved_offset=16777216))['status'],'passed')
        ks3['calibration']['warm_probe_seconds']=10.
        value=report(mlp,ks3,protocol)
        self.assertTrue(value['precision_passed']);self.assertFalse(value['budget_passed'])
        ks3['calibration']['warm_probe_seconds']=2.
        ks3['calibration']['chunk_losses']=[v+(.2 if i<64 else -.2) for i,v in enumerate(ks3['calibration']['chunk_losses'])]
        value=report(mlp,ks3,protocol)
        self.assertFalse(value['precision_passed']);self.assertTrue(value['budget_passed'])
    def test_geometry_sampling_covers_all_channels(self):
        import torch
        from wr_reproduction.probe_diagnostics import sample_indices,moments
        for elements in (5*8*1024*8,2816*8,8):
            indices=sample_indices(elements,'cpu')
            counts=torch.bincount(indices%8,minlength=8)
            self.assertTrue((counts>0).all().item())
            self.assertLessEqual((counts.max()-counts.min()).item(),4)
            self.assertEqual(indices[0].item(),0);self.assertEqual(indices[-1].item(),elements-1)
            # A channel-coded geometry makes aligned-stride aliasing observable.
            values=torch.arange(elements)%8
            self.assertLess(abs(moments(values.float())[0].item()-3.5),.01)
if __name__=='__main__':unittest.main()
