"""Control handoff must compare only the new matched architecture study."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import torch
from control393.model import MoEGPT,MOE_LAYERS,DENSE_SLOTS
from control393.ks3_model import MoEKS3GPT
from control393.base_gpt import NO_MLP_LAYERS,ATTN_GATE_LAYERS
from ks3_research.ops.screening import ReferenceCurves
from ks3_research.ops import researchctl

class PhaseTests(unittest.TestCase):
    def test_brief_status_preserves_measurements_without_mutating_evidence(self):
        result=dict(val_loss=3.67,full_validation=True,validation_tokens=10485760,
                    complete_wall_seconds=350.,time_to_target_seconds=348.,
                    validations=[dict(val_loss=3.67,batch_seconds=[1.]*40)],
                    tiled_attention_probe=dict(passed=True,checks=[dict(large='evidence')]))
        original=json.dumps(result,sort_keys=True)
        compact=researchctl.brief_result(result)
        self.assertEqual(json.dumps(result,sort_keys=True),original)
        self.assertNotIn('validations',compact)
        self.assertEqual(compact['complete_wall_seconds'],350.)
        self.assertEqual(compact['validation_tokens'],10485760)
        self.assertEqual(compact['tiled_attention_probe'],dict(passed=True,checks=1))

    def test_timestamped_cancel_is_recorded_and_finalized(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);state=root/'study';directory=state/'runs/test';directory.mkdir(parents=True)
            def write(name,value):
                (directory/name).write_text(json.dumps(value))
            write('run.json',dict(run_id='test',kind='smoke',unit='test.service',command=['mock']))
            write('candidate.json',{})
            write('screening_policy.json',dict(margin=.01,direction='worse_only',consecutive_observations=1))
            write('cancel.json',dict(datetime_utc='2026-10-24T00:00:00Z',reason='diagnostic_defect'))
            (state/'registry.json').write_text(json.dumps(dict(active_run='test')))
            process=Mock(pid=123);process.poll.return_value=None;process.wait.return_value=-15
            curves=Mock();curves.refresh.return_value=[];curves.snapshot.return_value=dict(sha256='test')
            with patch.object(researchctl,'ROOT',root),patch.object(researchctl,'STATE',state), \
                 patch.object(researchctl.subprocess,'Popen',return_value=process) as spawn, \
                 patch.object(researchctl,'ReferenceCurves',return_value=curves), \
                 patch.object(researchctl,'terminate') as terminate:
                result=researchctl.worker('test')
            terminate.assert_called_once_with(process)
            self.assertEqual(spawn.call_args.kwargs['env']['KS3_REGISTERED_RUN_ID'],'test')
            self.assertEqual(result['state'],'cancelled')
            self.assertIsNone(json.loads((state/'registry.json').read_text())['active_run'])
            evidence=json.loads((directory/'cancellation.json').read_text())
            self.assertEqual(evidence['datetime_utc'],'2026-10-24T00:00:00Z')
            self.assertIn('recorded_utc',evidence)

    def test_both_stop_markers_are_honored(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);state=root/'logs/ks3-autoresearch';state.mkdir(parents=True)
            control=root/'logs/control393';control.mkdir()
            with patch.object(researchctl,'ROOT',root),patch.object(researchctl,'STATE',state):
                self.assertIsNone(researchctl.stop_marker())
                for marker in (state/'STOP',control/'STOP'):
                    marker.touch()
                    self.assertEqual(researchctl.stop_marker(),marker)
                    marker.unlink()

    def test_three_successful_independent_controls_are_required(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);manifest=root/'configs/control393/control_set.json'
            manifest.parent.mkdir(parents=True)
            config=dict(architecture='pr393-moe-ks3')
            controls=[]
            with patch.object(researchctl,'ROOT',root):
                with self.assertRaises(RuntimeError):researchctl.require_control_cohort(config)
                for seed in (11,22,33):
                    directory=root/str(seed);(directory/'artifacts').mkdir(parents=True)
                    controls.append(dict(seed=seed,directory=str(directory)))
                    (directory/'artifacts/config.json').write_text(json.dumps(dict(seed=seed,architecture='pr393-moe')))
                    (directory/'artifacts/result.json').write_text(json.dumps(dict(exit_code=0,full_validation=True,validation_tokens=10485760,val_loss=3.79)))
                manifest.write_text(json.dumps(dict(status='completed',target=3.793633,controls=controls)))
                researchctl.require_control_cohort(config)
                with self.assertRaises(RuntimeError):researchctl.require_control_cohort(dict(config,screen_seconds=180))
                (root/'33/artifacts/result.json').write_text(json.dumps(dict(exit_code=0,full_validation=False,validation_tokens=262144,val_loss=3.7)))
                with self.assertRaises(RuntimeError):researchctl.require_control_cohort(config)

    def test_corrected_target_rejects_old_wins_and_stale_configs(self):
        from control393.launch_config import load_config
        from control393.launch import compare
        config=load_config('configs/control393/ks3_target368_resume.json')
        self.assertEqual(config['target_val_loss'],3.68)
        self.assertEqual(researchctl.TARGET,3.68)
        result=dict(full_validation=True,validation_tokens=10485760,val_loss=3.760293)
        self.assertFalse(compare(result,config,0,358.82)['exploratory_win'])
        result['val_loss']=3.68
        self.assertTrue(compare(result,config,0,440.)['exploratory_win'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'stale.json';p.write_text(json.dumps(dict(config,target_val_loss=3.793633)))
            with self.assertRaises(ValueError):load_config(p)

    def test_full_trials_require_corrected_attention_after_verified_defect(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'configs/control393/attention_correction.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(status='verified',
                corrected_training_semantics='document-causal-local-oracle-v1')))
            legacy=dict(architecture='pr393-moe-ks3')
            with patch.object(researchctl,'ROOT',root):
                with self.assertRaises(RuntimeError):researchctl.require_attention_correction(legacy,'trial')
                researchctl.require_attention_correction(legacy,'pilot')
                researchctl.require_attention_correction(dict(legacy,
                    attention_mask_semantics='document-causal-local-oracle-v1'),'trial')

    def test_matched_topology_and_parameter_shapes(self):
        with torch.device('meta'):
            kwargs=dict(vocab_size=50257,num_layers=11,num_heads=6,head_dim=128,model_dim=768,
                        max_seq_len=1024,ngram_dim=768,world_size=1,device=torch.device('meta'))
            a=MoEGPT(**kwargs);b=MoEKS3GPT(**kwargs,charts=1024)
        self.assertEqual(MOE_LAYERS,(1,3,5,6,8));self.assertEqual(DENSE_SLOTS,(0,2,4,10,11))
        self.assertEqual(NO_MLP_LAYERS,(7,9));self.assertEqual(ATTN_GATE_LAYERS,())
        for name in ['moe_router','copy_qk_bank','copy_k_sink','copy_qk_gain','mlp_bank']:
            self.assertEqual(getattr(a,name).shape,getattr(b,name).shape)
        self.assertEqual(a.moe_bank.shape,b.moe_bank.shape)
        self.assertEqual(b.ks3_coefficients.shape,(5,8,1024,1,6))

    def test_active_controls_exclude_historical_architecture(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);state=root/'study';state.mkdir()
            directory=root/'control';(directory/'artifacts').mkdir(parents=True)
            signature=dict(loss_definition='pr393-cplm-full10485760-canonical-final-yarn-v1',val_tokens=10485760,attention_windows=[768,2560])
            metric=dict(event='validation',val_loss=3.79,val_tokens=10485760,full_validation=True,wall_seconds=400.,evaluation_signature=signature)
            (directory/'artifacts/config.json').write_text('{}')
            (directory/'artifacts/metrics.jsonl').write_text(json.dumps(metric)+'\n')
            p=root/'configs/control393/control_set.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(status='completed',controls=[dict(directory=str(directory),run_id='seed11')])))
            old=state/'runs/run-old';(old/'artifacts').mkdir(parents=True)
            (old/'run.json').write_text(json.dumps(dict(kind='trial',run_id='run-old')))
            (old/'artifacts/config.json').write_text(json.dumps(dict(mode='trial')))
            (old/'artifacts/metrics.jsonl').write_text(json.dumps(dict(metric,val_loss=1.0))+'\n')
            points=ReferenceCurves(root,state,'new-candidate').refresh()
            self.assertEqual(len(points),1);self.assertEqual(points[0]['run_id'],'control393-seed11')
            self.assertEqual(points[0]['loss'],3.79)

if __name__=='__main__':unittest.main()
