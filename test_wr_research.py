"""CPU tests for the WR-only screening, immutable evidence, and verification state."""
import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wr_reproduction.research import (Metadata, point, decision, Screen, References,
    ROOT, CONTROL_LOSS, CONTROL_SECONDS, atomic, append, read, sha, digest,
    derive_saved_references, qualifies, queue_verification, verification_statistics)
from ks3_research.ops.screening import load_policy


class WRResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = Metadata(data=[dict(name='val.bin', sha256='test-data', bytes=100)])
        cls.policy = load_policy(ROOT / 'configs/ks3/screening_policy.json')

    def row(self, loss=4., wall=100., step=601):
        return dict(event='train', step=step, objective=loss, train_seconds=wall,
                    objective_signature=self.metadata.training(step-1))

    def p(self, loss=4., wall=100., step=601, run='candidate'):
        return point(self.row(loss, wall, step), run)

    def val(self, loss=3.4, wall=900., step=800):
        return dict(event='validation', loss=loss, train_seconds=wall, final=False,
                    val_tokens=10485760, step=step, evaluation_signature=self.metadata.validation(step))

    def test_exact_plus_point01_inclusive_and_better_continues(self):
        refs = [self.p(run='mlp')]
        self.assertTrue(decision(self.p(4.01), refs, False, self.policy)['abort'])
        self.assertFalse(decision(self.p(4.009999), refs, False, self.policy)['abort'])
        self.assertFalse(decision(self.p(2.), refs, False, self.policy)['abort'])

    def test_strict_gate_and_persistent_latch(self):
        refs = [self.p(run='mlp')]
        for loss in [8., 5.00001, 5.]:
            d = decision(self.p(loss), refs, False, self.policy)
            self.assertFalse(d['warmup_reached']);self.assertFalse(d['abort'])
        d = decision(self.p(4.999), [], False, self.policy)
        self.assertTrue(d['warmup_reached'])
        self.assertTrue(decision(self.p(6.), refs, d['warmup_reached'], self.policy)['abort'])

    def test_best_ks3_and_latest_point_not_earlier_minimum(self):
        refs = [self.p(4.2, 99., run='mlp'), self.p(3.8, 90., run='ks3'), self.p(4., 98., run='ks3')]
        d = decision(self.p(4.01), refs, False, self.policy)
        self.assertTrue(d['abort']);self.assertEqual(d['reference_run_id'], 'ks3')
        self.assertEqual(d['reference_objective'], 4.)

    def test_wall_time_not_steps_no_future_no_stale(self):
        refs = [self.p(4., 99., step=700, run='reference')]
        self.assertTrue(decision(self.p(4.1,100.,step=601), refs, False, self.policy)['abort'])
        for t in (84.999,100.001):
            self.assertFalse(decision(self.p(4.1), [self.p(4.,t,run='reference')], True, self.policy)['eligible'])
        self.assertTrue(decision(self.p(4.1), [self.p(4.,85.,run='reference')], True, self.policy)['eligible'])

    def test_every_objective_field_matters(self):
        changes = dict(stage=3, candidates=0, global_batch_tokens=7, logical_batch_tokens=7,
                       logical_ranks=1, batch_tokens=7, mtp=[1.,.1], prefix=.1,
                       cplm=False, moe_aux_weight=.02, attention_windows=[128,384],
                       reduction='sum', loss_definition='legacy')
        for key, value in changes.items():
            with self.subTest(key=key):
                r = self.p(run='reference');r['signature'][key]=value
                self.assertFalse(decision(self.p(4.1), [r], True, self.policy)['eligible'])
        self.assertFalse(decision(self.p(4.1), [point(self.val(),'validation')], True,self.policy)['eligible'])

    def test_early_mtp_schedule_excludes_same_stage_different_objective(self):
        self.assertFalse(decision(self.p(4.1,step=501), [self.p(3.,step=500,run='reference')],
                                  True,self.policy)['eligible'])

    def test_stop_nonfinite_override_gate_and_verification(self):
        for verification in (False, True):
            self.assertTrue(decision(self.p(10.), [], False,self.policy,stop=True,verification=verification)['abort'])
            for bad in (math.nan,math.inf):
                self.assertTrue(decision(self.p(bad), [],False,self.policy,verification=verification)['abort'])

    def test_verification_exemption_has_explicit_evidence(self):
        d = decision(self.p(4.5), [self.p(run='discovery')], True,self.policy,verification=True)
        self.assertFalse(d['abort']);self.assertTrue(d['proxy_would_abort'])
        self.assertTrue(d['comparative_pruning_exempt'])

    def test_validation_compatibility_and_strict_budget(self):
        expected=self.metadata.validation(595)
        self.assertTrue(qualifies(self.val(wall=CONTROL_SECONDS),expected))
        self.assertFalse(qualifies(self.val(wall=CONTROL_SECONDS+1e-6),expected))
        self.assertFalse(qualifies(self.val(loss=CONTROL_LOSS),expected))
        self.assertFalse(qualifies(self.val(loss=3.68),expected))
        self.assertTrue(qualifies(self.val(step=1000),expected))  # stage 3 has same eval windows
        for stage_step in (0,300,1040):
            self.assertFalse(qualifies(self.val(step=stage_step),expected))
        for key,value in dict(weights='tail',canonical_mask=True,tail_shipping=True,final_window_extension=True,
                              expert_evaluation='fp8',val_data=[{}],cplm=False).items():
            row=self.val();row['evaluation_signature'][key]=value
            self.assertFalse(qualifies(row,expected))
        row=self.val();row['final']=True
        self.assertIsNone(point(row,'candidate'))
        row=self.val();row['val_tokens']=1024
        self.assertIsNone(point(row,'candidate'))

    def test_reference_evidence_is_preserved(self):
        ref=self.p(run='reference');ref['evidence']=dict(sha256='immutable-hash',line=27)
        d=decision(self.p(4.1),[ref],False,self.policy)
        self.assertEqual(d['reference_evidence'],ref['evidence'])

    def test_exact_saved_index_no_source_mutation_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            original=ROOT/'logs/wr-reproduction/pr393-control-20261024T234443Z/events.jsonl'
            if not original.exists():original=ROOT/'research_results/wr_reproduction/research/pr393-control-20261024T234443Z-events.jsonl'
            before=sha(original)
            index=derive_saved_references(self.metadata,Path(d)/'index.json')
            self.assertEqual(sha(original),before)
            control=[r for r in index['observations'] if r['run_id']=='pr393-control-20261024T234443Z']
            v=next(r for r in control if r['kind']=='validation')
            self.assertEqual((v['loss'],v['wall_seconds']),(CONTROL_LOSS,CONTROL_SECONDS))
            first=next(r for r in control if r['kind']=='training')
            self.assertEqual(first['signature'],self.metadata.training(first['source_step']-1))
        for step,count in [(594,10240),(595,14336),(847,14336),(848,24576),(973,24576),(974,0)]:
            self.assertEqual(self.metadata.training(step)['candidates'],count)

    def test_registry_excludes_noncontinuous_diagnostics_and_self(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for rid,kind,continuous in [('ks3','discovery',True),('candidate','discovery',True),
                                        ('jump','discovery',False),('warmup','diagnostic',True),('profile','profile',True)]:
                directory=root/'runs'/rid
                atomic(directory/'run.json',dict(kind=kind,continuous_from_initialization=continuous))
                append(root/'registry.jsonl',dict(event='registered',run_id=rid))
                append(directory/'events.jsonl',self.row())
                append(directory/'events.jsonl',dict(event='warmup_step',objective=0.,train_seconds=1.))
            refs=References(root,'candidate')
            self.assertEqual({r['run_id'] for r in refs.refresh()},{'ks3'})
            append(root/'runs/ks3/events.jsonl',self.row(3.9,105.))
            self.assertEqual(len(refs.refresh()),2)

    def test_screen_latch_survives_restart_and_stop_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'runs/candidate'
            atomic(run/'run.json',dict(run_id='candidate',kind='discovery'))
            s=Screen(run,root);s.observe(self.row(4.9))
            s=Screen(run,root);self.assertTrue(s.latched)
            (root/'STOP').touch()
            self.assertEqual(s.observe(self.row(10.))['reason'],'explicit_study_stop')

    def test_failed_run_nonfinite_evidence_does_not_break_next_screen(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prior=root/'runs/failed';candidate=root/'runs/candidate'
            atomic(prior/'run.json',dict(kind='discovery',continuous_from_initialization=True))
            append(root/'registry.jsonl',dict(event='registered',run_id='failed'))
            atomic(candidate/'run.json',dict(run_id='candidate',kind='discovery'))
            original_rows=[self.row(4.,99.)]
            for bad in (math.nan,math.inf,-math.inf):
                original_rows.extend([self.row(bad,100.),self.val(loss=bad,wall=100.)])
            path=prior/'events.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in original_rows))
            before=sha(path)
            screen=Screen(candidate,root)
            self.assertEqual(len(screen.references),1)
            self.assertEqual(screen.references[0]['evidence']['sha256'],before)
            self.assertEqual(screen.references[0]['evidence']['line'],1)
            snapshot=read(candidate/('references-'+screen.reference_hash+'.json'))
            json.dumps(snapshot,allow_nan=False)
            result=screen.observe(self.row(4.01))
            self.assertTrue(result['abort']);self.assertEqual(result['reference_run_id'],'failed')
            self.assertEqual(sha(path),before)

    def test_nonfinite_reference_index_and_direct_inputs_excluded(self):
        bad=[self.p(value,99.,run='invalid') for value in (math.nan,math.inf,-math.inf)]
        for ref in bad:
            self.assertFalse(decision(self.p(4.5),[ref],True,self.policy)['eligible'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'reference-index.json').write_text(json.dumps(dict(observations=bad)))
            self.assertEqual(References(root,'candidate').refresh(),[])

    def test_verification_queue_restart_is_idempotent_and_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            discovery=dict(run_id='discovery',recipe=dict(name='same'),source_hashes={'kernel':'hash'},code_commit='commit',seed=168)
            q=queue_verification(d,discovery,dict(loss=3.4))
            self.assertEqual([r['seed'] for r in q['groups'][0]['seeds']],[169,170,171])
            q=queue_verification(d,dict(discovery,run_id='duplicate'),dict(loss=3.3))
            self.assertEqual(len(q['groups']),1)
            self.assertEqual(q['groups'][0]['frozen_run']['code_commit'],'commit')
            q=queue_verification(d,dict(discovery,source_hashes={'kernel':'new'}),dict(loss=3.3))
            self.assertEqual(len(q['groups']),2)

    def test_unchanged_reference_snapshots_skip_hash_but_new_evidence_refreshes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'runs/candidate'
            atomic(run/'run.json',dict(run_id='candidate',kind='discovery'))
            screen=Screen(run,root);before=screen.reference_hash
            screen.refreshed=0
            with patch('wr_reproduction.research.digest',side_effect=AssertionError('Unchanged evidence rehashed')):
                self.assertFalse(screen.observe(self.row(4.5))['eligible'])
            atomic(root/'reference-index.json',dict(observations=[self.p(4.,run='new-reference')]))
            screen.refreshed=0
            self.assertTrue(screen.observe(self.row(4.5))['abort'])
            self.assertNotEqual(before,screen.reference_hash)

    def test_statistics_include_every_seed_and_time(self):
        rows=[dict(seed=169,loss=3.4,train_seconds=900.),dict(seed=170,loss=3.5,train_seconds=900.),
              dict(seed=171,loss=3.4,train_seconds=903.)]
        stats=verification_statistics(rows)
        self.assertEqual(stats['count_beating_control'],1)
        self.assertEqual(stats['count'],3)
        self.assertAlmostEqual(stats['mean'],3.433333333333333)
        self.assertGreater(stats['sample_std'],0.)

    def test_async_observation_wait_is_off_training_thread_and_first_bad_aborts(self):
        from wr_reproduction.research_observer import Observations, ResearchStopped
        class Host:
            def copy_(self, loss, non_blocking):
                self.value=loss.value
                self.non_blocking=non_blocking
            def item(self):return self.value
        class Event:
            def record(self):pass
            def synchronize(self):
                if threading.current_thread() is threading.main_thread():
                    raise AssertionError('Training thread must not wait on observation events')
        fake=SimpleNamespace(empty=lambda *a,**kw:Host(),cuda=SimpleNamespace(Event=Event))
        with tempfile.TemporaryDirectory() as d,patch.dict('sys.modules',torch=fake):
            root=Path(d);run=root/'runs/candidate'
            atomic(run/'run.json',dict(run_id='candidate',kind='discovery'))
            atomic(root/'reference-index.json',dict(observations=[self.p(run='reference')]))
            logged=[]
            obs=Observations(run,lambda **kw:logged.append(kw),lambda t:100.,root)
            try:
                obs.submit(SimpleNamespace(dtype='float32',value=5.),1,self.metadata.training(600))
                obs.drain()
                obs.submit(SimpleNamespace(dtype='float32',value=4.01),2,self.metadata.training(600))
                with self.assertRaises(ResearchStopped):obs.drain()
                self.assertEqual(obs.abort['step'],2)
                self.assertEqual(read(run/'screening_state.json')['gap'],.01)
                self.assertEqual(len([r for r in logged if r['event']=='train']),2)
            finally:obs.close()

    def test_recipe_rejects_unimplemented_changes(self):
        from wr_reproduction.research_controller import validate_recipe
        recipe=read(ROOT/'configs/wr_reproduction/research_fp8_optimized.json')
        validate_recipe(recipe)
        for key,value in [('charts',2048),('max_train_seconds',903.),('validation_precision','fp8')]:
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**{key:value}))
        validate_recipe(dict(recipe,training_gs32=False))
        validate_recipe(dict(recipe,charts=1536))
        validate_recipe(dict(recipe,charts=1280,precision='bf16',bf16_kernel_mode='tuned'))
        with self.assertRaises(ValueError):validate_recipe(dict(recipe,charts=1280))
        with self.assertRaises(ValueError):validate_recipe(dict(recipe,precision='bf16',training_gs32=False))


if __name__=='__main__':
    unittest.main(verbosity=2)
