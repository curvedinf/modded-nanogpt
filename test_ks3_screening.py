"""CPU-only tests of the first-observation, dynamic best loss/time watchdog."""
import json
from pathlib import Path
import tempfile
import unittest

from ks3_research.ops.screening import (ReferenceCurves, assess, load_policy, point,
                                      TRAIN_DEFINITION, VAL_DEFINITION)


def metric(loss, wall=100., step=100, **changes):
    row = dict(event='training', train_objective=loss, wall_seconds=wall, step=step,
               objective_signature=dict(stage=2, candidates=14336, mtp=[1.], prefix=0.))
    row.update(changes)
    return row


def observation(loss, run='candidate', wall=100., **changes):
    return point(metric(loss, wall, **changes), {}, run)


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy('configs/ks3/screening_policy.json')

    def assess(self, loss, refs, reached=False, **changes):
        return assess(observation(loss, **changes), refs, reached, self.policy)

    def test_below_five_gate_is_strict_and_opens_on_first_below_observation(self):
        refs = [observation(4., 'mlp-control')]
        for loss in (10., 5.1, 5.):
            result = self.assess(loss, refs)
            self.assertFalse(result['warmup_reached'])
            self.assertFalse(result['eligible'])
            self.assertFalse(result['abort'])
        result = self.assess(4.99, refs)
        self.assertTrue(result['warmup_reached'])
        self.assertTrue(result['abort'])

    def test_exactly_point_zero_one_cancels_on_first_observation(self):
        refs = [observation(4., 'mlp-control')]
        result = self.assess(4.01, refs)
        self.assertEqual(result['gap'], .01)
        self.assertTrue(result['abort'])
        self.assertNotIn('bad_count', result)
        self.assertFalse(self.assess(4.009999, refs)['abort'])

    def test_latch_survives_loss_rising_and_incomparable_measurement(self):
        first = self.assess(4.9, [])
        self.assertTrue(first['warmup_reached'])
        gap = self.assess(6., [], first['warmup_reached'])
        self.assertTrue(gap['warmup_reached'])
        self.assertFalse(gap['abort'])
        self.assertTrue(self.assess(5.5, [observation(4.8, 'ks3-best')], gap['warmup_reached'])['abort'])

    def test_one_sided_better_candidate_always_continues(self):
        result = self.assess(3., [observation(4.5, 'mlp-control')])
        self.assertEqual(result['gap'], -1.5)
        self.assertFalse(result['abort'])

    def test_ks3_replaces_mlp_as_current_best(self):
        refs = [observation(4.5, 'mlp-control')]
        self.assertFalse(self.assess(4.42, refs)['abort'])
        refs.append(observation(4.4, 'run-ks3-better'))
        result = self.assess(4.42, refs)
        self.assertTrue(result['abort'])
        self.assertEqual(result['reference_run_id'], 'run-ks3-better')
        self.assertEqual(result['reference_wall_seconds'], 100.)
        self.assertEqual(result['reference_objective'], 4.4)
        self.assertEqual(result['gap'], .02)

    def test_objectives_and_metric_types_cannot_be_mixed(self):
        candidate = observation(4.2)
        for key, value in (('mtp',[1., .1]), ('prefix',.1), ('candidates',24576),
                           ('attention_windows',[128,384]), ('loss_definition','different')):
            ref = observation(4., 'bad-reference')
            ref['signature'][key] = value
            self.assertFalse(assess(candidate,[ref],True,self.policy)['eligible'])
        ref = observation(4., 'bad-reference')
        ref['signature']['mtp'] = [1.01]  # Former .04 tolerance is unsafe at a .01 margin.
        self.assertFalse(assess(candidate,[ref],True,self.policy)['eligible'])
        ref = dict(observation(4.,'validation'),kind='validation')
        self.assertFalse(assess(candidate,[ref],True,self.policy)['eligible'])

    def test_wall_time_never_step_matching_and_no_stale_or_future_references(self):
        result = self.assess(4.1, [observation(4.,'reference',wall=99.,step=1)], step=99999)
        self.assertTrue(result['abort'])
        self.assertEqual(result['reference_age_seconds'], 1.)
        for wall in (84.99,100.01):
            self.assertFalse(self.assess(4.1,[observation(4.,'reference',wall=wall)])['eligible'])

    def test_corrected_context_does_not_mix_with_verified_defective_training_masks(self):
        legacy=dict(architecture='pr393-moe')
        corrected=dict(architecture='pr393-moe-ks3',
                       attention_mask_semantics='document-causal-local-oracle-v1')
        candidate=point(metric(4.01),corrected,'candidate')
        old=point(metric(4.),legacy,'old-control')
        result=assess(candidate,[old],False,self.policy)
        self.assertTrue(result['warmup_reached'])
        self.assertFalse(result['eligible'])
        new=point(metric(4.),corrected,'verified-ks3')
        result=assess(candidate,[old,new],True,self.policy)
        self.assertTrue(result['abort'])
        self.assertEqual(result['reference_run_id'],'verified-ks3')
        self.assertEqual(result['gap'],.01)

    def test_attention_correction_retains_all_compatible_canonical_full_results(self):
        sig=dict(loss_definition='pr393-cplm-full10485760-canonical-final-yarn-v1',
                 val_tokens=10485760,attention_windows=[768,2560])
        row=dict(event='validation',val_loss=3.67,val_tokens=10485760,full_validation=True,
                 wall_seconds=400.,evaluation_signature=sig)
        candidate=point(row,dict(architecture='pr393-moe-ks3',
            attention_mask_semantics='document-causal-local-oracle-v1'),'candidate')
        reference=point(dict(row,val_loss=3.66,wall_seconds=390.),
                        dict(architecture='pr393-moe-ks3'),'legacy-ks3')
        result=assess(candidate,[reference],False,self.policy)
        self.assertTrue(result['abort'])
        self.assertEqual(result['reference_run_id'],'legacy-ks3')

    def test_choose_latest_training_point_per_run_not_a_cherry_picked_old_minimum(self):
        refs = [observation(3.9,'reference',wall=95.), observation(4.4,'reference',wall=99.)]
        self.assertFalse(self.assess(4.405,refs)['abort'])
        refs.append(observation(0.,'candidate'))
        self.assertEqual(self.assess(4.405,refs)['reference_objective'], 4.4)

    def test_full_validation_uses_canonical_measured_scores_achieved_by_elapsed_time(self):
        sig = dict(loss_definition=VAL_DEFINITION,val_tokens=10485760,attention_windows=[768,2560])
        row = dict(event='validation',val_loss=3.81,val_tokens=10485760,full_validation=True,
                   wall_seconds=2300.,evaluation_signature=sig)
        candidate = point(row,{},'candidate')
        ref = point(dict(row,val_loss=3.79,wall_seconds=2091.),{},'mlp-control')
        result = assess(candidate,[ref],False,self.policy)
        self.assertTrue(result['abort'])
        self.assertEqual(result['gap'],.02)
        self.assertIsNone(point(dict(row,val_tokens=1024),{},'partial'))
        self.assertIsNone(point(dict(row,full_validation=False),{},'partial'))
        legacy = dict(row,final=False,step=700)
        del legacy['evaluation_signature']
        self.assertIsNone(point(legacy,dict(plan=dict(base_steps=1044)),'legacy-interim'))

    def test_errors_are_exceptions_to_warmup(self):
        self.assertTrue(self.assess(float('nan'),[])['abort'])
        self.assertEqual(self.assess(float('inf'),[])['reason'],'nonfinite_loss')

    def test_dynamic_registry_refresh_excludes_self_and_noncontinuous_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root/'state'
            state.mkdir()
            (state/'mlp_reference.json').write_text(json.dumps(dict(observations=[metric(4.5)])))
            def save(run, kind, mode, loss, continuous=True):
                d = state/'runs'/run
                (d/'artifacts').mkdir(parents=True)
                (d/'run.json').write_text(json.dumps(dict(run_id=run,kind=kind)))
                (d/'artifacts/config.json').write_text(json.dumps(dict(mode=mode,continuous_training=continuous)))
                (d/'artifacts/metrics.jsonl').write_text(json.dumps(metric(loss))+'\n')
                return d/'artifacts/metrics.jsonl'
            path = save('ks3','trial','trial',4.4)
            save('candidate','trial','trial',0.)
            save('smoke','smoke','smoke',0.)
            save('pilot','pilot','pilot',0.)
            save('jumped','trial','trial',0.,False)
            curves = ReferenceCurves(root,state,'candidate')
            refs = curves.refresh()
            self.assertEqual({r['run_id'] for r in refs},{'mlp-control','ks3'})
            before = curves.snapshot()['sha256']
            with path.open('a') as out:
                out.write(json.dumps(metric(4.2,wall=105.,step=200))+'\n')
            refs = curves.refresh()
            self.assertNotEqual(curves.snapshot()['sha256'],before)
            result = self.assess(4.22,refs,wall=106.)
            self.assertTrue(result['abort'])
            self.assertEqual(result['reference_run_id'],'ks3')
            self.assertEqual(result['reference_wall_seconds'],105.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
