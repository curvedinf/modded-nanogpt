"""CPU checks for the direct-encoding handoff and its experiment boundary."""
import copy
import unittest
from unittest.mock import patch
import run_wr_matched_routed_direct_bf16 as handoff


class DirectEncodingGateTests(unittest.TestCase):
    def value(self):
        return copy.deepcopy(handoff.read(handoff.PROTOTYPE))

    def verify(self, value):
        original = handoff.read(handoff.PROTOTYPE)
        deployed = all((handoff.ROOT / item['path']).exists() and handoff.sha(handoff.ROOT / item['path']) == item['staged_sha256']
                       for item in original['staged_sources'])
        return handoff.verify_prototype(value, deployed=deployed)

    def test_explicit_stop_prevents_work(self):
        with patch.object(handoff, 'stopped', return_value=True), patch.object(handoff, 'read') as read:
            handoff.work('test-only')
            read.assert_not_called()

    def test_predecessor_must_finish_one_prepared_trial(self):
        done = dict(status='complete_choose_next_hypothesis', items=[dict(status='pruned')])
        self.assertTrue(handoff.predecessor_ready(done))
        self.assertFalse(handoff.predecessor_ready(dict(done, status='running')))
        self.assertFalse(handoff.predecessor_ready(dict(done, items=[])))
        with self.assertRaises(RuntimeError):
            handoff.predecessor_ready(dict(done, items=[dict(status='error')]))

    def test_audited_prototype_and_fence_pass(self):
        value = self.value()
        recipe, baseline = self.verify(value)
        self.assertIs(recipe['routed_training_gs32'], False)
        self.assertNotIn('routed_training_gs32', baseline)

    def test_shared_path_or_changed_patch_is_rejected(self):
        value = self.value()
        value['staged_sources'][0]['path'] = 'wr_reproduction/train.py'
        with self.assertRaisesRegex(RuntimeError, 'scope'):
            self.verify(value)
        value = self.value()
        value['patch_sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'patch changed'):
            self.verify(value)

    def test_unchanged_ks3_products_are_enforced(self):
        value = self.value()
        value['unchanged_sources']['wr_reproduction/ks3_bf16_gemm.py'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'dependency differs'):
            self.verify(value)

    def test_cpu_audit_and_baseline_must_match(self):
        value = self.value()
        value['cpu_audits'][0]['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'CPU audit changed'):
            self.verify(value)
        value = self.value()
        value['baseline_recipe']['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'baseline recipe changed'):
            self.verify(value)

    def test_only_routed_quantization_flag_can_change(self):
        value = self.value()
        value['recipe']['dense_projection_lr_multiplier'] = .8
        with self.assertRaisesRegex(RuntimeError, 'only the routed quantization flag'):
            self.verify(value)
        value = self.value()
        value['recipe']['routed_training_gs32'] = 0
        with self.assertRaisesRegex(RuntimeError, 'only the routed quantization flag'):
            self.verify(value)

    def test_oracle_must_test_the_requested_encoding(self):
        result = dict(status='correctness_passed', correctness=dict(passed=True, routed_training_projection_gs32=False, original_routed_gs32_regression=True))
        handoff.require_oracle(result, False)
        with self.assertRaisesRegex(RuntimeError, 'different encoding'):
            handoff.require_oracle(result, True)
        with self.assertRaisesRegex(RuntimeError, 'oracle failed'):
            handoff.require_oracle(dict(result, status='error'), False)

    def test_recorded_width512_failure_can_be_resolved_without_shared_fix(self):
        value = handoff.read(handoff.ROOT / self.value()['prerequisite_manifest'])
        self.assertTrue(handoff.predecessor_ready(value))
        wrong = copy.deepcopy(value)
        wrong['items'][0]['run_id'] = 'different-failure'
        with self.assertRaisesRegex(RuntimeError, 'resolution'):
            handoff.predecessor_ready(wrong)

    def test_direct_oracle_requires_original_routed_regression(self):
        result = dict(status='correctness_passed', correctness=dict(passed=True, routed_training_projection_gs32=False))
        with self.assertRaisesRegex(RuntimeError, 'regression'):
            handoff.require_oracle(result, False)

    def test_unfinished_verification_blocks_deployment_mutation(self):
        value = self.value()
        recipe, baseline = self.verify(value)
        stat = ''.join('1\t0\t' + name + '\n' for name in sorted(handoff.ALLOWED))
        with patch.object(handoff, 'verify_prototype', return_value=(recipe, baseline)), \
             patch.object(handoff.Path, 'exists', return_value=False), \
             patch.object(handoff.controller, 'git', return_value=''), \
             patch.object(handoff.subprocess, 'check_output', return_value=stat), \
             patch.object(handoff.subprocess, 'run') as run, \
             patch.object(handoff, 'stopped', return_value=False), \
             patch.object(handoff, 'require_verification_complete', side_effect=RuntimeError('verification unfinished')):
            with self.assertRaisesRegex(RuntimeError, 'verification unfinished'):
                handoff.deploy(value, {})
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:3], ['git', 'apply', '--check'])


if __name__ == '__main__':
    unittest.main()
