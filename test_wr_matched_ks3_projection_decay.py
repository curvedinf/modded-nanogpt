"""CPU tests for the scope, predecessor and verification gates of KS3 decay."""
import copy
import unittest
from unittest.mock import patch
import run_wr_matched_ks3_projection_decay as handoff


class DecayHandoffTests(unittest.TestCase):
    def value(self):
        return copy.deepcopy(handoff.read(handoff.PROTOTYPE))

    def verify(self, value):
        item = value['staged_sources'][0]
        deployed = handoff.sha(handoff.ROOT / handoff.SOURCE) == item['staged_sha256']
        return handoff.verify_prototype(value, deployed=deployed)

    def test_stop_prevents_work(self):
        with patch.object(handoff, 'stopped', return_value=True), patch.object(handoff, 'read') as read:
            handoff.work('test-only')
            read.assert_not_called()

    def test_predecessor_requires_terminal_success(self):
        done = dict(status='complete_choose_next_hypothesis', items=[dict(status='pruned')])
        self.assertTrue(handoff.predecessor_ready(done))
        self.assertFalse(handoff.predecessor_ready(dict(done, status='running')))
        self.assertFalse(handoff.predecessor_ready(dict(done, items=[])))
        with self.assertRaisesRegex(RuntimeError, 'diagnosis'):
            handoff.predecessor_ready(dict(done, items=[dict(status='error')]))

    def test_audited_prototype_and_fence_pass(self):
        recipe = self.verify(self.value())
        self.assertEqual(recipe['ks3_projection_weight_decay_multiplier'], .5)

    def test_patch_and_source_scope_are_pinned(self):
        value = self.value()
        value['staged_sources'][0]['path'] = 'wr_reproduction/train.py'
        with self.assertRaisesRegex(RuntimeError, 'scope'):
            self.verify(value)
        value = self.value()
        value['patch_sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'patch changed'):
            self.verify(value)

    def test_dependencies_and_cpu_evidence_are_pinned(self):
        value = self.value()
        value['unchanged_sources']['wr_reproduction/ks3_moe.py'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'dependency'):
            self.verify(value)
        value = self.value()
        value['cpu_optimizer_audit']['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'Audited input changed'):
            self.verify(value)

    def test_recipe_cannot_change_common_or_other_ks3_settings(self):
        for key, replacement in [('shared_fp8_amax', 'other'), ('charts', 768),
                                  ('ks3_projection_weight_decay_multiplier', .25)]:
            value = self.value()
            value['recipe'][key] = replacement
            with self.assertRaisesRegex(RuntimeError, 'only KS3 projection decay'):
                self.verify(value)

    def test_verification_blocks_mutation(self):
        value = self.value()
        recipe = self.verify(value)
        with patch.object(handoff, 'verify_prototype', return_value=recipe), \
             patch.object(handoff.Path, 'exists', return_value=False), \
             patch.object(handoff.controller, 'git', return_value=''), \
             patch.object(handoff.subprocess, 'check_output', return_value='1\t0\t'+handoff.SOURCE+'\n'), \
             patch.object(handoff.subprocess, 'run') as run, \
             patch.object(handoff, 'stopped', return_value=False), \
             patch.object(handoff, 'require_verification_complete', side_effect=RuntimeError('verification unfinished')):
            with self.assertRaisesRegex(RuntimeError, 'verification unfinished'):
                handoff.deploy(value, {})
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:3], ['git', 'apply', '--check'])


if __name__ == '__main__':
    unittest.main()
