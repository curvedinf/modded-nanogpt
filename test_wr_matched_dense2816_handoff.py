"""CPU tests for source isolation, STOP and the mandatory width2816 oracle."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_wr_matched_dense2816 as handoff


class HandoffTests(unittest.TestCase):
    def test_stop_prevents_state_reads(self):
        with patch.object(handoff, 'stopped', return_value=True), patch.object(handoff, 'read') as read:
            handoff.work('test-only')
            read.assert_not_called()

    def test_both_predecessors_must_complete(self):
        items = [dict(recipe='lower-rate', status='pruned'), dict(recipe='width', status='running')]
        state = dict(status='running', items=items)
        self.assertFalse(handoff.predecessor_ready(state))
        items[-1]['status'] = 'pruned'
        state['status'] = 'complete_choose_next_hypothesis'
        self.assertTrue(handoff.predecessor_ready(state))
        items[-1]['status'] = 'error'
        with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
            handoff.predecessor_ready(state)

    def test_verification_blocks_deployment(self):
        with patch.object(handoff, 'read', return_value=dict(groups=[dict(status='running')])):
            with self.assertRaisesRegex(RuntimeError, 'verification is unfinished'):
                handoff.require_verification_complete()

    def test_scope_and_hashes_checked_before_mutation(self):
        value = handoff.read(handoff.PROTOTYPE)
        bad = copy.deepcopy(value)
        bad['staged_sources'][0]['path'] = 'wr_reproduction/train.py'
        with self.assertRaisesRegex(RuntimeError, 'only the two KS3'):
            handoff.verify_prototype(bad)
        with patch.object(handoff, 'sha', return_value='changed'):
            with self.assertRaisesRegex(RuntimeError, 'patch changed'):
                handoff.verify_prototype(value)

    def test_changed_audit_rejected(self):
        value = copy.deepcopy(handoff.read(handoff.PROTOTYPE))
        for item in value['staged_sources']:
            item['base_sha256'] = handoff.sha(handoff.ROOT / item['path'])
        value['cpu_model_audit']['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'CPU audit changed'):
            handoff.verify_prototype(value)

    def test_unexpected_patch_destination_never_applied(self):
        with tempfile.TemporaryDirectory(dir=handoff.STATE) as directory:
            with patch.object(handoff, 'RECIPE_PATH', Path(directory) / 'recipe.json'), \
                    patch.object(handoff, 'verify_prototype', return_value={}), \
                    patch.object(handoff.controller, 'git', return_value=''), \
                    patch.object(handoff.subprocess, 'check_output', return_value='1\t1\twr_reproduction/train.py\n'), \
                    patch.object(handoff.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'outside the audited KS3'):
                    handoff.deploy(dict(patch='placeholder.patch'), {})
                run.assert_not_called()

    def test_oracle_status_and_numerical_pass_required(self):
        for value in [dict(status='error', correctness=dict(passed=True)),
                      dict(status='correctness_passed', correctness=dict(passed=False)),
                      dict(status='correctness_passed')]:
            with self.assertRaisesRegex(RuntimeError, 'oracle did not pass'):
                handoff.require_oracle(value)
        handoff.require_oracle(dict(status='correctness_passed', correctness=dict(passed=True)))


if __name__ == '__main__':
    unittest.main()
