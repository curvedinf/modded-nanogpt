"""CPU checks for guarded KS3-only deployment after frozen experiments."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_wr_matched_dense_lr as handoff


class HandoffTests(unittest.TestCase):
    def test_stop_prevents_work_before_any_state_read(self):
        with patch.object(handoff, 'stopped', return_value=True), patch.object(handoff, 'read') as read:
            handoff.work('test-only')
            read.assert_not_called()

    def test_both_predecessors_must_finish_without_error(self):
        items = [dict(recipe='warp', status='pruned'), dict(recipe='width', status='running')]
        value = dict(status='running', items=items)
        self.assertFalse(handoff.predecessor_ready(value))
        items[-1]['status'] = 'pruned'
        value['status'] = 'complete_choose_next_hypothesis'
        self.assertTrue(handoff.predecessor_ready(value))
        items[-1]['status'] = 'error'
        with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
            handoff.predecessor_ready(value)

    def test_prototype_scope_and_hash_are_checked_before_mutation(self):
        value = handoff.read(handoff.PROTOTYPE)
        bad = copy.deepcopy(value)
        bad['staged_sources'][0]['path'] = 'wr_reproduction/train.py'
        with self.assertRaisesRegex(RuntimeError, 'only the KS3'):
            handoff.verify_prototype(bad)
        with patch.object(handoff, 'sha', return_value='changed'):
            with self.assertRaisesRegex(RuntimeError, 'patch changed'):
                handoff.verify_prototype(value)

    def test_changed_audit_is_rejected(self):
        value = copy.deepcopy(handoff.read(handoff.PROTOTYPE))
        value['staged_sources'][0]['base_sha256'] = handoff.sha(handoff.ROOT / handoff.SOURCE)
        value['cpu_optimizer_audit']['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'CPU audit changed'):
            handoff.verify_prototype(value)

    def test_extra_patch_destination_cannot_be_applied(self):
        with tempfile.TemporaryDirectory(dir=handoff.STATE) as directory:
            recipe_path = Path(directory) / 'recipe.json'
            with patch.object(handoff, 'RECIPE_PATH', recipe_path), \
                    patch.object(handoff, 'verify_prototype', return_value={}), \
                    patch.object(handoff.controller, 'git', return_value=''), \
                    patch.object(handoff.subprocess, 'check_output', return_value='1\t1\twr_reproduction/train.py\n'), \
                    patch.object(handoff.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'outside the audited KS3'):
                    handoff.deploy(dict(patch='placeholder.patch'), {})
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
