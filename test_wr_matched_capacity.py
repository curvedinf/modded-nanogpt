"""CPU safety checks for deploying a reviewed KS3 change between experiments."""
import unittest
from unittest.mock import patch

import run_wr_matched_capacity as capacity


class CapacityTests(unittest.TestCase):
    def test_learning_queue_must_finish(self):
        self.assertFalse(capacity.learning_ready(dict(status='running', items=[])))
        self.assertFalse(capacity.learning_ready(dict(status='complete_choose_next_hypothesis', items=[])))
        self.assertTrue(capacity.learning_ready(dict(status='complete_choose_next_hypothesis', items=[dict(status='pruned')])))
        with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
            capacity.learning_ready(dict(items=[dict(status='error', run_id='failed', recipe='candidate')]))

    def test_source_scope_and_hashes_are_checked_before_deployment(self):
        with self.assertRaisesRegex(RuntimeError, 'unapproved'):
            capacity.verify_prototype(dict(staged_sources=[dict(path='wr_reproduction/train.py')]))
        prototype = capacity.read(capacity.PROTOTYPE)
        with patch.object(capacity, 'sha', side_effect=lambda path: prototype['patch_sha256'] if str(path).endswith('.patch') else 'wrong'):
            with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                capacity.verify_prototype(prototype)

    def test_stop_prevents_read_or_deploy(self):
        with patch.object(capacity, 'stopped', return_value=True), patch.object(capacity, 'read') as read:
            capacity.work('test-only')
            read.assert_not_called()

    def test_existing_unfinished_run_is_not_repeated(self):
        progress = dict(correctness=dict(run_id='interrupted'))
        with patch.object(capacity, 'read', return_value=None), patch.object(capacity.controller, 'create_run') as create:
            with self.assertRaisesRegex(RuntimeError, 'reconciliation'):
                capacity.execute_once(progress, 'correctness', {}, 'correctness', 172, 'test-only')
            create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
