"""CPU-only checks for sequential research queue prerequisites."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import run_wr_matched_ablations as queue


class QueueTests(unittest.TestCase):
    def test_all_four_outcomes_are_required(self):
        arms = [dict(architecture=name, status='pruned', run_id=name)
                for name in queue.ARCHITECTURES]
        self.assertFalse(queue.initial_ready(dict(arms=arms[:3])))
        self.assertTrue(queue.initial_ready(dict(arms=arms)))
        arms[-1]['status'] = 'running'
        self.assertFalse(queue.initial_ready(dict(arms=arms)))

    def test_errors_stop_the_queue(self):
        for status in ('error', 'incomplete', 'stopped'):
            with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
                queue.initial_ready(dict(arms=[dict(architecture='ks3_all', status=status, run_id='failed')]))

    def test_changed_recipe_rejected_before_fence_or_launch(self):
        with patch.object(queue, 'sha', return_value='changed'), patch.object(queue, 'assert_fence') as fence:
            with self.assertRaisesRegex(RuntimeError, 'recipe changed'):
                queue.checked_recipe(dict(recipe='example.json', recipe_sha256='prepared'))
            fence.assert_not_called()

    def test_explicit_stop_prevents_state_read_or_launch(self):
        with patch.object(queue, 'stopped', return_value=True), patch.object(queue, 'read') as read:
            queue.run_queue(None, 'test-only')
            read.assert_not_called()

    def test_waiting_dependency_does_not_acquire_predecessor_lease(self):
        state = dict(status='queued', items=[], wait_for_study='precision-study.json')
        with patch.object(queue, 'stopped', side_effect=[False, True]), \
             patch.object(queue, 'read', side_effect=[state, {'controls': []}, {}]), \
             patch.object(queue, 'result_path'), patch.object(queue, 'atomic'), \
             patch.object(queue, 'sha', return_value='scheduler'), \
             patch.object(queue.controller, 'git', return_value='commit'), \
             patch.object(queue, 'initial_ready', return_value=True), \
             patch.object(queue, 'dependency_ready', return_value=False) as dependency, \
             patch.object(queue.Path, 'open', side_effect=AssertionError('Acquired lease before prerequisite')) as opened:
            queue.run_queue(Path('prepared.json'), 'test-only')
            dependency.assert_called_once_with(state)
            opened.assert_not_called()

    def test_dependency_must_finish_without_error(self):
        state = dict(wait_for_run='rank2')
        with patch.object(queue, 'read', return_value=None):
            self.assertFalse(queue.dependency_ready(state))
        with patch.object(queue, 'read', return_value=dict(status='pruned')):
            self.assertTrue(queue.dependency_ready(state))
        with patch.object(queue, 'read', return_value=dict(status='error')):
            with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
                queue.dependency_ready(state)

    def test_qualifying_dependency_requires_completed_verification(self):
        result = dict(run_id='winner', qualifies=True)
        with patch.object(queue, 'read', return_value=dict(groups=[])):
            with self.assertRaisesRegex(RuntimeError, 'verification is unfinished'):
                queue.require_verified(result)
        group = dict(discovery_run=dict(run_id='winner'), status='complete')
        with patch.object(queue, 'read', return_value=dict(groups=[group])):
            queue.require_verified(result)

    def test_study_dependency_waits_through_timing_discovery_and_verification(self):
        state = dict(wait_for_study='precision-study.json')
        for status in ('waiting_for_prerequisites', 'oracle_then_paired_timing', 'dense_fp8_discovery'):
            with patch.object(queue, 'read', return_value=dict(status=status)):
                self.assertFalse(queue.dependency_ready(state))
        outcome = dict(run_id='winner', qualifies=True)
        study = dict(status='complete_choose_next_hypothesis', discovery=dict(result=outcome))
        with patch.object(queue, 'read', return_value=study), patch.object(queue, 'require_verified') as verify:
            self.assertTrue(queue.dependency_ready(state))
            verify.assert_called_once_with(outcome)

    def test_study_error_or_nonlocal_path_blocks_queue(self):
        with patch.object(queue, 'read', return_value=dict(status='error')):
            with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
                queue.dependency_ready(dict(wait_for_study='precision-study.json'))
        with patch.object(queue, 'read') as read:
            with self.assertRaisesRegex(RuntimeError, 'local JSON'):
                queue.dependency_ready(dict(wait_for_study='../outside.json'))
            read.assert_not_called()

    def test_prior_results_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(queue, 'RESULTS', Path(directory)):
            (Path(directory) / 'old.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'preserve prior evidence'):
                queue.result_path(dict(results_name='old.json'))
            with self.assertRaisesRegex(RuntimeError, 'local JSON'):
                queue.result_path(dict(results_name='../outside.json'))

    def test_failed_oracle_blocks_discovery(self):
        self._check_saved_oracle(dict(status='error', correctness=None), error='did not pass')

    def test_passed_oracle_must_match_current_source_and_recipe(self):
        passed = dict(status='correctness_passed', correctness=dict(passed=True))
        self._check_saved_oracle(passed, source='changed', error='differs from the candidate')
        self._check_saved_oracle(passed, recipe=dict(rank=2), error='differs from the candidate')
        self._check_saved_oracle(passed)

    def test_interrupted_oracle_is_not_silently_repeated(self):
        self._check_saved_oracle(None, error='needs reconciliation')

    def _check_saved_oracle(self, result, source='same', recipe=None, error=None):
        item = dict(correctness_required=True, correctness_run_id='oracle',
                    cpu_audit=dict(path='cpu-audit.json', sha256='audited'))
        candidate = dict(rank=4)
        run = dict(run_id='oracle', recipe=candidate if recipe is None else recipe,
                   source_hashes=dict(file='same'))
        reads = [dict(passed=True, gpu_initialized=False), run, result]
        with patch.object(queue, 'sha', return_value='audited'), \
                patch.object(queue, 'read', side_effect=reads), \
                patch.object(queue, 'atomic'), \
                patch.object(queue.controller, 'source_hashes', return_value=dict(file=source)), \
                patch.object(queue.controller, 'execute') as execute, \
                patch.object(queue.controller, 'create_run') as create:
            if error:
                with self.assertRaisesRegex(RuntimeError, error):
                    queue.ensure_correctness(item, candidate, {}, Path('manifest.json'), 'test', {})
            else:
                queue.ensure_correctness(item, candidate, {}, Path('manifest.json'), 'test', {})
            execute.assert_not_called()
            create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
