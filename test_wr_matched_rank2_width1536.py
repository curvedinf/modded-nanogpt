"""CPU lifecycle safeguards for the staged rank-two capacity experiment."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import run_wr_matched_rank2_width1536 as runner


class RankTwoCapacityTests(unittest.TestCase):
    def gate(self):
        return dict(status='correctness_passed', correctness=dict(passed=True, rank=2,
                    features=1536, charts=768, relative_rms_tolerance=.015,
                    checks=[dict(branch=b, tokens=n, relative_rms={'output': .001})
                            for b in ('dense', 'routed') for n in (1, 257, 4096)],
                    cache_checks=[dict(cache_bytes_exact=True)]*5))

    def test_predecessor_is_terminal_and_not_failed(self):
        self.assertFalse(runner.predecessor_ready({}))
        self.assertFalse(runner.predecessor_ready(dict(status='running', items=[dict(status='running')])))
        for status in ('pruned', 'complete'):
            self.assertTrue(runner.predecessor_ready(dict(status='complete_choose_next_hypothesis', items=[dict(status=status)])))
        with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
            runner.predecessor_ready(dict(status='running', items=[dict(status='error')]))

    def test_complete_numerical_gate(self):
        self.assertTrue(runner.correctness_ok(self.gate()))
        for error in (.015, float('nan'), float('inf')):
            value = self.gate(); value['correctness']['checks'][0]['relative_rms']['output'] = error
            self.assertFalse(runner.correctness_ok(value))

    def test_partial_or_wrong_shape_gate_rejected(self):
        for changes in (dict(rank=1), dict(features=1024), dict(charts=512), dict(passed=False), dict(relative_rms_tolerance=.02)):
            value = self.gate(); value['correctness'].update(changes)
            self.assertFalse(runner.correctness_ok(value))
        value = self.gate(); value['correctness']['checks'].pop()
        self.assertFalse(runner.correctness_ok(value))
        value = self.gate(); value['correctness']['cache_checks'][0]['cache_bytes_exact'] = False
        self.assertFalse(runner.correctness_ok(value))

    def test_correctness_override_leaves_discovery_command(self):
        for kind, seed in (('correctness', 172), ('discovery', 168)):
            run = dict(run_id='test-only', directory='/tmp/test-only', command=['original-trainer'], source_hashes={'frozen': 'source'})
            progress = dict(scheduler={})
            value = dict(oracle_source_hashes={'oracle.py': 'hash'})
            with patch.object(runner.controller, 'create_run', return_value=run) as create, \
                 patch.object(runner.controller, 'execute', return_value={'status': 'test-only'}) as execute, \
                 patch.object(runner, 'atomic'):
                actual, _ = runner.execute_kind(value, progress, {'recipe': 'fixed'}, kind, 'test-only')
                create.assert_called_once_with({'recipe': 'fixed'}, seed, kind, 'test-only')
                self.assertEqual(actual['source_hashes'], {'frozen': 'source'})
                self.assertEqual(actual['command'][1] if kind == 'correctness' else actual['command'][0],
                                 str(runner.ORACLE) if kind == 'correctness' else 'original-trainer')
                execute.assert_called_once_with(actual)

    def test_interrupted_run_is_not_relaunched(self):
        with patch.object(runner, 'read', return_value=None), patch.object(runner.controller, 'create_run') as create:
            with self.assertRaisesRegex(RuntimeError, 'reconciliation'):
                runner.execute_kind({}, {'discovery': {'run_id': 'existing'}}, {}, 'discovery', 'test-only')
            create.assert_not_called()

    def test_stop_before_work_does_not_read_or_deploy(self):
        with patch.object(runner, 'stopped', return_value=True), patch.object(runner, 'read') as read, \
             patch.object(runner, 'deploy') as deploy:
            runner.work('test-only')
            read.assert_not_called(); deploy.assert_not_called()

    def test_wrong_mutation_scope_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'single KS3'):
            runner.verify_prototype(dict(staged_sources=[dict(path='wr_reproduction/backend.py')]))


if __name__ == '__main__':
    unittest.main()
