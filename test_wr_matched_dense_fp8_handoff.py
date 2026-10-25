"""CPU checks for precision gates; no GPU work is performed."""
import copy
import unittest
from unittest.mock import patch
import run_wr_matched_dense_fp8 as handoff


class PrecisionGateTests(unittest.TestCase):
    def test_stop_prevents_work(self):
        with patch.object(handoff, 'stopped', return_value=True), patch.object(handoff, 'read') as read:
            handoff.work('test-only')
            read.assert_not_called()

    def test_scope_and_patch_hash_reject_before_deployment(self):
        value = copy.deepcopy(handoff.read(handoff.PROTOTYPE))
        value['staged_sources'][0]['path'] = 'wr_reproduction/train.py'
        with self.assertRaisesRegex(RuntimeError, 'scope'):
            handoff.verify_prototype(value)
        value = handoff.read(handoff.PROTOTYPE)
        with patch.object(handoff, 'sha', return_value='changed'):
            with self.assertRaisesRegex(RuntimeError, 'patch changed'):
                handoff.verify_prototype(value)

    def test_changed_audit_blocks_precision_gate(self):
        value = copy.deepcopy(handoff.read(handoff.PROTOTYPE))
        for item in value['staged_sources']:
            path = handoff.ROOT / item['path']
            item['base_sha256'] = handoff.sha(path) if path.exists() else None
        value['cpu_audits'][0]['sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'CPU audit changed'):
            handoff.verify_prototype(value)

    def test_both_timing_windows_must_improve(self):
        base = dict(first100_seconds=50., updates25_to100_seconds=30.)
        for candidate in [dict(first100_seconds=51., updates25_to100_seconds=29.),
                          dict(first100_seconds=49., updates25_to100_seconds=31.),
                          dict(first100_seconds=50., updates25_to100_seconds=30.)]:
            self.assertFalse(handoff.timing_decision(base, candidate)['passed'])
        self.assertTrue(handoff.timing_decision(base, dict(first100_seconds=49., updates25_to100_seconds=29.))['passed'])

    def test_only_complete100_update_nonvalidation_timings_accepted(self):
        rows = [dict(event='train', step=25, train_seconds=12.), dict(event='train', step=100, train_seconds=48.)]
        result = dict(status='timing_complete', validation=None, timing=dict(steps=100, train_seconds=49.))
        self.assertEqual(handoff.timing_measurement(result, rows), dict(first100_seconds=49., updates25_to100_seconds=36.))
        for bad in [dict(result, status='error'), dict(result, validation=dict(loss=3.)),
                    dict(result, timing=dict(steps=99, train_seconds=49.))]:
            with self.assertRaises(RuntimeError):
                handoff.timing_measurement(bad, rows)

    def test_nonfinite_time_is_rejected(self):
        rows = [dict(event='train', step=25, train_seconds=12.), dict(event='train', step=100, train_seconds=float('inf'))]
        result = dict(status='timing_complete', validation=None, timing=dict(steps=100, train_seconds=float('inf')))
        with self.assertRaisesRegex(RuntimeError, 'finite'):
            handoff.timing_measurement(result, rows)


if __name__ == '__main__':
    unittest.main()
