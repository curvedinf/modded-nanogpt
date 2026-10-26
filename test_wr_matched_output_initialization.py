"""CPU lifecycle and evidence safeguards for KS3 output initialization."""
import unittest
from unittest.mock import patch
import run_wr_matched_output_initialization as runner
from wr_reproduction.matched_study import RESULTS, read


class OutputInitializationTests(unittest.TestCase):
    def audits(self):
        return (read(RESULTS/'output-init-gain01-cpu-model-audit.json'),
                read(RESULTS/'output-init-gain01-math-audit.json'))

    def test_complete_audits_accepted(self):
        runner.audit_policy(*self.audits())

    def test_changed_shared_initialization_rejected(self):
        model, mathematical = self.audits()
        model['checks'][0]['global_rng_preserved'] = False
        with self.assertRaisesRegex(RuntimeError, 'Non-target'):
            runner.audit_policy(model, mathematical)

    def test_wrong_initialization_distribution_rejected(self):
        model, mathematical = self.audits()
        candidate = next(x for x in model['checks'] if x['gain'] == .1)
        candidate['initialization_policy']['routed_std'] *= 2
        with self.assertRaisesRegex(RuntimeError, 'Candidate initialization'):
            runner.audit_policy(model, mathematical)

    def test_default_regression_rejected(self):
        model, mathematical = self.audits()
        next(x for x in model['checks'] if x['gain'] == 0.)['initialization_policy']['gain'] = .1
        with self.assertRaisesRegex(RuntimeError, 'Default initialization'):
            runner.audit_policy(model, mathematical)

    def test_missing_architecture_or_math_case_rejected(self):
        model, mathematical = self.audits(); model['checks'].pop()
        with self.assertRaisesRegex(RuntimeError, 'misses'):
            runner.audit_policy(model, mathematical)
        model, mathematical = self.audits(); mathematical['checks'].pop()
        with self.assertRaisesRegex(RuntimeError, 'mathematical'):
            runner.audit_policy(model, mathematical)

    def test_first_update_gradient_claim_requires_evidence(self):
        model, mathematical = self.audits()
        mathematical['checks'][1]['spine_coefficient_gradient_norm'] = 0.
        with self.assertRaisesRegex(RuntimeError, 'gradient-flow'):
            runner.audit_policy(model, mathematical)

    def test_predecessor_terminal_and_not_failed(self):
        self.assertFalse(runner.predecessor_ready({}))
        self.assertFalse(runner.predecessor_ready(dict(status='running', discovery={})))
        for status in ('complete', 'pruned'):
            self.assertTrue(runner.predecessor_ready(dict(status='complete_choose_next_hypothesis',
                            discovery=dict(result=dict(status=status)))))
        with self.assertRaisesRegex(RuntimeError, 'needs diagnosis'):
            runner.predecessor_ready(dict(status='running', discovery=dict(result=dict(status='error'))))

    def test_stop_prevents_read_and_deploy(self):
        with patch.object(runner, 'stopped', return_value=True), patch.object(runner, 'read') as read, \
             patch.object(runner, 'deploy') as deploy:
            runner.work('test-only'); read.assert_not_called(); deploy.assert_not_called()

    def test_interrupted_discovery_is_not_relaunched(self):
        with patch.object(runner, 'read', return_value=None), patch.object(runner.controller, 'create_run') as create:
            with self.assertRaisesRegex(RuntimeError, 'reconciliation'):
                runner.discovery({}, dict(discovery=dict(run_id='existing')), {}, 'test-only')
            create.assert_not_called()

    def test_shared_source_mutation_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'single KS3'):
            runner.verify_prototype(dict(staged_sources=[dict(path='wr_reproduction/optimizer_kernel.py')]))


if __name__ == '__main__':
    unittest.main()
