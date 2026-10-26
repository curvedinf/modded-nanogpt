"""CPU policy-scope and lifecycle safeguards for KS3 geometry beta deployment."""
import copy
import unittest
from unittest.mock import patch
import run_wr_matched_geometry_betas as runner
from wr_reproduction.matched_study import RESULTS, read


class GeometryBetaTests(unittest.TestCase):
    def audits(self):
        return (read(RESULTS/'geometry-beta0999-cpu-model-audit.json'),
                read(RESULTS/'geometry-beta0999-cpu-optimizer-audit.json'))

    def test_complete_audits_accepted(self):
        runner.audit_policy(*self.audits())

    def test_changed_non_target_rate_rejected(self):
        model, optimizer = self.audits()
        optimizer['checks'][0]['effective_dense_multiplier'] *= 1.25
        with self.assertRaisesRegex(RuntimeError, 'non-target'):
            runner.audit_policy(model, optimizer)

    def test_wrong_effective_beta_rejected(self):
        model, optimizer = self.audits()
        optimizer['checks'][2]['effective_routed_geometry_betas'] = [.9,.99]
        with self.assertRaisesRegex(RuntimeError, 'non-target'):
            runner.audit_policy(model, optimizer)

    def test_default_regression_rejected(self):
        model, optimizer = self.audits()
        optimizer['default_parameter_tables_identical'] = False
        with self.assertRaisesRegex(RuntimeError, 'default'):
            runner.audit_policy(model, optimizer)

    def test_missing_architecture_or_equation_audit_rejected(self):
        model, optimizer = self.audits()
        optimizer['checks'].pop()
        with self.assertRaisesRegex(RuntimeError, 'misses'):
            runner.audit_policy(model, optimizer)
        model, optimizer = self.audits()
        model['adam_equation_audit']['checks'][1]['updates'] = 1
        with self.assertRaisesRegex(RuntimeError, 'equation'):
            runner.audit_policy(model, optimizer)

    def test_predecessor_terminal_and_not_failed(self):
        self.assertFalse(runner.predecessor_ready({}))
        self.assertFalse(runner.predecessor_ready(dict(status='running',items=[dict(status='running')])))
        for status in ('complete','pruned'):
            self.assertTrue(runner.predecessor_ready(dict(status='complete_choose_next_hypothesis',items=[dict(status=status)])))
        with self.assertRaisesRegex(RuntimeError,'needs diagnosis'):
            runner.predecessor_ready(dict(status='running',items=[dict(status='error')]))

    def test_stop_prevents_read_and_deploy(self):
        with patch.object(runner,'stopped',return_value=True),patch.object(runner,'read') as read,patch.object(runner,'deploy') as deploy:
            runner.work('test-only');read.assert_not_called();deploy.assert_not_called()

    def test_interrupted_discovery_is_not_relaunched(self):
        with patch.object(runner,'read',return_value=None),patch.object(runner.controller,'create_run') as create:
            with self.assertRaisesRegex(RuntimeError,'reconciliation'):
                runner.discovery({},dict(discovery=dict(run_id='existing')),{},'test-only')
            create.assert_not_called()

    def test_shared_source_mutation_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'three KS3'):
            runner.verify_prototype(dict(staged_sources=[dict(path='wr_reproduction/optimizer_kernel.py')]))


if __name__=='__main__':unittest.main()
