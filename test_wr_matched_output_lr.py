"""CPU policy-boundary and lifecycle tests for KS3 output-rate deployment."""
import unittest
from unittest.mock import patch
import run_wr_matched_output_lr as runner
from wr_reproduction.matched_study import RESULTS, read


class OutputRateTests(unittest.TestCase):
    def audits(self):
        return (read(RESULTS/'output-lr1-cpu-model-audit.json'),
                read(RESULTS/'output-lr1-cpu-optimizer-audit.json'))

    def test_audits_accepted(self):
        runner.audit_policy(*self.audits())

    def test_input_or_padding_change_rejected(self):
        for position in (0,15):
            model,optimizer=self.audits()
            optimizer['checks'][2]['dense_per_matrix_lr_multipliers'][position]=3.
            with self.assertRaisesRegex(RuntimeError,'per-matrix'):
                runner.audit_policy(model,optimizer)

    def test_shared_or_retained_policy_change_rejected(self):
        for key,value in [('shared_parameter_configs_identical',False),('retained_original_output_multiplier',1.)]:
            model,optimizer=self.audits();optimizer['checks'][5][key]=value
            with self.assertRaisesRegex(RuntimeError,'non-target'):
                runner.audit_policy(model,optimizer)

    def test_missing_default_or_scalar_staging_rejected(self):
        model,optimizer=self.audits();optimizer['checks'].pop()
        with self.assertRaisesRegex(RuntimeError,'misses'):runner.audit_policy(model,optimizer)
        model,optimizer=self.audits();optimizer['checks'][2]['original_scalar_staging_verified']=False
        with self.assertRaisesRegex(RuntimeError,'non-target'):runner.audit_policy(model,optimizer)

    def test_kernel_change_rejected(self):
        model,optimizer=self.audits();model['kernel_arithmetic_and_model_class_unchanged']=False
        with self.assertRaisesRegex(RuntimeError,'model'):runner.audit_policy(model,optimizer)

    def test_predecessor_must_finish_without_error(self):
        self.assertFalse(runner.predecessor_ready({}))
        self.assertFalse(runner.predecessor_ready(dict(status='running',items=[dict(status='running')])))
        for status in ('complete','pruned'):
            self.assertTrue(runner.predecessor_ready(dict(status='complete_choose_next_hypothesis',items=[dict(status=status)])))
        with self.assertRaisesRegex(RuntimeError,'needs diagnosis'):
            runner.predecessor_ready(dict(status='complete_choose_next_hypothesis',items=[dict(status='error')]))

    def test_stop_prevents_deployment(self):
        with patch.object(runner,'stopped',return_value=True),patch.object(runner,'read') as read,patch.object(runner,'deploy') as deploy:
            runner.work('test-only');read.assert_not_called();deploy.assert_not_called()

    def test_interrupted_discovery_not_relaunched(self):
        with patch.object(runner,'read',return_value=None),patch.object(runner.controller,'create_run') as create:
            with self.assertRaisesRegex(RuntimeError,'reconciliation'):
                runner.discovery({},dict(discovery=dict(run_id='existing')),{},'test-only')
            create.assert_not_called()

    def test_shared_source_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'one KS3'):
            runner.verify_prototype(dict(staged_sources=[dict(path='wr_reproduction/optimizer_kernel.py')]))


if __name__=='__main__':unittest.main()
