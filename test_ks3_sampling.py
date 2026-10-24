"""CPU regression coverage for larger-batch candidate budgets and retained state."""
from concurrent.futures import ThreadPoolExecutor
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ks3_research.continuation import HoldLast
from ks3_research.planning import load_config, plan, schedule_for, signature
from ks3_research.sampling import KS3SampledSoftmax, candidate_stages, scheduled_counts
from mi300.perf.sampled_softmax_overlap import CandidateStaging
from mi300.sampled_softmax import CandidateBuilder, candidate_count_at


class CPUStaging(CandidateStaging):
    """Real single-writer prep/take protocol with synchronous CPU copies."""
    def __init__(self, **kwargs):
        self.prep = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.retired = 0

    def fill_slot(self, candidates, target_pos, prefix_pos):
        return NS(candidates=torch.from_numpy(candidates.copy()),
                  target_pos=torch.from_numpy(target_pos.copy()),
                  prefix_pos=None if prefix_pos is None else torch.from_numpy(prefix_pos.copy()))

    def upload(self, staged, candidates, target_pos, prefix_pos):
        candidates[:len(staged.candidates)].copy_(staged.candidates)
        target_pos[:len(staged.target_pos)].copy_(staged.target_pos)
        if staged.prefix_pos is not None:
            prefix_pos[:len(staged.prefix_pos)].copy_(staged.prefix_pos)

    def wait_uploaded(self, staged):
        pass

    def mark_readers_done(self):
        self.retired += 1


class SamplingTests(unittest.TestCase):
    def test_default_matches_control_every_step_and_continues_full(self):
        config = load_config('configs/ks3/ablation_b96k_s900_trial.json')
        schedule = schedule_for(config, 900, torch.device('cpu'))
        counts = HoldLast(scheduled_counts(schedule, candidate_stages({})))
        self.assertEqual(list(counts), [candidate_count_at(schedule, s)
                                      for s in range(schedule.total_steps+1)])
        self.assertEqual(counts[1000000], 0)

    def test_capacity_regression_and_prefetch_keep_targets_prefix_and_sweep(self):
        schedule = NS(boundaries=[(0,1),(1,2),(2,5),(5,6),(6,7)], total_steps=7)
        counts = [[10240],[10240],[16384,16384,24576],[],[]]
        with patch('ks3_research.sampling.CandidateStaging', CPUStaging):
            sampled = KS3SampledSoftmax(schedule, 50304, 8, 98304, 0, 1,
                                        torch.device('cpu'), counts)
        sampled.counts = HoldLast(sampled.counts)
        prefix = (np.arange(50304, dtype=np.int64)+1) % 50304
        prefix[::17] = -1
        sampled.set_prefix_table(prefix)
        reference = CandidateBuilder(50304,24576,98304)
        reference.reset(0,1)
        ids = (id(sampled.builder),id(sampled.staging),id(sampled.loss_inputs[16384].target_pos))
        try:
            # Reproduce the observed failure's14504 distinct targets, then exercise
            # a candidate-count boundary and final full-vocabulary continuation.
            bad_targets = np.resize(np.arange(14504,dtype=np.int64),98304)
            with self.assertRaisesRegex(AssertionError, '14504 distinct targets'):
                CandidateBuilder(50304,24576,98304).build(14336,bad_targets,None)
            for step, n in ((2,14504),(3,15800),(4,21000)):
                targets = torch.from_numpy(np.resize(np.arange(n,dtype=np.int64),98304))
                p = sampled.counts[step]
                expected, positions, prefix_positions = reference.build(p,targets.numpy(),prefix)
                expected, positions, prefix_positions = [v.copy() for v in (expected,positions,prefix_positions)]
                sampled.prefetch(step,targets)
                sampled.upload(step,targets)
                inputs = sampled.gather(step,torch.empty(0))
                torch.testing.assert_close(sampled.candidates[:p],torch.from_numpy(expected),rtol=0,atol=0)
                np.testing.assert_array_equal(inputs.target_pos.numpy(),positions)
                np.testing.assert_array_equal(inputs.prefix_pos.numpy(),prefix_positions)
                torch.testing.assert_close(sampled.candidates[inputs.target_pos],targets,rtol=0,atol=0)
                self.assertEqual(sampled.builder.sweep_offset,reference.sweep_offset)
                self.assertEqual(ids,(id(sampled.builder),id(sampled.staging),id(inputs.target_pos)))
                self.assertEqual(inputs.rows.shape,(p,8))
                sampled.mark_readers_done()
            offset = sampled.builder.sweep_offset
            for step in (5,7,1000000):
                sampled.prefetch(step,targets)
                sampled.upload(step,targets)
                self.assertIsNone(sampled.gather(step,torch.empty(0)))
            self.assertEqual(sampled.builder.sweep_offset,offset)
            self.assertIsNone(sampled.staging.pending)
            self.assertEqual(sampled.staging.retired,3)
        finally:
            sampled.staging.prep.shutdown(wait=True)

    def test_budget_change_requires_new_pilot_and_invalid_configs_rejected(self):
        config = load_config('configs/ks3/ablation_b96k_s900_trial.json')
        changed = copy.deepcopy(config)
        changed['sampled_candidates'][2] = [16384,16384,24576]
        self.assertNotEqual(signature(config),signature(changed))
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(changed,dict(signature=signature(config)))
        self.assertEqual(candidate_stages({})[2],[14336,14336,24576])
        for invalid in ([[10240]], [[10240],[10240],[14337],[],[]],
                        [[10240],[10240],[16384],[],[10240]],
                        [[10240],[10240],[51200],[],[]]):
            with self.assertRaises(ValueError):
                candidate_stages(dict(sampled_candidates=invalid))


if __name__ == '__main__':
    unittest.main(verbosity=2)
