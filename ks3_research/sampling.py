"""KS3 candidate budgets; preserve the control's sampling/staging implementation.

The buffer initialization follows mi300.sampled_softmax (record #360). Candidate
selection, negative-sweep state, upload/prefetch ordering and gather are inherited.
Only the explicitly configured per-stage candidate counts differ.
"""
import torch

from mi300.sampled_softmax import (CANDIDATES_BY_STAGE, CandidateBuilder,
                                  CandidateStaging, SampledLoss, SampledSoftmax)


def candidate_stages(config):
    values = config.get('sampled_candidates',
                        [list(CANDIDATES_BY_STAGE.get(i, ())) for i in range(5)])
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError('Specify five sampled-candidate stage lists; [] means full vocabulary')
    if any(not isinstance(v, list) or any(type(p) is not int or p < 2048 or
           p > 50304 or p % 2048 for p in v) for v in values):
        raise ValueError('Sampled counts must be positive multiples of2048 below the vocabulary size')
    if values[-1]:
        raise ValueError('The continuing final stage must use the full vocabulary')
    return [list(v) for v in values]


def scheduled_counts(schedule, values):
    counts = []
    for (start, end), stage in zip(schedule.boundaries, values):
        counts.extend(stage[(step-start)*len(stage)//(end-start)] if stage else 0
                      for step in range(start, end))
    return counts + [0]


class KS3SampledSoftmax(SampledSoftmax):
    def __init__(self, schedule, vocab_size, model_dim, max_rows,
                 rank, world_size, device, candidate_counts):
        self.schedule = schedule
        self.rank, self.world_size = rank, world_size
        self.counts = scheduled_counts(schedule, candidate_counts)
        p_values = sorted(set(self.counts)-{0})
        if not p_values or max(p_values) > vocab_size:
            raise ValueError('Sampling requires at least one valid candidate budget')
        max_p = max(p_values)
        self.builder = CandidateBuilder(vocab_size, max_p, max_rows)
        self.staging = CandidateStaging(max_candidates=max_p, max_rows=max_rows, device=device)
        self.prefix_table = None
        self.candidates = torch.zeros(max_p, dtype=torch.int64, device=device)
        target_pos = torch.zeros(max_rows, dtype=torch.int64, device=device)
        prefix_pos = torch.full((max_rows,), -1, dtype=torch.int64, device=device)
        vocab_pos = torch.empty(vocab_size, dtype=torch.int32, device=device)
        self.arange = torch.arange(max_p, dtype=torch.int32, device=device)
        self.loss_inputs = {
            p: SampledLoss(rows=torch.zeros(p, model_dim, dtype=torch.bfloat16, device=device),
                          rows_t=torch.zeros(model_dim, p, dtype=torch.bfloat16, device=device),
                          target_pos=target_pos, prefix_pos=prefix_pos, vocab_pos=vocab_pos)
            for p in p_values
        }
        self.reset()
