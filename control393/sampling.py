"""PR393 candidate construction, with existing MI300 staging."""
import numpy as np
import math
from dataclasses import dataclass
import torch
from mi300.sampled_softmax import NEGATIVE_STRIDE
from mi300.sampled_softmax import SampledLoss
from ks3_research.sampling import KS3SampledSoftmax
COPY_TOKEN_ID=50257


@dataclass(slots=True)
class SampledHeadInputs(SampledLoss):
    candidate_ids: torch.Tensor
    copy_column: int


def host_copy_column(staged):
    candidates=staged.slot.candidates[:staged.num_candidates].numpy()
    column=int(np.searchsorted(candidates,COPY_TOKEN_ID))
    if column>=len(candidates) or candidates[column]!=COPY_TOKEN_ID:
        raise ValueError('CPLM candidate set omitted the copy token')
    return column

class CandidateBuilder:
    """Host half of the candidate build: numpy only, no device work, so it can run on a worker thread.

    Scratch arrays are allocated once and reused, so at most one build may run at a time.
    """
    def __init__(self, vocab_size: int, max_candidates: int, max_rows: int):
        assert math.gcd(NEGATIVE_STRIDE, vocab_size) == 1
        self.vocab_size = V = vocab_size
        self.mark = np.zeros(V + 1, dtype=bool)          # candidate bitmap
        self.mark_u8 = self.mark.view(np.uint8)
        # class -> position in C. Entry V is a sentinel that stays -1, so a prefix target of -1
        # ("no prefix") indexes it and reaches the kernel as its no-op.
        self.pos = np.empty(V + 1, dtype=np.int64)
        self.pos[V] = -1
        self.arange = np.arange(max_candidates, dtype=np.int64)
        self.prefix_targets, self.target_pos, self.prefix_pos = np.empty((3, max_rows), dtype=np.int64)
        # The stride sweep, doubled so any window of it is one contiguous slice.
        sweep = (NEGATIVE_STRIDE * np.arange(V, dtype=np.int64)) % V
        self.sweep = np.concatenate((sweep, sweep))
        self.sweep_offset = 0

    def reset(self, rank: int, world_size: int):
        # Ranks start the sweep at staggered offsets, so their negatives differ.
        self.sweep_offset = (self.vocab_size * rank) // world_size

    def _negatives(self, need: int) -> np.ndarray:
        """`need` distinct unmarked classes: the next window of the stride sweep."""
        V, mark = self.vocab_size, self.mark
        draw = min(V, int(need * 1.7) + 256)
        idx = self.sweep[self.sweep_offset:self.sweep_offset + draw]
        self.sweep_offset = (self.sweep_offset + draw) % V
        idx = idx[~mark[idx]]
        if idx.size >= need:
            return idx[:need]
        free = np.flatnonzero(~mark[:V])
        assert free.size >= need, f"only {free.size} free classes for {need}"
        return free[::max(1, free.size // need)][:need]

    def build(self, P: int, targets: np.ndarray, prefix_table: np.ndarray | None):
        """(candidates, target_pos, prefix_pos) for one microbatch; views into scratch.

        prefix_pos is None when there is no prefix table yet (warmup), leaving the device buffer at -1.
        """
        V, mark, pos, T = self.vocab_size, self.mark, self.pos, targets.shape[0]
        mark.fill(False)
        mark[targets] = True
        mark[COPY_TOKEN_ID] = True  # the CPLM <copy> gate is always a candidate
        num_targets = int(np.count_nonzero(mark[:V]))
        assert num_targets <= P, f"{num_targets} always-included classes (targets and <copy>) > P={P}"
        if num_targets != P:
            self.mark_u8[self._negatives(P - num_targets)] = 1
        candidates = np.flatnonzero(mark[:V])  # ascending
        pos[:V].fill(-1)
        pos[candidates] = self.arange[:P]
        target_pos = self.target_pos[:T]
        np.take(pos, targets, out=target_pos)
        if prefix_table is None:
            return candidates, target_pos, None
        prefix_targets, prefix_pos = self.prefix_targets[:T], self.prefix_pos[:T]
        np.take(prefix_table, targets, out=prefix_targets)
        np.take(pos, prefix_targets, out=prefix_pos)
        return candidates, target_pos, prefix_pos



class SampledSoftmax(KS3SampledSoftmax):
    def __init__(self,*args,host_metadata=False,**kwargs):
        self.host_metadata=host_metadata
        super().__init__(*args,**kwargs)
        self.builder=CandidateBuilder(self.builder.vocab_size,max(self.counts),args[3])
        self.builder.reset(self.rank,self.world_size)

    def gather(self,step,lm_head_f8_col):
        if not self.host_metadata or not self.counts[step]:
            return super().gather(step,lm_head_f8_col)
        uploaded_step,staged=self.uploaded
        assert uploaded_step==step
        copy_column=host_copy_column(staged)
        inputs=super().gather(step,lm_head_f8_col)
        # The existing upload WAR/RAW events protect this shared index view
        # through the whole forward and backward, including index_select's
        # weight-gradient scatter. Only one microbatch is active per step.
        return SampledHeadInputs(inputs.rows,inputs.rows_t,inputs.target_pos,inputs.prefix_pos,
                                 inputs.vocab_pos,self.candidates[:self.counts[step]],copy_column)
