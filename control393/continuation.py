"""Open-ended KS3 training without restarting any training state at a schedule boundary.

The original MI300 control modules remain unchanged. These adapters keep their
prefetch/update ordering while removing finite training-step and history limits.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from itertools import count
import math

import torch

from mi300.data import ScheduledBatches
from mi300.ngram_table import MAX_CYCLE_STEPS, NgramTable
from mi300.perf.row_prefetch import RowPrefetch, PREP_SLACK_STEPS, cycle_wants
from control393.tail_average import TailAverages, BANK_BLEND_PERIOD, BANK_BLEND_RATE


class HoldLast(Sequence):
    """Finite schedule storage with a constant final value for all later steps."""
    def __init__(self, values):
        if not values:
            raise ValueError('Cannot continue an empty schedule')
        self.values = values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.values[index]
        return self.values[min(index, len(self.values) - 1)]


class StepNumbers:
    """Identity lookup for unbounded global steps/positions; no stored step list."""
    def __getitem__(self, step):
        if not isinstance(step, int) or step < 0:
            raise IndexError(step)
        return step


class ContinuingBatches(ScheduledBatches):
    def __init__(self, loader, schedule, start_step=0):
        super().__init__(loader, schedule, [])
        self.steps = StepNumbers()
        self.next_index = start_step
        stage, _ = schedule.lookup(start_step)
        self.geometry = (stage.batch_size, stage.train_max_seq_len)

    def peek(self, step):
        while step not in self.ahead:
            if self.next_index > step:
                raise AssertionError(f'step {step} is no longer ahead')
            self._fetch_next()
        return self.ahead[step]


class ContinuingPrefetch(RowPrefetch):
    """Same row-pull lifecycle, deriving only the current/next sparse cycles."""
    def __init__(self, ngram, value_embeds, batches, is_update, staging):
        self.ngram, self.value_embeds, self.batches = ngram, value_embeds, batches
        self.steps = self.position = StepNumbers()
        self.is_update, self.staging = is_update, staging
        self.prep_pending = self.next_pull = self.live_cycle = None

    def _cycle_from(self, first):
        for last in range(first, first + MAX_CYCLE_STEPS):
            if self.is_update(last):
                return list(range(first, last + 1))
        raise AssertionError('Sparse update cadence exceeds the cache capacity')

    def post_prep(self, step):
        if self.prep_pending is not None:
            return
        event = self._cycle_from(step + 1)[-1]
        if event - step <= PREP_SLACK_STEPS:
            cycle = self._cycle_from(event + 1)
            self.prep_pending = (event, self.staging.prep.submit(
                cycle_wants, self._batches(cycle), self.value_embeds.want_rows))


class ContinuingNgramTable(NgramTable):
    def ensure_history(self, event):
        if event >= self.beta2_history.numel():
            old = self.beta2_history
            grown = old.new_zeros(max(event + 1, old.numel() * 2))
            grown[:old.numel()].copy_(old)
            self.beta2_history = grown

    def adam_update(self, step, lr, weight_decay, eps):
        self.ensure_history(self.adam_events + 1)
        return super().adam_update(step, lr, weight_decay, eps)


class ContinuingTails(TailAverages):
    """Preserve the original tail through its horizon, then retain its EMA state.

    Beyond that point, regular tick cadences continue; the original special final
    two bank ticks must not accidentally turn into a tick on EVERY later step.
    Evaluations borrow the averaged weights and restore the raw iterate exactly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for acc in self.accumulators:
            if acc.tag == 'bank-blend':
                acc.rate = lambda s, n: BANK_BLEND_RATE if (
                    s % BANK_BLEND_PERIOD == 1 or n - 2 <= s < n) else None

    def ready(self):
        """Every shipped average must have observed real training weights."""
        return all(set(acc.labels).issubset(acc.seeded) for acc in self.accumulators)

    @contextmanager
    def evaluation_weights(self, enabled):
        if not enabled:
            yield 'raw'
            return
        if not self.ready():
            raise RuntimeError('Tail evaluation requested before every accumulator was seeded')
        labels = {label for acc in self.accumulators for label in acc.labels}
        saved = {label: self._param(label).detach().clone() for label in labels}
        try:
            self.ship()
            yield 'tail_average'
        finally:
            with torch.no_grad():
                for label, value in saved.items():
                    self._param(label).copy_(value)


@contextmanager
def canonical_validation(manager, step):
    """Temporarily use the control's final windows and full sequence of YaRN scales.

    Jumping directly from an early window to the final one would produce different
    frequencies. Apply every remaining stage transition in its original order,
    then restore the training frequencies, tables and forward configuration.
    """
    model, schedule = manager.model, manager.schedule
    names = ('ws_short', 'ws_long', 'train_max_seq_len', 'mtp_weights', 'prefix_weight')
    saved = {name: getattr(manager, name) for name in names}
    yarns = [(y, y.angular_freq.clone(), y.attn_scale, y.valid_rows) for y in model.yarns]
    was_training = model.training
    try:
        for start, _ in schedule.boundaries[1:]:
            if start > step:
                manager.advance_schedule(start)
        manager.apply_final_ws_ext()
        model.complete_yarn_tables()
        model.eval()
        yield
    finally:
        for name, value in saved.items():
            setattr(manager, name, value)
        with torch.no_grad():
            for yarn, freq, scale, valid_rows in yarns:
                yarn.angular_freq.copy_(freq)
                yarn.attn_scale = scale
                yarn._build_rows(0, valid_rows)
        model.train(was_training)


class TargetLoop:
    """An infinite trial loop with wall-time evaluation deadlines and full-CE stops.

    Deadlines use complete elapsed time, including evaluations. If an evaluation
    spans multiple deadlines, skip missed slots instead of evaluating back to back.
    """
    def __init__(self, target, interval_seconds, first_seconds=None):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError('A positive finite full-validation interval is required')
        self.target = target
        self.interval = interval_seconds
        self.first = interval_seconds if first_seconds is None else first_seconds
        if not math.isfinite(self.first) or self.first < 0:
            raise ValueError('Invalid first full-validation deadline')
        self.deadline = self.first
        self.first_hit = None

    def steps(self, start=0):
        for step in count(start):
            if self.first_hit is not None:
                return
            yield step

    def due(self, wall, update_complete):
        return update_complete and wall >= self.deadline

    def observe(self, metric):
        if metric.get('full_validation') is not True or metric.get('val_tokens') != 10485760:
            raise ValueError('Trial target detection requires the complete validation stream')
        if not math.isfinite(metric['val_loss']):
            raise RuntimeError('Nonfinite validation loss')
        wall = metric['wall_seconds']
        if not math.isfinite(wall) or wall < 0:
            raise ValueError('Invalid complete elapsed time')
        # Store the first measured target observation; never interpolate a hit.
        if metric['val_loss'] <= self.target and self.first_hit is None:
            self.first_hit = dict(metric)
        slots = max(1, math.floor((wall - self.first) / self.interval) + 1)
        self.deadline = self.first + slots * self.interval
        return self.first_hit is not None
