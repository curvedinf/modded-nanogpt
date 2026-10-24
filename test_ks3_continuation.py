"""CPU continuation regressions; GPU integration is the registry's --kind smoke.

These tests run without touching the measured GPU job. They exercise the real
prefetch protocol with a CPU transfer backend, not a replacement training loop.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ks3_research.continuation import (ContinuingBatches, ContinuingPrefetch,
    ContinuingNgramTable, ContinuingTails, HoldLast, TargetLoop, canonical_validation)
from ks3_research.launch import compare
from ks3_research.optim import ContinuingManager
from ks3_research.planning import load_config, schedule_for
from mi300.data import ScheduledBatches
from mi300.model.attention import Yarn
from mi300.ngram_table import is_update_step, sorted_unique_rows
from mi300.perf.row_prefetch import RowPrefetch
from mi300.sampled_softmax import CandidateBuilder, candidate_count_at
from mi300.tail_average import TailAverages
from mi300.training import TrainingManager


class FakeRows:
    """Deterministic CPU row exchange/update backend for the real prefetch code."""
    def __init__(self):
        self.live = self.cached = self.grads_in_flight = self.landing = None
        self.pending = []
        self.weights = np.zeros(32)
        self.events = self.restores = 0

    def request(self, rows, host):
        return NS(want=np.asarray(rows))

    def exchange_ids(self, pull):
        pass

    def fill(self, rows):
        self.live = self.cached = self.request(rows, rows)
        return self.live

    def start_cycle(self, pull):
        assert self.cached is pull and not self.pending
        self.live = pull

    def restore_cache(self):
        self.restores += 1
        self.cached = self.live

    def slots(self, pull, rows):
        assert self.cached is self.live
        slots = np.searchsorted(pull.want, rows)
        np.testing.assert_array_equal(pull.want[slots], rows)
        return slots

    def start_land(self, pull):
        assert self.landing is None
        self.landing = pull

    def complete_land(self):
        if self.landing is not None:
            self.cached = self.landing
            self.landing = None

    def bring_current(self, pull):
        pass

    def send_grads(self):
        self.grads_in_flight = self.pending
        self.pending = []

    def zero_sent_grads(self):
        pass

    def adam_update(self, *args):
        for rows in self.grads_in_flight:
            np.add.at(self.weights, rows, 1)
        self.grads_in_flight = None
        self.events += 1

    def want_rows(self, arrays):
        return sorted_unique_rows(arrays)


class FakeValues(FakeRows):
    def complete_land(self):
        super().complete_land()
        self.live = self.cached


def batch_stream(schedule):
    stage, _ = schedule.lookup(0)
    geometry = stage.batch_size, stage.train_max_seq_len
    offset = 0
    while True:
        tokens = np.arange(offset, offset+geometry[0], dtype=np.int32)
        rows = tokens % 32
        offset += geometry[0]
        params = yield NS(tokens=tokens, inputs_cpu=rows, ngram_ids_cpu=rows, ngram_ids=rows)
        if params is not None:
            geometry = params


def run_prefetch(open_ended):
    config = load_config('configs/ks3/primary.json')
    schedule = schedule_for(config, 648, torch.device('cpu'))
    for stage, size in zip(schedule.stages, (1, 2, 3, 2, 1)):
        stage.batch_size = size
    last = schedule.total_steps + 12
    steps = list(range(last + 12))
    batches = (ContinuingBatches(batch_stream(schedule), schedule) if open_ended else
               ScheduledBatches(batch_stream(schedule), schedule, steps))
    table, values = FakeRows(), FakeValues()
    sampled = CandidateBuilder(1009, 64, 3)
    sampled.reset(0, 1)
    all_tokens, sample_sweeps, event_steps = [], [], []
    with ThreadPoolExecutor(max_workers=1) as prep:
        staging = NS(prep=prep, ngram_ring=NS(upload=lambda x:x), value_embed_ring=NS(upload=lambda x:x))
        prefetch = (ContinuingPrefetch(table, values, batches, is_update_step, staging) if open_ended else
                    RowPrefetch(table, values, batches, steps, is_update_step, staging))
        for step in range(last):
            batch = batches.peek(step)
            slots = prefetch.forward_slots(step, batch.ngram_ids)
            values.complete_land()
            sampled.build(64, batch.tokens % 1009, None)
            sample_sweeps.append(sampled.sweep_offset)
            prefetch.after_forward(step)
            table.pending.append(table.live.want[slots])
            values.pending.append(batch.inputs_cpu)
            prefetch.post_prep(step)
            event = prefetch.optimizer_event(step, 0.1, 0., 1e-8)
            if event:
                event.launch()
                event.update()
                event.finish()
                event_steps.append(step)
            values.complete_land()
            self_batch = batches.take(step)
            assert self_batch is batch
            all_tokens.extend(batch.tokens.tolist())
            # Repeated evaluations overwrite only the cache. An already prepared
            # cycle and all input/candidate state must survive the interruption.
            if step in (3, 339, schedule.total_steps-1, schedule.total_steps+3):
                table.cached = object()
            assert len(batches.ahead) <= 8
        prefetch.close()
        assert not table.pending and table.events == len(event_steps)
    return all_tokens, sample_sweeps, event_steps, table.weights, values.weights, table.restores


def fake_optimizer():
    labels = ('lm_head', 'embed', 'vo_bank', 'mlp_bank', 'qk_bank', 'value_embeds')
    params = {label: torch.nn.Parameter(torch.arange(12.).reshape(4, 3)+i+1)
              for i, label in enumerate(labels)}
    configs = {p:NS(optim='anvil' if 'bank' in l else 'adam', reshape=(4, 3), chunk_size=4)
               for l,p in params.items()}
    return NS(_param_by_label=params, param_cfgs=configs)


def cpu_lerp(out, value, rate):
    out.lerp_(value.float(), rate)


class ContinuationTests(unittest.TestCase):
    def test_target_loop_crosses_horizon_and_control_time_then_stops_on_exact_target(self):
        target = 3.793633
        loop = TargetLoop(target, 7.)
        wall, observations, trained = 0., [], []
        for step in loop.steps():
            trained.append(step)
            wall += 1.
            if loop.due(wall, is_update_step(step)):
                wall += 3.  # Actual evaluation and restoration time.
                metric = dict(full_validation=True, val_tokens=10485760, wall_seconds=wall,
                              val_loss=(4.3, 4.0, target)[len(observations)])
                observations.append(metric)
                loop.observe(metric)
        self.assertGreater(trained[-1], 8)  # Initial planned horizon, not a stop.
        self.assertGreater(loop.first_hit['wall_seconds'], 10.)  # Control duration, not a stop.
        self.assertEqual(len(observations), 3)
        self.assertEqual(loop.first_hit, observations[-1])
        self.assertFalse(loop.due(wall, False))
        loop.observe(dict(observations[-1], wall_seconds=100, val_loss=3.7))
        self.assertEqual(loop.first_hit, observations[-1])
        self.assertEqual(loop.deadline, 105.)

    def test_partial_or_nonfinite_validation_cannot_hit_target(self):
        loop = TargetLoop(3.793633, 600.)
        metric = dict(full_validation=True, val_tokens=10485760, val_loss=3., wall_seconds=601.)
        for bad in (dict(metric, full_validation=False), dict(metric, val_tokens=1024)):
            with self.assertRaises(ValueError):
                loop.observe(bad)
        with self.assertRaises(RuntimeError):
            loop.observe(dict(metric, val_loss=float('nan')))
        self.assertIsNone(loop.first_hit)
        self.assertFalse(loop.observe(dict(metric, val_loss=3.793634)))

    def test_faster_result_is_independent_of_any_old_planning_budget(self):
        config = dict(control_val_loss=3.793633, control_wall_seconds=100., plan=dict(wall_budget_seconds=1.))
        metric = dict(full_validation=True, validation_tokens=10485760, val_loss=3.793633)
        self.assertTrue(compare(metric, config, 0, 99.)['exploratory_win'])
        slower = compare(metric, config, 0, 101.)
        self.assertTrue(slower['target_reached'])
        self.assertFalse(slower['exploratory_win'])
        self.assertFalse(compare(metric, config, 1, 99.)['target_reached'])

    def test_open_prefetch_matches_finite_reference_across_boundary_and_evaluations(self):
        actual, expected = run_prefetch(True), run_prefetch(False)
        for a,b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(actual[0], list(range(len(actual[0]))))
        self.assertEqual(actual[-1], 4)
        self.assertGreater(actual[2][-1], 720)

    def test_far_future_schedule_uses_final_objective_and_positive_lr(self):
        schedule = schedule_for(load_config('configs/ks3/primary.json'), 648, torch.device('cpu'))
        counts = HoldLast([candidate_count_at(schedule, s) for s in range(schedule.total_steps+1)])
        for step in (schedule.total_steps, schedule.total_steps+1, 10**12):
            self.assertIs(schedule.lookup(step)[0], schedule.stages[-1])
            self.assertEqual(schedule.mtp_weights[step].tolist(), [1.])
            self.assertEqual(schedule.prefix_weights[step].item(), 0.)
            self.assertEqual(counts[step], 0)
            self.assertEqual(schedule.get_lr(step), .3)
        self.assertEqual(len(list(counts)), schedule.total_steps+1)

    def test_sparse_history_grows_without_resetting_moments_or_event_numbers(self):
        table = ContinuingNgramTable.__new__(ContinuingNgramTable)
        table.beta2_history = torch.tensor([0., .95, .95**2])
        table.exp_avg_sq, table.last_event, table.adam_events = torch.tensor([.8]), torch.tensor([2]), 2
        old_moment, old_last = table.exp_avg_sq, table.last_event
        for event in range(3, 100):
            table.ensure_history(event)
            table.beta2_history[event] = .95**2
        torch.testing.assert_close(table.beta2_history[:3], torch.tensor([0., .95, .95**2]), rtol=0, atol=0)
        self.assertIs(table.exp_avg_sq, old_moment)
        self.assertIs(table.last_event, old_last)
        self.assertEqual(table.adam_events, 2)
        self.assertEqual(table.beta2_history[99], torch.tensor(.95**2))

    def test_optimizer_preserves_original_updates_and_clamps_only_rail_schedule(self):
        schedule = schedule_for(load_config('configs/ks3/primary.json'), 648, torch.device('cpu'))
        def manager(cls):
            result = cls.__new__(cls)
            result.schedule, result.split_step = schedule, schedule.split_step
            result.adam_defaults = dict(lr=.008, weight_decay=.005, eps=1e-10)
            cfgs = {i:NS(label=l, optim=o, initial_lr=.1, lr=0., momentum=0.)
                    for i,(l,o) in enumerate((('mlp_bank','anvil'),('value_embeds','adam'),('embed','adam')))}
            calls = []
            result.optimizer = NS(param_cfgs=cfgs,
                set_rails=lambda **kw:calls.append(('rails', kw)),
                step=lambda **kw:calls.append(('step', kw)),
                copy_lm_state_to_embed=lambda:calls.append(('split',)))
            pulls = NS(optimizer_event=lambda s, **kw: (s, kw))
            return result, pulls, calls
        original, p1, c1 = manager(TrainingManager)
        actual, p2, c2 = manager(ContinuingManager)
        for step in range(schedule.total_steps):
            original.step_optimizers(step, p1, frozenset())
            actual.step_optimizers(step, p2, frozenset())
            self.assertEqual(vars(original.optimizer.param_cfgs[0]), vars(actual.optimizer.param_cfgs[0]))
        self.assertEqual(c1, c2)
        self.assertEqual(c2.count(('split',)), 1)
        for step in (schedule.total_steps, schedule.total_steps+3, 10**9+3):
            actual.step_optimizers(step, p2, frozenset())
            self.assertEqual(actual.optimizer.param_cfgs[0].momentum, .85)
            self.assertEqual(c2[-1][1]['sparse_update'][0], step)
        self.assertEqual(c2.count(('split',)), 1)

    def test_stage_lr_scales_apply_to_dense_and_sparse_without_reset_or_terminal_decay(self):
        schedule = schedule_for(load_config('configs/ks3/primary.json'),648,torch.device('cpu'))
        manager = ContinuingManager.__new__(ContinuingManager)
        manager.schedule,manager.split_step = schedule,schedule.split_step
        manager.lr_stage_scales=(.9,.9,1.,1.,1.)
        manager.adam_defaults=dict(lr=.008,weight_decay=.005,eps=1e-10)
        cfgs={i:NS(label=label,optim=optim,initial_lr=lr,lr=0.) for i,(label,optim,lr) in enumerate(
            (('mlp_bank','anvil',.023),('value_embeds','adam',.56),('ks3_scale','adam',.001)))}
        calls=[]
        persistent_moment=torch.tensor([.3,.7])
        manager.optimizer=NS(param_cfgs=cfgs,moment=persistent_moment,
            set_rails=lambda **kw:None,step=lambda **kw:calls.append(kw),
            copy_lm_state_to_embed=lambda:self.fail('Unexpected split/reset'))
        pulls=NS(optimizer_event=lambda step,**kw:(step,kw))
        transition=schedule.boundaries[2][0]
        for step,scale in ((0,.9),(transition-1,.9),(transition,1.),(schedule.total_steps,1.),(10**9,1.)):
            manager.step_optimizers(step,pulls,frozenset())
            multiplier=schedule.get_lr(step)*scale
            for cfg in cfgs.values():
                self.assertAlmostEqual(cfg.lr,cfg.initial_lr*multiplier)
            sparse_step,sparse=calls[-1]['sparse_update']
            self.assertEqual(sparse_step,step)
            self.assertAlmostEqual(sparse['lr'],.008*multiplier)
            self.assertEqual(sparse['weight_decay'],.005)
            self.assertIs(manager.optimizer.moment,persistent_moment)
            torch.testing.assert_close(persistent_moment,torch.tensor([.3,.7]),atol=0,rtol=0)
        self.assertEqual(manager.effective_lr_multiplier(10**9),.3)

    def test_tail_weights_restore_exactly_and_continue_regular_ema_ticks(self):
        with patch('mi300.tail_average.lerp_upcast_', cpu_lerp), patch(
                'mi300.tail_average.dist.all_gather_into_tensor', lambda dst,src:dst.copy_(src.clone())):
            old_opt, new_opt = fake_optimizer(), fake_optimizer()
            old = TailAverages(old_opt, 0, is_update_step, 720)
            new = ContinuingTails(new_opt, 0, is_update_step, 720)
            with torch.no_grad():
                for step in range(720):
                    for opt in (old_opt, new_opt):
                        for value in opt._param_by_label.values():
                            value.add_(.001 * ((step % 7) - 3))
                    old.tick(step)
                    new.tick(step)
            for a,b in zip(old.accumulators, new.accumulators):
                for label in a.labels:
                    torch.testing.assert_close(a.bufs[label], b.bufs[label], rtol=0, atol=0)
            raw = {l:p.detach().clone() for l,p in new_opt._param_by_label.items()}
            old.ship()
            for failure in (False, True):
                try:
                    with new.evaluation_weights(True):
                        for l,p in new_opt._param_by_label.items():
                            torch.testing.assert_close(p, old_opt._param_by_label[l], rtol=0, atol=0)
                        if failure:
                            raise RuntimeError('synthetic validation failure')
                except RuntimeError:
                    self.assertTrue(failure)
                for l,p in new_opt._param_by_label.items():
                    torch.testing.assert_close(p, raw[l], rtol=0, atol=0)
            bank = next(a for a in new.accumulators if a.tag == 'bank-blend')
            self.assertEqual([s for s in range(720, 732) if bank.rate(s, 720) is not None], [721,725,729])
            before = bank.bufs['mlp_bank'].clone()
            with torch.no_grad():
                new_opt._param_by_label['mlp_bank'].add_(1.)
            for step in range(720, 732):
                new.tick(step)
            self.assertFalse(torch.equal(bank.bufs['mlp_bank'], before))

    def test_early_validation_uses_canonical_final_yarn_and_restores_training(self):
        config = load_config('configs/ks3/primary.json')
        schedule = schedule_for(config, 648, torch.device('cpu'))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.yarn = Yarn(64, 8, attn_scale=.12, device='cpu')
                self.yarn_paired_head = Yarn(64, 8, paired=True, attn_scale=.12, device='cpu')
                self.yarn_wide = Yarn(128, 8, attn_scale=.12, device='cpu')
            @property
            def yarns(self):
                return self.yarn, self.yarn_paired_head, self.yarn_wide
            def complete_yarn_tables(self):
                for y in self.yarns:
                    y.ensure_full()
        def create():
            m = ContinuingManager.__new__(ContinuingManager)
            m.model, m.schedule = Model(), schedule
            m.ws_short, m.ws_long = schedule.stages[0].window_sizes
            m.advance_schedule(0)
            return m
        reference = create()
        for start,_ in schedule.boundaries[1:]:
            reference.advance_schedule(start)
        reference.apply_final_ws_ext()
        for at in (0, schedule.boundaries[2][0], schedule.total_steps+4):
            m = create()
            for start,_ in schedule.boundaries[1:]:
                if start <= at:
                    m.advance_schedule(start)
            m.advance_schedule(at)
            state = [(y.angular_freq.clone(), y.factor1.clone(), y.factor2.clone(), y.attn_scale) for y in m.model.yarns]
            original = (m.ws_short, m.ws_long, m.train_max_seq_len, m.mtp_weights, m.prefix_weight)
            with canonical_validation(m, at):
                self.assertEqual((m.ws_short, m.ws_long), (6, 20))
                for a,b in zip(m.model.yarns, reference.model.yarns):
                    torch.testing.assert_close(a.angular_freq, b.angular_freq, atol=0, rtol=0)
                    torch.testing.assert_close(a.factor1, b.factor1, atol=0, rtol=0)
                    self.assertEqual(a.attn_scale, b.attn_scale)
            self.assertEqual((m.ws_short, m.ws_long, m.train_max_seq_len, m.mtp_weights, m.prefix_weight), original)
            for y,(freq,f1,f2,scale) in zip(m.model.yarns, state):
                for a,b in ((y.angular_freq,freq), (y.factor1,f1), (y.factor2,f2)):
                    torch.testing.assert_close(a,b,atol=0,rtol=0)
                self.assertEqual(y.attn_scale, scale)
            self.assertTrue(m.model.training)

if __name__ == '__main__':
    unittest.main(verbosity=2)
