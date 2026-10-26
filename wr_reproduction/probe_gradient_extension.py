"""Optional KS3-only diagnostics on the unchanged V2 probe training loop."""
import argparse
import os
import time
from pathlib import Path
import torch
from wr_reproduction import probe_train
from wr_reproduction.ks3_update_observer import UpdateObserver
from wr_reproduction.probe_study import read, atomic, stopped, assert_fence


class ObservedRunner(probe_train.Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.ks3:
            raise ValueError('This extension is KS3-only; frozen MLP control stays unchanged')
        self.update_observer = UpdateObserver(self.model, self.tm, self.log)

    def log(self, **record):
        observer = getattr(self, 'update_observer', None)
        if observer is not None:
            if record.get('event') == 'probe':
                observer.arm(2)
            if 'cumulative_instrumentation_seconds' in record:
                record['frozen_v2_instrumentation_seconds'] = record['cumulative_instrumentation_seconds']
                record['gradient_extension_seconds'] = observer.total_seconds
                record['cumulative_instrumentation_seconds'] += observer.total_seconds
                record['probe_overhead_fraction'] = record['cumulative_instrumentation_seconds'] / record['train_seconds']
            record['gradient_extension'] = 'active-slot-pre-optimizer-v1'
        return super().log(**record)


def gate(r):
    """One120s diagnostic prefix to resolve missing gradients, never a loss trial."""
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_continuation import baseline_and_restore, compare
    from wr_reproduction.probe_precision_calibration import audit
    from wr_reproduction.probe_calibration import state
    from wr_reproduction.probe_train import Observations
    batches = r.loaders(list(range(r.schedule.total_steps)))
    r.model.limit_yarn_rebuild(r.max_tokens)
    torch.cuda.synchronize(); start = time.perf_counter()
    r.observations = Observations(r.research_run, r.log, lambda stamp: stamp - start, r.research_run.parent.parent)
    r.model.prefix_table.copy_(build_prefix_table_bucket(r.model.vocab_size, 0, 1))
    for sampler in r.samplers:
        sampler.set_prefix_table(r.model.prefix_table.cpu().numpy())
    r.log(event='gradient_diagnostic_gate_begin',prefix_budget_seconds=120.,not_a_convergence_run=True,
          reason='Correct unavailable post-optimizer gradients before selecting joint-rate hypotheses')
    try:
        for step in range(r.schedule.total_steps):
            r.observations.check()
            if time.perf_counter() - start >= 120. and step % 2 == 0 and not any(r.tables.pending) and not any(r.gradients.pending):
                r.observations.drain(); torch.cuda.synchronize(); break
            loss = r.step(step, batches, next_step=step + 1, prefetch=True)
            r.tail.tick(step); r.observations.submit(loss, step + 1, r.metadata.training(step))
        else:
            raise RuntimeError('No safe checkpoint in diagnostic prefix')
        elapsed = time.perf_counter() - start
        assert elapsed < 130.
        r.observations.close(); r.observations = None
        if stopped():
            raise RuntimeError('Explicit study STOP')
        r.tm.advance_schedule(step)
        reference = baseline_and_restore(r, batches, step)
        # Explicit state/storage/RNG audit of the new pre-optimizer sampling alone.
        before = state(r)
        captured = r.update_observer.capture(False, allow_missing=True)
        restoration = audit(before, state(r)); del captured
        r.update_observer.arm(2)
        loss = r.step(step, batches, next_step=step + 1, prefetch=True)
        torch.cuda.synchronize(); parity = compare(r, loss.item(), reference); del reference
        assert parity['all_parameters_bitwise_equal'], parity
        # Odd update applies geometry Adam and pending sparse/FP8 state normally.
        odd_loss = r.step(step + 1, batches, prefetch=False)
        torch.cuda.synchronize(); assert torch.isfinite(odd_loss).item()
        rows = r.update_observer.reports
        assert len(rows) == 2 and not rows[0]['do_adam'] and rows[1]['do_adam']
        assert all(v['gradient_available'] for v in rows[1]['measurements'].values())
        assert any(v['gradient_sample_rms'] > 0 for v in rows[1]['measurements'].values())
        result = dict(passed=True,mode='gradient-extension',prefix_elapsed_seconds=elapsed,step=step,
                      all_original_logical_ranks=8,full_graph_warmup=True,not_a_screening_reference=True,
                      full_convergence_verified=False,state_restoration=restoration,
                      continuation_parity={k:v for k,v in parity.items() if k!='parameters'},
                      odd_continuation_loss=odd_loss.item(),measurements=rows,
                      two_update_diagnostic_seconds=r.update_observer.total_seconds,
                      projected_seven_checkpoint_seconds=7*r.update_observer.total_seconds,
                      projected_fraction_of900=7*r.update_observer.total_seconds/900.)
        assert result['projected_fraction_of900'] < .003, result['projected_fraction_of900']
        atomic(r.research_run/'calibration.json',result)
        r.log(event='calibration_complete',**result)
    finally:
        if r.observations:
            r.observations.close(); r.observations=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args()
    cfg=read(a.run/'run.json')
    if any(Path(f).exists() for f in cfg['stop_files']):raise RuntimeError('Explicit study STOP')
    assert_fence(cfg['recipe'])
    from wr_reproduction.probe_controller import sources
    if sources()!=cfg['source_hashes']:raise RuntimeError('Source hashes changed before launch')
    if not a.gate:
        probe_train.Runner=ObservedRunner
        probe_train.main()
        return
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.set_num_threads(1)
    from wr_reproduction.local_collectives import LocalCollectives
    with LocalCollectives():
        r=ObservedRunner('pr393',a.run/'events.jsonl',True,cfg['recipe'].get('precision','bf16'),a.run)
        from wr_reproduction.matched_architecture import audit_model
        r.log(event='feedforward_inventory',**audit_model(r.model,cfg['recipe']['architecture']))
        r.warmup();gate(r)


if __name__=='__main__':main()
