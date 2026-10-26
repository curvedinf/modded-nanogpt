"""One bounded near-gate prefix per architecture; never a screening reference.

The original schedule and eight logical microbatches run continuously for120s.
Audits and the A/B replay are explicitly off that diagnostic prefix clock. Their
probe costs include preparation, scoring and restoration, as in the timed runs.
"""
import gc
import time
import torch
from wr_reproduction.probe_calibration import state,comparable,oracle
from wr_reproduction.probe_continuation import baseline_and_restore,compare
from wr_reproduction.probe_evaluation import FixedProbe
from wr_reproduction.probe_study import atomic,now,stopped,paired


def audit(before,after):
    changed=[k for k in set(before)|set(after) if before.get(k)!=after.get(k)]
    changed=[k for k in changed if not (k.startswith('rng/') or k=='sparse/table_sample')
             or comparable({k:before.get(k)})!=comparable({k:after.get(k)})]
    result=dict(checked_entries=len(before),changed=changed,passed=not changed,
                pointer_restoration_required=True,
                sparse_table_contract='readonly pointer/version plus1024 sampled rows; not a full130GB content hash')
    assert result['passed'],changed[:20]
    return result


def run(r):
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_train import Observations
    cfg=r.research_config;directory=r.research_run
    batches=r.loaders(list(range(r.schedule.total_steps)));r.model.limit_yarn_rebuild(r.max_tokens)
    gc.collect();torch.cuda.synchronize();start=time.perf_counter()
    r.observations=Observations(directory,r.log,lambda stamp:stamp-start,directory.parent.parent)
    r.model.prefix_table.copy_(build_prefix_table_bucket(r.model.vocab_size,0,1))
    for sampler in r.samplers:sampler.set_prefix_table(r.model.prefix_table.cpu().numpy())
    r.log(event='precision_calibration_begin',prefix_budget_seconds=120.,original_schedule=True,
          logical_ranks=8,comparative_pruning=False,not_a_convergence_run=True)
    try:
        for step in range(r.schedule.total_steps):
            r.observations.check()
            # Test an even update immediately after sparse/Adam pending work cleared.
            if (time.perf_counter()-start>=120. and step%2==0
                    and not any(r.tables.pending) and not any(r.gradients.pending)):
                r.observations.drain();torch.cuda.synchronize();break
            loss=r.step(step,batches,next_step=step+1,prefetch=True);r.tail.tick(step)
            r.observations.submit(loss,step+1,r.metadata.training(step))
        else:raise RuntimeError('No safe calibration checkpoint in bounded original prefix')
        checkpoint=time.perf_counter()-start
        assert checkpoint<130.,('Prefix cap unexpectedly exceeded',checkpoint)
        r.log(event='precision_checkpoint',step=step,checkpoint_elapsed_seconds=checkpoint,
              remaining_diagnostics_off_prefix_clock=True,full_validation=False)
        r.observations.close();r.observations=None
        if stopped():raise RuntimeError('Explicit study STOP')

        # The baseline uses the exact same upcoming data and restores graph-owned
        # storage, sampler state and pending FP8 updates before the probe branch.
        r.tm.advance_schedule(step)
        reference=baseline_and_restore(r,batches,step)
        probe=FixedProbe(r,cfg['probe_protocol'],cfg['backend_hash'])
        before=state(r);cold=probe.observe(step,time.perf_counter());after=state(r)
        restoration=audit(before,after)
        atomic(directory/'state-restoration.json',restoration)
        direct=oracle(probe)
        difference=max(abs(a-b) for a,b in zip(cold['chunk_losses'],direct))
        assert difference<1e-6,difference
        warm=probe.observe(step,time.perf_counter())
        repeat_difference=max(abs(a-b) for a,b in zip(cold['chunk_losses'],warm['chunk_losses']))
        assert repeat_difference<1e-6,repeat_difference
        # Acquire one bounded contingency sample at the same weights. If1M is
        # underpowered this avoids another120s training prefix just to read more
        # held-out text. It never enters a screening registry.
        larger_protocol=dict(cfg['probe_protocol'],tokens=2*cfg['probe_protocol']['tokens'])
        larger=FixedProbe(r,larger_protocol,cfg['backend_hash'])
        larger_row=larger.observe(step,time.perf_counter())
        assert larger_row['chunk_losses'][:len(cold['chunk_losses'])]==cold['chunk_losses']
        larger_restoration=audit(before,state(r))
        del larger
        loss=r.step(step,batches,next_step=step+1,prefetch=True);r.tail.tick(step)
        torch.cuda.synchronize();parity=compare(r,loss.item(),reference);del reference
        atomic(directory/'continuation-parity.json',parity)

        # Now dense and sparse gradients are pending for the next odd update.
        # Audit a second probe in this state, then actually replay that all-rank
        # update to exercise delayed refresh and both accumulation paths.
        assert any(r.tables.pending) and any(r.gradients.pending)
        pending_before=state(r);next_weights=probe.observe(step+1,time.perf_counter());pending_after=state(r)
        pending_restoration=audit(pending_before,pending_after)
        atomic(directory/'pending-state-restoration.json',pending_restoration)
        second_loss=r.step(step+1,batches,next_step=None,prefetch=True);r.tail.tick(step+1)
        torch.cuda.synchronize();assert torch.isfinite(second_loss).item()

        for name,row in (('cold',cold),('warm',warm),('contingency_2x',larger_row),('after_update',next_weights)):
            row.update(event='precision_calibration_probe',sample=name,
                       calibration_checkpoint_elapsed_seconds=checkpoint,
                       diagnostic_timing_only=True,not_a_screening_reference=True)
            r.log(**row)
        value=dict(utc=now(),passed=True,mode='precision',architecture=cfg['recipe']['architecture'],
                   step=step,prefix_elapsed_seconds=checkpoint,original_schedule=True,all_original_logical_ranks=8,
                   cold_probe_seconds=cold['probe_seconds'],warm_probe_seconds=warm['probe_seconds'],
                   pending_probe_seconds=next_weights['probe_seconds'],chunk_losses=cold['chunk_losses'],
                   probe_signature=cold['probe_signature'],signature_hash=cold['signature_hash'],
                   oracle_max_abs=difference,repeat_max_abs=repeat_difference,
                   deterministic_repetition_is_not_sampling_uncertainty=True,
                   state_restoration=restoration,pending_state_restoration=pending_restoration,
                   continuation_parity={k:v for k,v in parity.items() if k!='parameters'},
                   post_probe_odd_update_loss=second_loss.item(),full_convergence_verified=False,
                   paired_adjacent_update_diagnostic=paired(next_weights['chunk_losses'],cold['chunk_losses']),
                   cross_architecture_precision_requires_paired_calibration_report=True,
                   contingency=dict(tokens=larger_protocol['tokens'],chunk_losses=larger_row['chunk_losses'],
                                    probe_signature=larger_row['probe_signature'],signature_hash=larger_row['signature_hash'],
                                    probe_seconds=larger_row['probe_seconds'],state_restoration=larger_restoration,
                                    cold_cost_upper_estimate=cold['probe_seconds']+larger_row['probe_seconds'],
                                    cost_note='Already compiled by1M; conservative first-use bound adds the entire1M cold call. Not a measured2M cold cost.'),
                   timing='120s continuous original training prefix; state audits and A/B replay are separate diagnostic work')
        atomic(directory/'calibration.json',value);r.log(event='calibration_complete',**value)
    finally:
        if r.observations:r.observations.close();r.observations=None
