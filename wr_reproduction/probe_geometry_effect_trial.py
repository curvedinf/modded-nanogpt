"""One bounded mechanism diagnostic; never contributes screening references.

Train the unchanged incumbent for240s after full original graph warmup. At one
checkpoint, remove cubic and warp terms separately after coefficient packing.
All counterfactuals use the same weights/text. They are not retrained ablations.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import (ROOT, STATE, RESULTS, FENCE, PROTOCOL,
    STOP_FILES, read, atomic, append, now, stopped, assert_fence, environment,
    systemd_environment, publish, digest, paired)

KIND = 'same-checkpoint-spine-warp-after-coefficient-pack-v1'


def validate(value):
    assert_fence(value)
    baseline = control.recipe('ks3_experts')
    descriptive = {'name', 'hypothesis', 'hypothesis_class', 'geometry_effect_diagnostic'}
    if {k:v for k,v in value.items() if k not in descriptive} != {k:v for k,v in baseline.items() if k not in descriptive}:
        raise ValueError('Diagnostic training must use the unchanged incumbent recipe')
    if value.get('geometry_effect_diagnostic') != KIND:
        raise ValueError('Explicit evaluation-only diagnostic identifier required')


def create_run(value, unit):
    from wr_reproduction import research
    if stopped(): raise RuntimeError('Explicit study STOP')
    validate(value)
    dirty = control.git('status', '--porcelain', '--', 'wr_reproduction', 'configs/wr_reproduction', 'ks3_research/vendor')
    if dirty: raise RuntimeError('Commit source/config before diagnostic launch: ' + dirty)
    rid = value['name'] + '-calibration-' + now().replace('-', '').replace(':', '').split('.')[0] + '-' + uuid.uuid4().hex[:6]
    directory = STATE / 'runs' / rid; directory.mkdir(parents=True)
    head = control.git('rev-parse', 'HEAD'); fence = assert_fence(value)
    run = dict(study='probe-study', run_id=rid, directory=str(directory), seed=169,
        kind='calibration', recipe=value, unit=unit, created_utc=now(),
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_geometry_effect_trial', 'train', str(directory)],
        source_hashes=control.sources(), code_commit=head, launch_head_commit=head,
        environment=environment(), protocol_path=str(PROTOCOL), fence_path=str(FENCE),
        backend_hash=fence['backend_hash'], backend_fence=fence, probe_protocol=read(PROTOCOL),
        validation_data=research.validation_data(), continuous_from_initialization=False,
        comparative_pruning_exempt=False, stop_files=[str(p) for p in STOP_FILES],
        calibration='geometry-effect', not_a_screening_reference=True,
        prefix_budget_seconds=240., geometry_effect_diagnostic=KIND,
        scope='Diagnostic training is the unchanged original expert-only incumbent. Counterfactual evaluation and audits are separate from the bounded prefix clock. No terminal full validation or reserved text.')
    atomic(directory / 'run.json', run); atomic(directory / 'config.json', value)
    atomic(RESULTS / (rid + '-config.json'), run)
    append(STATE / 'registry.jsonl', dict(event='registered', run_id=rid, directory=str(directory),
        kind='calibration', seed=169, code_commit=head, not_a_screening_reference=True))
    control.status(run, 'registered'); control.note(run, hypothesis=value['hypothesis'],
        decision='Bounded same-checkpoint component diagnostic; no screening or qualification', code_commit=head)
    return run


def runner_class():
    from wr_reproduction.probe_gradient_extension import ObservedRunner
    from wr_reproduction.ks3_update_observer import UpdateObserver, specifications
    class ComponentObserver(UpdateObserver):
        def capture(self, do_adam, allow_missing=False):
            if self.specs is None:
                self.specs = specifications(self.model)
                item = next(x for x in self.specs if x['key'] == 'routed.coefficients')
                self.specs.remove(item); groups = item['indices'].reshape(-1, 6)
                for i, name in enumerate(('spine_c0', 'spine_c1', 'spine_c2', 'spine_c3', 'warp_c4', 'warp_c5')):
                    self.specs.append(dict(item, key='routed.'+name, component=name,
                        indices=groups[:, i], total_elements=item['total_elements']//6))
            return super().capture(do_adam, allow_missing)
    class DiagnosticRunner(ObservedRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tm.step_optimizers = self.update_observer.original
            self.update_observer = ComponentObserver(self.model, self.tm, self.log)
    return DiagnosticRunner


def diagnostic(r, numerical):
    import time
    import torch
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_train import Observations
    from wr_reproduction.probe_calibration import state
    from wr_reproduction.probe_precision_calibration import audit
    from wr_reproduction.probe_continuation import baseline_and_restore, compare
    from wr_reproduction.probe_evaluation import FixedProbe
    from wr_reproduction.ks3_geometry_effect import MODES, counterfactual, reports
    from wr_reproduction.matched_architecture import audit_model
    from wr_reproduction import ks3_moe as core
    cfg = r.research_config; m = r.model
    batches = r.loaders(list(range(r.schedule.total_steps))); m.limit_yarn_rebuild(r.max_tokens)
    torch.cuda.synchronize(); start = time.perf_counter()
    r.observations = Observations(r.research_run, r.log, lambda stamp:stamp-start, r.research_run.parent.parent)
    m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size, 0, 1))
    for sampler in r.samplers: sampler.set_prefix_table(m.prefix_table.cpu().numpy())
    r.log(event='geometry_effect_prefix_begin', prefix_budget_seconds=240., not_a_screening_reference=True,
        same_checkpoint_counterfactuals=True, unchanged_incumbent_training=True, logical_ranks=8)
    try:
        for step in range(r.schedule.total_steps):
            r.observations.check()
            if time.perf_counter()-start >= 240. and step%2 == 0 and not any(r.tables.pending) and not any(r.gradients.pending):
                r.observations.drain(); torch.cuda.synchronize(); break
            loss = r.step(step, batches, next_step=step+1, prefetch=True)
            r.tail.tick(step); r.observations.submit(loss, step+1, r.metadata.training(step))
        else: raise RuntimeError('No safe checkpoint in bounded geometry diagnostic')
        checkpoint = time.perf_counter()-start
        assert checkpoint < 250., checkpoint
        r.observations.close(); r.observations = None
        r.log(event='geometry_effect_checkpoint', step=step, prefix_elapsed_seconds=checkpoint,
            remaining_diagnostics_off_prefix_clock=True, not_a_screening_reference=True)
        assert core.EXECUTED_MOE_BACKWARDS == set(range(5))
        assert audit_model(m, 'ks3_experts')['active_conventional_mlp_instances'] == 5
        if stopped(): raise RuntimeError('Explicit study STOP')
        r.tm.advance_schedule(step)
        continuation = baseline_and_restore(r, batches, step)
        probe = FixedProbe(r, cfg['probe_protocol'], cfg['backend_hash'])
        before = state(r); observations = []; effects = []; elapsed_start = time.perf_counter()
        for mode in MODES:
            with counterfactual(r, mode, capture=mode=='full') as samples:
                row = probe.observe(step, time.perf_counter())
            if mode == 'full': effects = reports(samples)
            row.update(event='geometry_effect_counterfactual', mode=mode,
                measurement='diagnostic-same-checkpoint-counterfactual-cplm-ce',
                base_text_signature_hash=row['signature_hash'],
                signature_hash=digest(dict(base=row['signature_hash'], diagnostic=KIND, mode=mode)),
                prefix_elapsed_seconds=checkpoint, not_a_screening_reference=True,
                coefficient_scale_preserved=True, terminal_full_validation=False)
            if observations:
                row['paired_vs_full'] = paired(row['chunk_losses'], observations[0]['chunk_losses'])
                coarse = lambda xs:[sum(xs[i:i+4])/4 for i in range(0, len(xs), 4)]
                row['paired_32768_vs_full'] = paired(coarse(row['chunk_losses']), coarse(observations[0]['chunk_losses']))
            observations.append(row); r.log(**row)
        diagnostic_seconds = time.perf_counter()-elapsed_start
        restoration = audit(before, state(r))
        r.update_observer.arm(2)
        loss = r.step(step, batches, next_step=step+1, prefetch=True); r.tail.tick(step)
        torch.cuda.synchronize(); parity = compare(r, loss.item(), continuation); del continuation
        assert parity['all_parameters_bitwise_equal'], parity['maximum_parameter_relative_rms']
        odd = r.step(step+1, batches, prefetch=False); r.tail.tick(step+1)
        torch.cuda.synchronize(); assert torch.isfinite(odd).item()
        gradient_rows = r.update_observer.reports
        adam = next(x for x in gradient_rows if x['do_adam'])
        assert all(v['gradient_available'] for v in adam['measurements'].values())
        result = dict(passed=True, diagnostic=KIND, prefix_elapsed_seconds=checkpoint, step=step,
            numerical_producer_gate=numerical, full_graph_warmup=True, original_logical_ranks=8,
            routed_backward_sites=sorted(core.EXECUTED_MOE_BACKWARDS), five_original_dense_MLPs=True,
            same_checkpoint_counterfactuals=observations, local_same_coordinate_effects=effects,
            state_restoration=restoration, continuation_parity={k:v for k,v in parity.items() if k!='parameters'},
            next_adam_loss=odd.item(), component_gradient_updates=gradient_rows,
            diagnostic_scoring_and_local_effect_seconds=diagnostic_seconds,
            not_a_screening_reference=True, full_convergence_verified=False, reserved_text_used=False,
            limitations=['Component removal is after original six-coefficient quantization, before FP16 nonlinear evaluation; no surviving coefficient requantization.',
                'GS32 activation scales change under component removal. Local deltas are measured on identical coordinates; full-model CE allows downstream activations and routing to respond.',
                'Unpacked diagnostics still include original FP16 nonlinearity and BF16 storage. They are not ideal continuous FP32 geometry.',
                'Local output moments precede post/router scaling. Full-model counterfactual CE includes the complete accepted evaluation math.',
                'Local moments use4096 deterministic feature/output samples per site, excluding padding, on the first fixed-text chunk.',
                'Diagnostic-only240s training prefix followed by separate scoring/state audits; never a900s endpoint, retrained ablation or reusable screening curve.'])
        atomic(r.research_run/'calibration.json', result); r.log(event='calibration_complete', **result)
    finally:
        if r.observations: r.observations.close(); r.observations = None


def train(directory):
    import torch
    cfg = read(directory/'run.json'); validate(cfg['recipe'])
    if cfg.get('kind') != 'calibration' or not cfg.get('not_a_screening_reference'):
        raise RuntimeError('Geometry effect runner is diagnostic-only')
    if stopped(): raise RuntimeError('Explicit study STOP')
    if control.sources() != cfg['source_hashes']: raise RuntimeError('Sources changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS', 'MOE_GRAD_FP8', 'MOE_DW_PROMOTE', 'MOE_DX_PROMOTE'):
        if key in os.environ: raise RuntimeError('Forbidden inherited override: ' + key)
    os.environ['TRAIN_SEED'] = str(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(1)
    from wr_reproduction.ks3_geometry_effect import numerical_check
    numerical = numerical_check()
    atomic(directory/'numerical-gate.json', numerical)
    append(directory/'events.jsonl', dict(event='geometry_effect_numerical_gate', **numerical))
    from wr_reproduction.local_collectives import LocalCollectives
    with LocalCollectives():
        r = runner_class()('pr393', directory/'events.jsonl', True, 'bf16', directory)
        r.warmup(); diagnostic(r, numerical)


def work(request):
    cfg = read(request)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stopped(): return
        run = create_run(cfg['recipe'], cfg['unit']); result = control.execute(run)
        value = dict(utc=now(), run_id=run['run_id'], code_commit=run['code_commit'],
            passed=result['status']=='calibration_complete' and bool((result.get('calibration') or {}).get('passed')),
            result=result, not_a_screening_reference=True)
        for root in (STATE, RESULTS): atomic(root/'geometry-effect-diagnostic.json', value)


def launch(path):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value = read(path); validate(value)
    unit = 'wr-probe-geometry-effect-' + uuid.uuid4().hex[:8]
    request = STATE/'requests'/(unit+'.json')
    atomic(request, dict(utc=now(), unit=unit, recipe=value))
    subprocess.run(['systemd-run', '--unit='+unit, '--working-directory='+str(ROOT),
        '--property=KillMode=control-group', '--property=TimeoutStopSec=15',
        '--property=PartOf=wr-ks3-probe-research.service', *systemd_environment(),
        str(control.PYTHON), '-m', 'wr_reproduction.probe_geometry_effect_trial', 'work', str(request)], check=True)
    print(unit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'work', 'train']); p.add_argument('path', type=Path)
    args = p.parse_args()
    try: globals()[args.action](args.path)
    except BaseException:
        error = dict(utc=now(), error=traceback.format_exc())
        atomic(STATE/'geometry-effect-blocker.json', error); publish('directed-status.json', dict(status='error', **error)); raise
