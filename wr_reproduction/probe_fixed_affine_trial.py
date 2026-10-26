"""Train all six original rank-one coefficients with only warp scale/shift fixed.

Only the reduced KS3 geometry gradient is masked, immediately before its
original Adam update. Forward/backward kernels, projection policy, all five
original dense MLPs and the frozen probe protocol remain unchanged.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import (ROOT, STATE, RESULTS, PROTOCOL, FENCE,
    STOP_FILES, read, atomic, append, now, stopped, assert_fence, environment,
    systemd_environment, publish)

KIND = 'routed-warp-affine-fixed-scale-shift-v1'


def validate(value):
    assert_fence(value)
    baseline = control.recipe('ks3_experts')
    descriptions = {'name', 'hypothesis', 'hypothesis_class', 'fixed_affine', 'mechanism_evidence'}
    if {k:v for k,v in value.items() if k not in descriptions} != {k:v for k,v in baseline.items() if k not in descriptions}:
        raise ValueError('Only the explicit routed affine-only freeze may differ from the incumbent')
    if value.get('fixed_affine') != KIND:
        raise ValueError('Explicit fixed-affine control required')


def mask_affine_gradient_(gradient):
    if gradient is None or gradient.shape[-1] != 8:
        raise ValueError('Expected reduced rank-one geometry gradient')
    gradient[..., 6:].zero_()
    return gradient


def cpu_checks():
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2969)
        p = torch.nn.Parameter(torch.zeros(2, 3, 8)); p.data[..., 6] = .75
        active = torch.nn.Parameter(p[..., :6].detach().clone())
        unrelated = torch.nn.Parameter(torch.randn(2, 3))
        unrelated_ref = torch.nn.Parameter(unrelated.detach().clone())
        initial = p[..., 6:].detach().clone()
        opt = torch.optim.Adam([p, unrelated], lr=.001, betas=(.9, .99), weight_decay=0, foreach=False)
        ref = torch.optim.Adam([active, unrelated_ref], lr=.001, betas=(.9, .99), weight_decay=0, foreach=False)
        for step in range(20):
            gradient = torch.randn_like(p); original = gradient.clone(); pointer = gradient.data_ptr()
            assert mask_affine_gradient_(gradient) is gradient
            assert gradient.data_ptr() == pointer and torch.equal(gradient[..., :6], original[..., :6])
            assert torch.count_nonzero(gradient[..., 6:]) == 0
            p.grad = gradient; active.grad = original[..., :6].clone()
            unrelated.grad = torch.randn_like(unrelated); unrelated_ref.grad = unrelated.grad.clone()
            opt.step(); ref.step()
            assert torch.equal(p[..., :6], active) and torch.equal(p[..., 6:], initial)
            assert torch.equal(unrelated, unrelated_ref)
            for name in ('exp_avg', 'exp_avg_sq'):
                assert torch.equal(opt.state[p][name][..., :6], ref.state[active][name])
                assert torch.count_nonzero(opt.state[p][name][..., 6:]) == 0
        c=torch.tensor([.3,.8,1.7],dtype=torch.float64)
        def affine_signal(coefficients):
            raw=torch.tensor([.75,0.],dtype=torch.float64,requires_grad=True)
            t=4*(c/4).tanh();scale=torch.nn.functional.softplus(raw[0])+1e-4
            z=(scale*(t+.5*raw[1].tanh())).tanh()
            c4,c5=coefficients
            warped=((4*c5*z+2*c4)*z-3*c5)*z-c4
            return torch.autograd.grad(warped.sum(),raw)[0]
        absent=affine_signal([0.,0.]);present=affine_signal([.05,-.03])
        assert torch.equal(absent,torch.zeros_like(absent)) and (present.abs()>0).all()
        for value in (None, torch.zeros(2, 7)):
            try: mask_affine_gradient_(value)
            except ValueError: pass
            else: raise AssertionError('Invalid geometry gradient accepted')
    return dict(passed=True, independent_adam_steps=20, all_six_coefficient_parameters_and_moments_bitwise_equal=True,
        affine_parameters_bitwise_initial=True, affine_adam_moments_zero=True,
        gradient_storage_preserved=True, nongeometry_parameters_untouched=True,
        discarded_c4_c5_freeze_implies_zero_affine_gradients=True,
        retained_nonzero_warp_produces_affine_gradients=present.tolist())


def runner_class():
    import torch
    from wr_reproduction.probe_geometry_effect_trial import runner_class as diagnostic_class
    class FixedAffineRunner(diagnostic_class()):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            p = self.model.ks3_geometry; cfg = self.tm.optimizer.param_cfgs[p]
            assert p.shape == (5, 8, 1024, 8) and p.dtype == torch.float32
            assert cfg.optim == 'adam' and cfg.lr_mul == .125 and cfg.wd_mul == 0
            assert list(cfg.adam_betas) == [.9, .99]
            assert torch.count_nonzero(p[..., :6]) == 0
            assert torch.all(p[..., 6] == .75) and torch.count_nonzero(p[..., 7]) == 0
            original = self.update_observer.original
            def step(step, *args, **kwargs):
                # Runner has reduced the original eight logical ranks already.
                # Keep non-Adam rank-local accumulation and its cadence intact.
                if self.tm.is_adam_step(step): mask_affine_gradient_(p.grad)
                return original(step, *args, **kwargs)
            self.update_observer.original = step
            self.log(event='fixed_affine_architecture', trainable_geometry_columns=[0, 1, 2, 3, 4, 5],
                frozen_geometry_columns=[6, 7], frozen_values=[.75, 0.],
                initial_function_unchanged=True, independent_rank_one_cubic_and_warp_paths=True,
                forward_backward_kernels_unchanged=True, original_projection_rates=True,
                five_original_dense_MLPs=True, no_added_graph_buffers=True)
        def log(self, **record):
            record['fixed_affine'] = KIND
            if record.get('event') == 'ks3_gradient_update':
                record['gradient_measurement_before_freeze_mask'] = True
                record['frozen_geometry_columns'] = [6, 7]
            return super().log(**record)
    return FixedAffineRunner


def frozen_state(r):
    import torch
    p = r.model.ks3_geometry; state = r.tm.optimizer.param_states[p]
    expected = torch.tensor([.75, 0.], device=p.device).expand_as(p[..., 6:])
    value = dict(frozen_parameter_values=p[..., 6:].numel(),
        all_frozen_parameters_bitwise_initial=torch.equal(p[..., 6:], expected),
        all_frozen_first_moments_zero=bool(torch.count_nonzero(state['exp_avg'][..., 6:]) == 0),
        all_frozen_second_moments_zero=bool(torch.count_nonzero(state['exp_avg_sq'][..., 6:]) == 0),
        all_six_coefficients_learned=bool((p[..., :6].abs().amax(dim=(0,1,2)) > 0).all()),
        warp_coefficients_learned_each_routed_site=bool((p[..., 4:6].abs().amax(dim=(1,2)) > 0).all()),
        all_six_first_moments_nonzero=bool((state['exp_avg'][..., :6].abs().amax(dim=(0,1,2)) > 0).all()))
    assert all(value[k] for k in value if k != 'frozen_parameter_values'), value
    return value


def graph_gate(r):
    import time
    import torch
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_continuation import baseline_and_restore, compare
    from wr_reproduction.probe_calibration import state, oracle
    from wr_reproduction.probe_precision_calibration import audit
    from wr_reproduction.probe_evaluation import FixedProbe
    from wr_reproduction.matched_architecture import audit_model
    from wr_reproduction import ks3_moe as core
    m = r.model; cfg = r.research_config; m.limit_yarn_rebuild(r.max_tokens)
    m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size, 0, 1))
    for sampler in r.samplers: sampler.set_prefix_table(m.prefix_table.cpu().numpy())
    batches = r.loaders(list(range(12)))
    for step in range(8):
        if stopped(): raise RuntimeError('Explicit study STOP')
        loss = r.step(step, batches, next_step=step+1, prefetch=True); r.tail.tick(step)
        torch.cuda.synchronize(); assert torch.isfinite(loss).item()
    freeze = frozen_state(r)
    assert core.EXECUTED_MOE_BACKWARDS == set(range(5))
    assert audit_model(m, 'ks3_experts')['active_conventional_mlp_instances'] == 5
    r.tm.advance_schedule(8); baseline = baseline_and_restore(r, batches, 8)
    probe = FixedProbe(r, cfg['probe_protocol'], cfg['backend_hash'])
    before = state(r); cold = probe.observe(8, time.perf_counter()); restoration = audit(before, state(r))
    independent = oracle(probe)
    error = max(abs(a-b) for a,b in zip(cold['chunk_losses'], independent)); assert error < 1e-6
    warm = probe.observe(8, time.perf_counter()); assert cold['chunk_losses'] == warm['chunk_losses']
    r.update_observer.arm(2)
    loss = r.step(8, batches, next_step=9, prefetch=True); r.tail.tick(8); torch.cuda.synchronize()
    parity = compare(r, loss.item(), baseline); del baseline
    assert parity['all_parameters_bitwise_equal'], parity['maximum_parameter_relative_rms']
    assert any(r.tables.pending) and any(r.gradients.pending)
    before = state(r); pending = probe.observe(9, time.perf_counter()); pending_restoration = audit(before, state(r))
    odd = r.step(9, batches, prefetch=False); r.tail.tick(9); torch.cuda.synchronize()
    assert torch.isfinite(odd).item(); freeze_after = frozen_state(r)
    adam = next(x for x in r.update_observer.reports if x['do_adam'])
    assert all(v['gradient_available'] for v in adam['measurements'].values())
    for key in ('raw_scale', 'raw_shift'):
        assert adam['measurements']['routed.'+key]['update_rms'] == 0.
    assert all(adam['measurements']['routed.'+key]['update_rms'] > 0
        for key in ('spine_c0','spine_c1','spine_c2','spine_c3','warp_c4','warp_c5'))
    assert all(adam['measurements']['routed.'+key]['gradient_sample_rms'] > 0
        for key in ('raw_scale','raw_shift'))
    projected = cold['probe_seconds'] + 6*warm['probe_seconds'] + 7*r.update_observer.total_seconds
    assert projected/900 < .03, projected
    value = dict(passed=True, mode='graph', full_graph_warmup=True, actual_prefix_updates=8,
        original_logical_ranks=8, routed_backward_sites=sorted(core.EXECUTED_MOE_BACKWARDS),
        five_original_dense_MLPs=True, frozen_state_after_prefix=freeze, frozen_state_after_probe=freeze_after,
        fixed_probe_oracle_max_abs=error, state_restoration=restoration, pending_state_restoration=pending_restoration,
        continuation_parity={k:v for k,v in parity.items() if k!='parameters'}, odd_continuation_loss=odd.item(),
        cold_probe_seconds=cold['probe_seconds'], warm_probe_seconds=warm['probe_seconds'], pending_probe_seconds=pending['probe_seconds'],
        projected_seven_probe_and_diagnostic_seconds=projected, projected_fraction_of900=projected/900,
        gradient_measurements=r.update_observer.reports, not_a_screening_reference=True,
        not_a_speed_or_convergence_win=True, full_convergence_verified=False,
        gradient_mask_boundary='After eight-rank reduction, before original geometry Adam; raw diagnostic gradients are sampled before masking.',
        kernels_unchanged=True, parameter_storage_unchanged=True)
    atomic(r.research_run/'calibration.json', value); r.log(event='calibration_complete', **value)


def create_run(value, seed, kind, unit, calibration=None, frozen=None):
    from wr_reproduction import research
    if stopped(): raise RuntimeError('Explicit study STOP')
    validate(value); fence = assert_fence(value)
    dirty = control.git('status', '--porcelain', '--', 'wr_reproduction', 'configs/wr_reproduction', 'ks3_research/vendor')
    if dirty: raise RuntimeError('Commit implementation/config before launch: ' + dirty)
    code = control.sources()
    if frozen and code != frozen['source_hashes']: raise RuntimeError('Frozen verification source changed')
    rid = value['name']+'-'+kind+'-'+now().replace('-', '').replace(':', '').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory = STATE/'runs'/rid; directory.mkdir(parents=True); head = control.git('rev-parse', 'HEAD')
    run = dict(study='probe-study', run_id=rid, directory=str(directory), seed=seed, kind=kind,
        recipe=value, unit=unit, created_utc=now(),
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_fixed_affine_trial', 'train', str(directory), '--mode', calibration or 'discovery'],
        code_commit=frozen['code_commit'] if frozen else head, launch_head_commit=head, source_hashes=code,
        environment=environment(), protocol_path=str(PROTOCOL), fence_path=str(FENCE),
        backend_hash=fence['backend_hash'], backend_fence=fence, probe_protocol=read(PROTOCOL),
        validation_data=research.validation_data(), continuous_from_initialization=not calibration,
        comparative_pruning_exempt=kind=='verification', stop_files=[str(p) for p in STOP_FILES],
        calibration=calibration, fixed_affine=KIND, frozen_geometry_columns=[6, 7],
        unchanged_forward_backward_kernels=True, additional_control_cost_on_continuous_clock=True)
    atomic(directory/'run.json', run); atomic(directory/'config.json', value)
    atomic(RESULTS/(rid+'-config.json'), run)
    append(STATE/'registry.jsonl', dict(event='registered', run_id=rid, directory=str(directory), kind=kind, seed=seed, code_commit=run['code_commit']))
    control.status(run, 'registered'); control.note(run, hypothesis=value['hypothesis'], decision='Launch '+kind, code_commit=run['code_commit'])
    return run


def train(directory, mode):
    import torch
    cfg = read(directory/'run.json'); validate(cfg['recipe'])
    if stopped(): raise RuntimeError('Explicit study STOP')
    if control.sources() != cfg['source_hashes']: raise RuntimeError('Sources changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS', 'MOE_GRAD_FP8', 'MOE_DW_PROMOTE', 'MOE_DX_PROMOTE'):
        if key in os.environ: raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED'] = str(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(1)
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction import probe_train
    with LocalCollectives():
        r = runner_class()('pr393', directory/'events.jsonl', True, 'bf16', directory)
        r.warmup()
        if mode == 'graph': graph_gate(r)
        else:
            probe_train.train(r)
            if not stopped(): r.log(event='fixed_affine_terminal_audit', **frozen_state(r))


def require_gate(value):
    gate = read(RESULTS/'fixed-affine-graph-gate.json', {})
    if not gate.get('passed'): raise RuntimeError('Passed fixed-affine graph gate required')
    if gate['recipe'] != value or gate['source_hashes'] != control.sources():
        raise RuntimeError('Fixed-affine recipe or sources changed since gate')


def work(request):
    cfg = read(request); mode = cfg['mode']; value = cfg['recipe']
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stopped(): return
        if mode == 'discovery': require_gate(value)
        run = create_run(value, 169, 'calibration' if mode=='graph' else 'discovery', cfg['unit'], 'graph' if mode=='graph' else None)
        result = control.execute(run)
        if mode == 'graph':
            gate = dict(utc=now(), run_id=run['run_id'], code_commit=run['code_commit'], recipe=value, source_hashes=run['source_hashes'],
                passed=result['status']=='calibration_complete' and bool((result.get('calibration') or {}).get('passed')), result=result)
            for root in (STATE, RESULTS): atomic(root/'fixed-affine-graph-gate.json', gate)
        if result['qualifies']:
            control.create_run = create_run; control.verification(run, result, cfg['unit'])


def launch(path, mode):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value = read(path); validate(value)
    if mode == 'discovery': require_gate(value)
    unit = 'wr-probe-fixed-affine-'+uuid.uuid4().hex[:8]; request = STATE/'requests'/(unit+'.json')
    atomic(request, dict(utc=now(), unit=unit, mode=mode, recipe=value))
    subprocess.run(['systemd-run', '--unit='+unit, '--working-directory='+str(ROOT),
        '--property=KillMode=control-group', '--property=TimeoutStopSec=15', '--property=PartOf=wr-ks3-probe-research.service',
        *systemd_environment(), str(control.PYTHON), '-m', 'wr_reproduction.probe_fixed_affine_trial', 'work', str(request)], check=True)
    print(unit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'work', 'train', 'cpu-check'])
    p.add_argument('path', type=Path, nargs='?'); p.add_argument('--mode', choices=['graph', 'discovery'], default='graph'); args = p.parse_args()
    try:
        if args.action == 'cpu-check':
            import json
            print(json.dumps(cpu_checks(), indent=2))
        elif args.action in ('launch', 'train'): globals()[args.action](args.path, args.mode)
        else: work(args.path)
    except BaseException:
        error = dict(utc=now(), error=traceback.format_exc())
        atomic(STATE/'fixed-affine-blocker.json', error); publish('directed-status.json', dict(status='error', **error)); raise
