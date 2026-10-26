"""Numerical, actual-model cost/state, then fixed-text loss/time gates."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import (ROOT, STATE, RESULTS, PROTOCOL, FENCE,
    STOP_FILES, read, atomic, append, events, now, stopped, assert_fence, environment,
    systemd_environment, publish)

KIND = 'direct-fp16-geometry-coefficients-v1'


def validate(value):
    assert_fence(value)
    baseline = control.recipe('ks3_experts')
    omit = {'name', 'hypothesis', 'hypothesis_class', 'coefficient_precision', 'coefficient_precision_evidence'}
    if {k:v for k,v in value.items() if k not in omit} != {k:v for k,v in baseline.items() if k not in omit}:
        raise ValueError('Only routed geometry coefficient precision may change')
    if value.get('coefficient_precision') != KIND:
        raise ValueError('Unsupported coefficient precision')
    if value['coefficient_precision_evidence']['maximum_actual_model_prefix_ratio'] != 1.0:
        raise ValueError('Predeclared no-slower actual-model gate changed')


def create_run(value, seed, kind, unit, calibration=None, frozen=None):
    from wr_reproduction import research
    if stopped(): raise RuntimeError('Explicit study STOP')
    validate(value); fence = assert_fence(value)
    dirty = control.git('status', '--porcelain', '--', 'wr_reproduction', 'configs/wr_reproduction', 'ks3_research/vendor')
    if dirty: raise RuntimeError('Commit implementation/config before launch: '+dirty)
    code = control.sources()
    if frozen and code != frozen['source_hashes']: raise RuntimeError('Frozen verification source changed')
    rid = value['name']+'-'+kind+'-'+now().replace('-', '').replace(':', '').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory = STATE/'runs'/rid; directory.mkdir(parents=True); head = control.git('rev-parse', 'HEAD')
    run = dict(study='probe-study', run_id=rid, directory=str(directory), seed=seed, kind=kind,
        recipe=value, unit=unit, created_utc=now(),
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_fp16_coefficient_trial', 'train', str(directory), '--mode', calibration or 'discovery'],
        code_commit=frozen['code_commit'] if frozen else head, launch_head_commit=head, source_hashes=code,
        environment=environment(), protocol_path=str(PROTOCOL), fence_path=str(FENCE),
        backend_hash=fence['backend_hash'], backend_fence=fence, probe_protocol=read(PROTOCOL),
        validation_data=research.validation_data(), continuous_from_initialization=not calibration,
        comparative_pruning_exempt=kind=='verification', stop_files=[str(p) for p in STOP_FILES],
        calibration=calibration, coefficient_precision=KIND,
        kernel_scope='Direct FP16 geometry coefficients in KS3 rank1 forward and smooth STE; original FP16 Horner, GS32 activation/projection packing, BF16 products, optimizer settings and shared backend retained')
    atomic(directory/'run.json', run); atomic(directory/'config.json', value); atomic(RESULTS/(rid+'-config.json'), run)
    append(STATE/'registry.jsonl', dict(event='registered', run_id=rid, directory=str(directory), kind=kind, seed=seed, code_commit=run['code_commit']))
    control.status(run, 'registered'); control.note(run, hypothesis=value['hypothesis'], decision='Launch '+kind, code_commit=run['code_commit'])
    return run


def train(directory, mode):
    import torch
    cfg = read(directory/'run.json'); value = cfg['recipe']; validate(value)
    if stopped(): raise RuntimeError('Explicit study STOP')
    if control.sources() != cfg['source_hashes']: raise RuntimeError('Sources changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS', 'MOE_GRAD_FP8', 'MOE_DW_PROMOTE', 'MOE_DX_PROMOTE'):
        if key in os.environ: raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED'] = str(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(1)
    if mode == 'numerical':
        from wr_reproduction.check_ks3_fp16_coefficients import check
        result = dict(mode=mode, **check()); atomic(directory/'calibration.json', result)
        append(directory/'events.jsonl', dict(event='calibration_complete', **result)); return
    from wr_reproduction.ks3_fp16_coefficients import install
    install()
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.probe_geometry_effect_trial import runner_class
    from wr_reproduction import probe_train, ks3_moe as core
    from wr_reproduction.matched_architecture import audit_model
    with LocalCollectives():
        r = runner_class()('pr393', directory/'events.jsonl', True, 'bf16', directory)
        inventory = audit_model(r.model, 'ks3_experts'); assert inventory['active_conventional_mlp_instances'] == 5
        r.log(event='feedforward_inventory', **inventory)
        r.log(event='fp16_coefficient_architecture', coefficient_dtype='float16', raw_scale_shift_unchanged=True,
            both_components_trainable=True, no_new_parameters=True, zero_initial_function_unchanged=True,
            original_projection_and_geometry_rates=True, original_FP16_nonlinearity=True,
            hypothesis='Accuracy through direct FP16 coefficients and reduced quantizer work; no speed or convergence benefit assumed')
        r.warmup()
        if mode == 'discovery': probe_train.train(r); return
        from wr_reproduction.probe_precision_calibration import run
        from wr_reproduction.probe_precision_trial import timing
        run(r); calibration = read(directory/'calibration.json')
        assert core.EXECUTED_MOE_BACKWARDS == set(range(5))
        ref_id = value['coefficient_precision_evidence']['speed_reference_run_id']
        ref_run = read(STATE/'runs'/ref_id/'run.json')
        assert ref_run['seed'] == cfg['seed'] == 169 and ref_run['backend_hash'] == cfg['backend_hash']
        omit={'name','hypothesis','hypothesis_class','coefficient_precision','coefficient_precision_evidence'}
        assert {k:v for k,v in ref_run['recipe'].items() if k not in omit}=={k:v for k,v in value.items() if k not in omit}
        for path in ('wr_reproduction/ks3_moe.py','wr_reproduction/ks3_bf16_gemm.py','wr_reproduction/ks3_weight_pack.py'):
            assert ref_run['source_hashes'][path]==cfg['source_hashes'][path],path
        current_rows=events(directory/'events.jsonl');reference_rows=events(STATE/'runs'/ref_id/'events.jsonl')
        a={x['step']:x for x in current_rows if x.get('event')=='train'}
        b={x['step']:x for x in reference_rows if x.get('event')=='train'}
        for step in (1,25,100):assert a[step]['objective_signature']==b[step]['objective_signature']
        candidate = timing(events(directory/'events.jsonl')); reference = timing(events(STATE/'runs'/ref_id/'events.jsonl'))
        ratios = {k:candidate[k]/reference[k] for k in ('first100_seconds', 'after25_through100_seconds')}
        cost_pass = all(x <= 1.0 for x in ratios.values())
        speed = dict(passed=cost_pass, candidate=candidate, reference=reference, ratios=ratios,
            reference_run_id=ref_id, reference_code_commit=ref_run['code_commit'],
            criterion='Both first100 and updates26-100 actual-model durations must be no slower than the saved original incumbent prefix. No allowed slowdown.',
            limitation='Single prefix; no claim of runtime-noise precision. Any convergence gain must survive continuous elapsed-time screening and full validation.')
        projected = calibration['cold_probe_seconds']+6*calibration['warm_probe_seconds']
        passed = bool(calibration['passed'] and calibration['continuation_parity']['all_parameters_bitwise_equal'] and cost_pass and projected/900 <= .03)
        result = dict(passed=passed, mode='speed_state', state_calibration=calibration, speed=speed,
            projected_seven_probe_seconds=projected, projected_fraction_of900=projected/900,
            all_five_routed_backwards_executed=True, five_original_dense_MLPs=True,
            not_a_screening_reference=True, full_convergence_verified=False,
            overhead_limitation='Measured projection includes probe setup/scoring/restoration; policy and diagnostics remain on the actual clock.')
        atomic(directory/'probe-state-calibration.json', calibration); atomic(directory/'calibration.json', result)
        r.log(event='fp16_coefficient_speed_state_gate', **result)


def require_gate(mode, value):
    gate = read(RESULTS/('fp16-coefficient-'+mode+'-gate.json'), {})
    if not gate.get('passed'): raise RuntimeError('Missing passed fp16-coefficient '+mode+' gate')
    if gate['recipe'] != value or gate['source_hashes'] != control.sources():
        raise RuntimeError('Recipe or source changed since fp16-coefficient '+mode+' gate')


def work(request):
    cfg = read(request); mode = cfg['mode']; value = cfg['recipe']
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stopped(): return
        if mode != 'numerical': require_gate('numerical', value)
        if mode == 'discovery': require_gate('speed_state', value)
        run = create_run(value, 169, 'discovery' if mode=='discovery' else 'calibration', cfg['unit'], None if mode=='discovery' else mode)
        result = control.execute(run)
        if mode != 'discovery':
            gate = dict(utc=now(), run_id=run['run_id'], code_commit=run['code_commit'], recipe=value, source_hashes=run['source_hashes'],
                passed=result['status']=='calibration_complete' and bool((result.get('calibration') or {}).get('passed')), result=result)
            for root in (STATE, RESULTS): atomic(root/('fp16-coefficient-'+mode+'-gate.json'), gate)
        if result['qualifies']:
            control.create_run = create_run; control.verification(run, result, cfg['unit'])


def launch(path, mode):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value = read(path); validate(value)
    if mode != 'numerical': require_gate('numerical', value)
    if mode == 'discovery': require_gate('speed_state', value)
    unit = 'wr-probe-fp16-coefficient-'+uuid.uuid4().hex[:8]; request = STATE/'requests'/(unit+'.json')
    atomic(request, dict(utc=now(), unit=unit, mode=mode, recipe=value))
    subprocess.run(['systemd-run', '--unit='+unit, '--working-directory='+str(ROOT),
        '--property=KillMode=control-group', '--property=TimeoutStopSec=15', '--property=PartOf=wr-ks3-probe-research.service',
        *systemd_environment(), str(control.PYTHON), '-m', 'wr_reproduction.probe_fp16_coefficient_trial', 'work', str(request)], check=True)
    print(unit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'work', 'train']); p.add_argument('path', type=Path)
    p.add_argument('--mode', choices=['numerical', 'speed_state', 'discovery'], default='numerical'); args = p.parse_args()
    try:
        if args.action in ('launch', 'train'): globals()[args.action](args.path, args.mode)
        else: work(args.path)
    except BaseException:
        error = dict(utc=now(), error=traceback.format_exc())
        atomic(STATE/'fp16-coefficient-blocker.json', error); publish('directed-status.json', dict(status='error', **error)); raise
