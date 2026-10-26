"""Exact BF16 gradient producer for the compact residual: numerical, speed/state, loss/time."""
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

from wr_reproduction.probe_local_residual_trial import SETTINGS
KIND = 'compact-residual-exact-bf16-gradient-gather-v1'
GATHER = dict(kind=KIND, speed_reference_run_id='probe-experts-local-residual-k3-discovery-20261026T122452-435b40', maximum_prefix_ratio=1.0)


def validate(value):
    assert_fence(value)
    baseline = control.recipe('ks3_experts'); baseline['local_residual'] = SETTINGS
    omit = {'name', 'hypothesis', 'hypothesis_class', 'local_residual_gather'}
    if {k:v for k,v in value.items() if k not in omit} != {k:v for k,v in baseline.items() if k not in omit}:
        raise ValueError('Only the explicit exact gradient producer may change from the frozen compact residual')
    if value.get('local_residual_gather') != GATHER:
        raise ValueError('Unsupported compact-residual gather recipe or speed reference')


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
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_local_residual_gather_trial', 'train', str(directory), '--mode', calibration or 'discovery'],
        code_commit=frozen['code_commit'] if frozen else head, launch_head_commit=head, source_hashes=code,
        environment=environment(), protocol_path=str(PROTOCOL), fence_path=str(FENCE),
        backend_hash=fence['backend_hash'], backend_fence=fence, probe_protocol=read(PROTOCOL),
        validation_data=research.validation_data(), continuous_from_initialization=not calibration,
        comparative_pruning_exempt=kind=='verification', stop_files=[str(p) for p in STOP_FILES],
        calibration=calibration, local_residual=SETTINGS, local_residual_gather=GATHER,
        kernel_scope='ExactBF16 gradient gather applied only to the frozen compact-residual backward. No feature/function/gradient-boundary/optimizer or shared-kernel changes.')
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
        from wr_reproduction.check_ks3_local_residual_gather import check
        result = dict(mode=mode, **check()); atomic(directory/'calibration.json', result)
        append(directory/'events.jsonl', dict(event='calibration_complete', **result)); return
    from wr_reproduction.ks3_local_residual_gather import install
    install()
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.probe_local_residual_train import LocalResidualRunner, diagnostics
    from wr_reproduction import probe_train
    from wr_reproduction.matched_architecture import audit_model
    with LocalCollectives(), diagnostics():
        r = LocalResidualRunner('pr393', directory/'events.jsonl', True, 'bf16', directory)
        inventory = audit_model(r.model, 'ks3_experts')
        assert inventory['active_conventional_mlp_instances'] == 5
        r.log(event='feedforward_inventory', **inventory); r.warmup()
        r.log(event='local_residual_gather_architecture',forward_unchanged=True,
            original_BF16_gradient_boundary_unchanged=True,optimizer_unchanged=True,
            original_Torch_gate_reduction_unchanged=True,shared_backend_unchanged=True)
        if mode == 'discovery': probe_train.train(r); return
        from wr_reproduction.probe_precision_calibration import run
        from wr_reproduction.probe_precision_trial import speed_report
        run(r); calibration=read(directory/'calibration.json')
        ref_id=GATHER['speed_reference_run_id'];ref=read(STATE/'runs'/ref_id/'run.json')
        assert ref['seed']==cfg['seed']==169 and ref['backend_hash']==cfg['backend_hash']
        assert ref['continuous_from_initialization'] and not ref.get('calibration')
        omit={'name','hypothesis','hypothesis_class','local_residual_gather'}
        assert {k:v for k,v in ref['recipe'].items() if k not in omit}=={k:v for k,v in value.items() if k not in omit}
        for path in ('wr_reproduction/ks3_local_residual.py','wr_reproduction/ks3_moe.py','wr_reproduction/probe_local_residual_train.py'):
            assert ref['source_hashes'][path]==cfg['source_hashes'][path],path
        speed=speed_report(events(directory/'events.jsonl'),events(STATE/'runs'/ref_id/'events.jsonl'))
        speed.update(reference_run_id=ref_id,reference_code_commit=ref['code_commit'])
        projected=calibration['cold_probe_seconds']+6*calibration['warm_probe_seconds']
        from wr_reproduction import ks3_moe as core
        assert core.EXECUTED_MOE_BACKWARDS==set(range(5))
        learned=r.model.ks3_local.abs().amax(dim=(1,2));assert (learned>0).all()
        passed=bool(calibration['passed'] and calibration['continuation_parity']['all_parameters_bitwise_equal']
                    and speed['passed'] and projected/900<=.03)
        result=dict(passed=passed,mode='speed_state',state_calibration=calibration,speed=speed,
            projected_seven_probe_seconds=projected,projected_fraction_of900=projected/900,
            per_site_knot_learned_absmax=learned.cpu().tolist(),all_five_routed_backwards_executed=True,
            five_original_dense_MLPs=True,not_a_screening_reference=True,full_convergence_verified=False,
            overhead_limitation='Probe preparation/scoring/restoration measured; policy and sparse diagnostics stay on the actual discovery clock.')
        atomic(directory/'probe-state-calibration.json',calibration);atomic(directory/'calibration.json',result)
        r.log(event='local_residual_gather_speed_state_gate',**result)


def require_gate(mode, value):
    gate = read(RESULTS/('local-residual-gather-'+mode+'-gate.json'), {})
    if not gate.get('passed'): raise RuntimeError('Missing passed local-residual '+mode+' gate')
    if gate['recipe'] != value or gate['source_hashes'] != control.sources():
        raise RuntimeError('Recipe or source changed since local-residual '+mode+' gate')


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
            for root in (STATE, RESULTS): atomic(root/('local-residual-gather-'+mode+'-gate.json'), gate)
        if result['qualifies']:
            control.create_run = create_run; control.verification(run, result, cfg['unit'])


def launch(path, mode):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value = read(path); validate(value)
    if mode != 'numerical': require_gate('numerical', value)
    if mode == 'discovery': require_gate('speed_state', value)
    unit = 'wr-probe-local-residual-gather-'+uuid.uuid4().hex[:8]; request = STATE/'requests'/(unit+'.json')
    atomic(request, dict(utc=now(), unit=unit, mode=mode, recipe=value))
    subprocess.run(['systemd-run', '--unit='+unit, '--working-directory='+str(ROOT),
        '--property=KillMode=control-group', '--property=TimeoutStopSec=15', '--property=PartOf=wr-ks3-probe-research.service',
        *systemd_environment(), str(control.PYTHON), '-m', 'wr_reproduction.probe_local_residual_gather_trial', 'work', str(request)], check=True)
    print(unit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'work', 'train']); p.add_argument('path', type=Path)
    p.add_argument('--mode', choices=['numerical', 'speed_state', 'discovery'], default='numerical'); args = p.parse_args()
    try:
        if args.action in ('launch', 'train'): globals()[args.action](args.path, args.mode)
        else: work(args.path)
    except BaseException:
        error = dict(utc=now(), error=traceback.format_exc())
        atomic(STATE/'local-residual-gather-blocker.json', error); publish('directed-status.json', dict(status='error', **error)); raise
