"""Fused bounded-gain producers: numerical, actual-model speed/state, loss/time."""
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

from wr_reproduction.probe_pairwise_gain_trial import SETTINGS
KIND = 'pairwise-gain-fused-independent-feature-producers-v1'
FUSION = dict(kind=KIND, unfused_reference_run_id='probe-experts-pairwise-gain-b05-discovery-20261026T144110-77190b', incumbent_reference_run_id='probe-ks3-experts-reference-20261026T043959-e5168d', maximum_incumbent_prefix_ratio=1.02, require_strictly_faster_than_unfused=True)


def validate(value):
    assert_fence(value)
    baseline = control.recipe('ks3_experts'); baseline['pairwise_gain'] = SETTINGS
    omit = {'name', 'hypothesis', 'hypothesis_class', 'pairwise_gain_fused'}
    if {k:v for k,v in value.items() if k not in omit} != {k:v for k,v in baseline.items() if k not in omit}:
        raise ValueError('Only explicit feature/prepare producer fusion may change from the frozen bounded gain')
    if value.get('pairwise_gain_fused') != FUSION:
        raise ValueError('Unsupported bounded-gain fusion or predeclared speed references')


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
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_pairwise_gain_fused_trial', 'train', str(directory), '--mode', calibration or 'discovery'],
        code_commit=frozen['code_commit'] if frozen else head, launch_head_commit=head, source_hashes=code,
        environment=environment(), protocol_path=str(PROTOCOL), fence_path=str(FENCE),
        backend_hash=fence['backend_hash'], backend_fence=fence, probe_protocol=read(PROTOCOL),
        validation_data=research.validation_data(), continuous_from_initialization=not calibration,
        comparative_pruning_exempt=kind=='verification', stop_files=[str(p) for p in STOP_FILES],
        calibration=calibration, pairwise_gain=SETTINGS, pairwise_gain_fused=FUSION,
        kernel_scope='Fuse original independent geometry with the gain feature consumer and backward gain preparation. Preserve original intermediateBF16 rounding, geometryVJP, direct partnerVJP, reductions, projection/gather kernels, parameters, rates and shared backend.')
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
        from wr_reproduction.check_ks3_pairwise_gain_fused import check
        result = dict(mode=mode, **check(SETTINGS)); atomic(directory/'calibration.json', result)
        append(directory/'events.jsonl', dict(event='calibration_complete', **result)); return
    from wr_reproduction.ks3_pairwise_gain_fused import install
    install()
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.probe_pairwise_gain_train import PairGainRunner, diagnostics
    from wr_reproduction import probe_train
    from wr_reproduction.matched_architecture import audit_model
    with LocalCollectives(), diagnostics():
        r = PairGainRunner('pr393', directory/'events.jsonl', True, 'bf16', directory)
        inventory = audit_model(r.model, 'ks3_experts')
        assert inventory['active_conventional_mlp_instances'] == 5
        r.log(event='feedforward_inventory', **inventory); r.warmup()
        r.log(event='pairwise_gain_fused_architecture',mathematical_function_unchanged=True,
            original_BF16_gradient_boundary_unchanged=True,optimizer_unchanged=True,
            original_Torch_gate_reduction_unchanged=True,shared_backend_unchanged=True,
            fused_producers=['independent_geometry_and_feature_gain','independent_geometry_and_gain_prepare'])
        if mode == 'discovery': probe_train.train(r); return
        from wr_reproduction.probe_precision_calibration import run
        from wr_reproduction.probe_precision_trial import speed_report
        run(r); calibration=read(directory/'calibration.json')
        candidate_rows=events(directory/'events.jsonl');comparisons=[]
        for role,key,ceiling in [('unfused','unfused_reference_run_id',1.),('incumbent','incumbent_reference_run_id',1.02)]:
            ref_id=FUSION[key];ref=read(STATE/'runs'/ref_id/'run.json')
            assert ref['seed']==cfg['seed']==169 and ref['backend_hash']==cfg['backend_hash']
            assert ref['continuous_from_initialization'] and not ref.get('calibration')
            omit={'name','hypothesis','hypothesis_class','pairwise_gain_fused','pairwise_gain','gradient_diagnostics_extension'}
            assert {k:v for k,v in ref['recipe'].items() if k not in omit}=={k:v for k,v in value.items() if k not in omit}
            assert ref['recipe'].get('gradient_diagnostics_extension') in (None,'active-slot-pre-optimizer-v1')
            assert ref['recipe'].get('pairwise_gain') == (SETTINGS if role=='unfused' else None)
            paths=['wr_reproduction/ks3_moe.py','wr_reproduction/ks3_bf16_gemm.py']
            if role=='unfused':paths+=['wr_reproduction/ks3_pairwise_gain.py','wr_reproduction/probe_pairwise_gain_train.py']
            for path in paths:assert ref['source_hashes'][path]==cfg['source_hashes'][path],path
            ref_rows=events(STATE/'runs'/ref_id/'events.jsonl')
            current={row['step']:row for row in candidate_rows if row.get('event')=='train'}
            previous={row['step']:row for row in ref_rows if row.get('event')=='train'}
            for step in (1,25,100):assert current[step]['objective_signature']==previous[step]['objective_signature']
            item=speed_report(candidate_rows,ref_rows)
            item.update(reference_run_id=ref_id,reference_code_commit=ref['code_commit'],role=role,maximum_ratio=ceiling)
            item['passed']=all(x<ceiling if role=='unfused' else x<=ceiling for x in item['ratios'].values())
            item['criterion']='Both first100 and26-100 strictly faster than unfused; no more than2% above original incumbent.'
            comparisons.append(item)
        speed=dict(passed=all(x['passed'] for x in comparisons),comparisons=comparisons,
            criterion='Both first100 and26-100 strictly faster than unfused; no more than2% above original incumbent.',
            limitation='Single measured prefixes, not a speed confidence interval or convergence win.')
        projected=calibration['cold_probe_seconds']+6*calibration['warm_probe_seconds']
        from wr_reproduction import ks3_moe as core
        assert core.EXECUTED_MOE_BACKWARDS==set(range(5))
        from wr_reproduction.ks3_pairwise_gain_fused import EXECUTED
        assert EXECUTED=={'fused_features','fused_baseline_prepare'}
        learned=r.model.ks3_pair.abs().amax(dim=(1,2));assert (learned>0).all()
        passed=bool(calibration['passed'] and calibration['continuation_parity']['all_parameters_bitwise_equal']
                    and speed['passed'] and projected/900<=.03)
        result=dict(passed=passed,mode='speed_state',state_calibration=calibration,speed=speed,
            projected_seven_probe_seconds=projected,projected_fraction_of900=projected/900,
            per_site_gain_learned_absmax=learned.cpu().tolist(),executed_producers=sorted(EXECUTED),all_five_routed_backwards_executed=True,
            five_original_dense_MLPs=True,not_a_screening_reference=True,full_convergence_verified=False,
            overhead_limitation='Probe preparation/scoring/restoration measured; policy and sparse diagnostics stay on the actual discovery clock.')
        atomic(directory/'probe-state-calibration.json',calibration);atomic(directory/'calibration.json',result)
        r.log(event='pairwise_gain_fused_speed_state_gate',**result)


def require_gate(mode, value):
    gate = read(RESULTS/('pairwise-gain-fused-'+mode+'-gate.json'), {})
    if not gate.get('passed'): raise RuntimeError('Missing passed pairwise-gain-fused '+mode+' gate')
    if gate['recipe'] != value or gate['source_hashes'] != control.sources():
        raise RuntimeError('Recipe or source changed since pairwise-gain-fused '+mode+' gate')


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
            for root in (STATE, RESULTS): atomic(root/('pairwise-gain-fused-'+mode+'-gate.json'), gate)
        if result['qualifies']:
            control.create_run = create_run; control.verification(run, result, cfg['unit'])


def launch(path, mode):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value = read(path); validate(value)
    if mode != 'numerical': require_gate('numerical', value)
    if mode == 'discovery': require_gate('speed_state', value)
    unit = 'wr-probe-pairwise-gain-fused-'+uuid.uuid4().hex[:8]; request = STATE/'requests'/(unit+'.json')
    atomic(request, dict(utc=now(), unit=unit, mode=mode, recipe=value))
    subprocess.run(['systemd-run', '--unit='+unit, '--working-directory='+str(ROOT),
        '--property=KillMode=control-group', '--property=TimeoutStopSec=15', '--property=PartOf=wr-ks3-probe-research.service',
        *systemd_environment(), str(control.PYTHON), '-m', 'wr_reproduction.probe_pairwise_gain_fused_trial', 'work', str(request)], check=True)
    print(unit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'work', 'train']); p.add_argument('path', type=Path)
    p.add_argument('--mode', choices=['numerical', 'speed_state', 'discovery'], default='numerical'); args = p.parse_args()
    try:
        if args.action in ('launch', 'train'): globals()[args.action](args.path, args.mode)
        else: work(args.path)
    except BaseException:
        error = dict(utc=now(), error=traceback.format_exc())
        atomic(STATE/'pairwise-gain-fused-blocker.json', error); publish('directed-status.json', dict(status='error', **error)); raise
