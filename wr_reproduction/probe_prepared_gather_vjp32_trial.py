"""Prepared geometry plus exact gather/row32 VJP: numerical, actual-model speed/state, loss/time."""
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

KIND = 'prepared-geometry-exact-gather-vjp32-original-function-v1'
SETTINGS = dict(kind=KIND, original_selected_tiles=True, training_only=True,
    original_GS32_quantization=True, new_cache_bytes=1474560, geometry_rows=32,
    gradient_producer='exact-gather-original-torch-gate-sum')
EVIDENCE = dict(speed_reference_run_ids=[
    'probe-experts-prepared-geometry-discovery-20261026T155013-becd49',
    'probe-experts-bf16-gather-vjp32-discovery-20261026T140450-037785'],
    maximum_actual_model_prefix_ratio=1.0, require_faster_than_both_ingredients=True,
    profile_run_id='ks3-bf16-shared-fp8-profile-20261025T095458-924154')


def validate(value):
    assert_fence(value)
    baseline=control.recipe('ks3_experts');baseline['geometry_backward_rows']=32
    omit={'name','hypothesis','hypothesis_class','prepared_gather_vjp32','prepared_gather_vjp32_evidence'}
    if {k:v for k,v in value.items() if k not in omit}!={k:v for k,v in baseline.items() if k not in omit}:
        raise ValueError('Only prepared geometry plus the validated exact gather/row32 combination may change from the incumbent')
    if value.get('prepared_gather_vjp32')!=SETTINGS or value.get('prepared_gather_vjp32_evidence')!=EVIDENCE:
        raise ValueError('Unsupported prepared geometry cache/tiles/quantization or speed criterion')


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
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_prepared_gather_vjp32_trial', 'train', str(directory), '--mode', calibration or 'discovery'],
        code_commit=frozen['code_commit'] if frozen else head, launch_head_commit=head, source_hashes=code,
        environment=environment(), protocol_path=str(PROTOCOL), fence_path=str(FENCE),
        backend_hash=fence['backend_hash'], backend_fence=fence, probe_protocol=read(PROTOCOL),
        validation_data=research.validation_data(), continuous_from_initialization=not calibration,
        comparative_pruning_exempt=kind=='verification', stop_files=[str(p) for p in STOP_FILES],
        calibration=calibration, prepared_gather_vjp32=SETTINGS,
        kernel_scope='Reuse prepared original GS32 geometry operands with the validated exact BF16 gradient gather and32-row VJP. Cache adds no math change to that ingredient. Keep original forward/evaluation, product tiles, coordinate-gradient BF16 boundary, all rates and five denseMLPs. No shared backend changes.')
    atomic(directory/'run.json', run); atomic(directory/'config.json', value); atomic(RESULTS/(rid+'-config.json'), run)
    append(STATE/'registry.jsonl', dict(event='registered', run_id=rid, directory=str(directory), kind=kind, seed=seed, code_commit=run['code_commit']))
    control.status(run, 'registered'); control.note(run, hypothesis=value['hypothesis'], decision='Launch '+kind, code_commit=run['code_commit'])
    return run


def train(directory,mode):
    import torch
    cfg=read(directory/'run.json');value=cfg['recipe'];validate(value)
    if stopped():raise RuntimeError('Explicit study STOP')
    if control.sources()!=cfg['source_hashes']:raise RuntimeError('Sources changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    if mode=='numerical':
        from wr_reproduction.check_ks3_prepared_gather_vjp32 import check
        result=dict(mode=mode,**check());atomic(directory/'calibration.json',result)
        append(directory/'events.jsonl',dict(event='calibration_complete',**result));return
    from wr_reproduction.ks3_prepared_gather_vjp32 import install,bind,prepare,EXECUTED
    install()
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.probe_geometry_effect_trial import runner_class
    from wr_reproduction import probe_train,ks3_moe as core
    from wr_reproduction.matched_architecture import audit_model
    with LocalCollectives():
        r=runner_class()('pr393',directory/'events.jsonl',True,'bf16',directory)
        # Replicated Adam has now moved geometry into its final flat parameter
        # bank. Establish cache lookup keys before any training graph capture.
        bind(r.model);prepare(r.model)
        inventory=audit_model(r.model,'ks3_experts');assert inventory['active_conventional_mlp_instances']==5
        r.log(event='feedforward_inventory',**inventory)
        r.log(event='prepared_gather_vjp32_architecture',settings=SETTINGS,
            extra_buffer_bytes=1474560,new_parameters=0,original_function_and_rates=True,
            original_FP16_nonlinearity=True,original_geometry_VJP_equations=True,
            original_evaluation_forward=True,original_selected_matmul_tiles=True)
        r.warmup()
        if mode=='discovery':probe_train.train(r);return
        from wr_reproduction.probe_precision_calibration import run
        from wr_reproduction.probe_precision_trial import speed_report
        run(r);calibration=read(directory/'calibration.json')
        assert core.EXECUTED_MOE_BACKWARDS==set(range(5))
        expected={'cached_'+name+':'+str(site) for name in ('geometry','vjp') for site in range(5)}
        assert expected|{'prepare_all_five_sites','original_evaluation_geometry'} <= EXECUTED,EXECUTED
        m=r.model
        assert m.ks3_prepared_geometry.numel()*m.ks3_prepared_geometry.element_size()==1474560
        from wr_reproduction import ks3_bf16_gather_vjp32 as ingredient
        assert ingredient.EXECUTED_MODES=={'training_bf16_exact_gather_vjp32'}
        comparisons=[];candidate_rows=events(directory/'events.jsonl')
        for ref_id in EVIDENCE['speed_reference_run_ids']:
            ref_run=read(STATE/'runs'/ref_id/'run.json')
            assert ref_run['seed']==cfg['seed']==169 and ref_run['backend_hash']==cfg['backend_hash']
            assert ref_run['continuous_from_initialization'] and not ref_run.get('calibration')
            omit={'name','hypothesis','hypothesis_class','prepared_gather_vjp32','prepared_gather_vjp32_evidence',
                'prepared_geometry','prepared_geometry_evidence','bf16_gather_vjp32','bf16_combination_evidence',
                'geometry_backward_rows','gradient_diagnostics_extension'}
            assert ref_run['recipe'].get('gradient_diagnostics_extension') in (None,'active-slot-pre-optimizer-v1')
            assert {k:v for k,v in ref_run['recipe'].items() if k not in omit}=={k:v for k,v in value.items() if k not in omit}
            paths=['wr_reproduction/ks3_moe.py','wr_reproduction/ks3_bf16_gemm.py',
                'wr_reproduction/ks3_weight_pack.py','wr_reproduction/bf16_tiles.py',
                'wr_reproduction/ks3_bf16_gather.py','wr_reproduction/ks3_bf16_gather_vjp32.py']
            if 'prepared_geometry' in ref_run['recipe']:
                from wr_reproduction.probe_prepared_geometry_trial import SETTINGS as cache_settings
                assert ref_run['recipe']['prepared_geometry']==cache_settings
                assert ref_run['recipe']['geometry_backward_rows']==64
                paths.append('wr_reproduction/ks3_prepared_geometry.py')
            else:
                from wr_reproduction.probe_bf16_gather_vjp32_trial import SETTINGS as backward_settings
                assert ref_run['recipe']['bf16_gather_vjp32']==backward_settings
                assert ref_run['recipe']['geometry_backward_rows']==32
            for path in paths:assert ref_run['source_hashes'][path]==cfg['source_hashes'][path],path
            reference_rows=events(STATE/'runs'/ref_id/'events.jsonl')
            a={x['step']:x for x in candidate_rows if x.get('event')=='train'}
            b={x['step']:x for x in reference_rows if x.get('event')=='train'}
            for step in (1,25,100):assert a[step]['objective_signature']==b[step]['objective_signature']
            item=speed_report(candidate_rows,reference_rows)
            item.update(reference_run_id=ref_id,reference_code_commit=ref_run['code_commit'])
            comparisons.append(item)
        speed=dict(passed=all(item['passed'] for item in comparisons),comparisons=comparisons,
            criterion='Both first100 and updates26-100 must be strictly faster than BOTH constituent discovery prefixes.',
            limitation='Single measured prefixes; not runtime sampling uncertainty or convergence evidence.')
        projected=calibration['cold_probe_seconds']+6*calibration['warm_probe_seconds']
        passed=bool(calibration['passed'] and calibration['continuation_parity']['all_parameters_bitwise_equal'] and speed['passed'] and projected/900<=.03)
        result=dict(passed=passed,mode='speed_state',state_calibration=calibration,speed=speed,
            projected_seven_probe_seconds=projected,projected_fraction_of900=projected/900,
            all_five_routed_backwards_executed=True,five_original_dense_MLPs=True,
            executed_geometry_paths=sorted(EXECUTED),extra_buffer_bytes=1474560,
            cache_graph_execution_covered=True,original_evaluation_forward=True,
            not_a_screening_reference=True,full_convergence_verified=False,
            overhead_limitation='Measured projection includes probe setup/scoring/restoration; policy and diagnostics remain on the actual clock.')
        atomic(directory/'probe-state-calibration.json',calibration);atomic(directory/'calibration.json',result)
        r.log(event='prepared_gather_vjp32_speed_state_gate',**result)


def require_gate(mode, value):
    gate = read(RESULTS/('prepared-gather-vjp32-'+mode+'-gate.json'), {})
    if not gate.get('passed'): raise RuntimeError('Missing passed prepared-gather-vjp32 '+mode+' gate')
    if gate['recipe'] != value or gate['source_hashes'] != control.sources():
        raise RuntimeError('Recipe or source changed since prepared-gather-vjp32 '+mode+' gate')


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
            for root in (STATE, RESULTS): atomic(root/('prepared-gather-vjp32-'+mode+'-gate.json'), gate)
        if result['qualifies']:
            control.create_run = create_run; control.verification(run, result, cfg['unit'])


def launch(path, mode):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value = read(path); validate(value)
    if mode != 'numerical': require_gate('numerical', value)
    if mode == 'discovery': require_gate('speed_state', value)
    unit = 'wr-probe-prepared-gather-vjp32-'+uuid.uuid4().hex[:8]; request = STATE/'requests'/(unit+'.json')
    atomic(request, dict(utc=now(), unit=unit, mode=mode, recipe=value))
    subprocess.run(['systemd-run', '--unit='+unit, '--working-directory='+str(ROOT),
        '--property=KillMode=control-group', '--property=TimeoutStopSec=15', '--property=PartOf=wr-ks3-probe-research.service',
        *systemd_environment(), str(control.PYTHON), '-m', 'wr_reproduction.probe_prepared_gather_vjp32_trial', 'work', str(request)], check=True)
    print(unit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'work', 'train']); p.add_argument('path', type=Path)
    p.add_argument('--mode', choices=['numerical', 'speed_state', 'discovery'], default='numerical'); args = p.parse_args()
    try:
        if args.action in ('launch', 'train'): globals()[args.action](args.path, args.mode)
        else: work(args.path)
    except BaseException:
        error = dict(utc=now(), error=traceback.format_exc())
        atomic(STATE/'prepared-gather-vjp32-blocker.json', error); publish('directed-status.json', dict(status='error', **error)); raise
