"""Explicit KS3-only FP8 numerical, actual-model speed/state, and discovery gates.

Reuses the frozen probe math, existing prepared-operand tests and process owner.
Diagnostic prefixes never enter the fixed-probe reference store.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import *


def validate(value):
    assert_fence(value)
    expected=dict(architecture='ks3_experts',precision='fp8',charts=1024,rank=1,
                  training_gs32=True,weight_packing='batched',bf16_kernel_mode='legacy',
                  fp8_prepared_native_weights=True,fp8_prepared_native_activations=True,
                  fp8_prepared_native_dual=True,fp8_prepared_native_gather=True)
    for key,want in expected.items():
        if value.get(key)!=want:raise ValueError('Unsupported precision gate scope: '+key)
    if value.get('pairwise'):raise ValueError('This trial retains independent rank1 experts without pair terms')


def create_run(value,seed,kind,unit,calibration=None,frozen=None):
    if stopped():raise RuntimeError('Explicit study STOP')
    validate(value);fence=assert_fence(value)
    dirty=control.git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
    if dirty:raise RuntimeError('Commit implementation/config before launch: '+dirty)
    code=control.sources()
    if frozen and code!=frozen['source_hashes']:raise RuntimeError('Verification source changed')
    rid=value['name']+'-'+kind+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(control.PYTHON),'-m','wr_reproduction.probe_precision_trial','train',str(directory),
             '--mode',calibration or 'discovery']
    head=control.git('rev-parse','HEAD')
    run=dict(study='probe-study',run_id=rid,directory=str(directory),seed=seed,kind=kind,recipe=value,
             command=command,unit=unit,created_utc=now(),source_hashes=code,
             code_commit=frozen['code_commit'] if frozen else head,launch_head_commit=head,
             environment=environment(),protocol_path=str(PROTOCOL),fence_path=str(FENCE),
             backend_hash=fence['backend_hash'],backend_fence=fence,probe_protocol=read(PROTOCOL),
             validation_data=base.validation_data(),continuous_from_initialization=not calibration,
             comparative_pruning_exempt=kind=='verification',stop_files=[str(p) for p in STOP_FILES],
             calibration=calibration,precision_scope='KS3 expert products only; five original dense MLPs and shared backend unchanged')
    atomic(directory/'run.json',run);atomic(directory/'config.json',value);atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind=kind,seed=seed,code_commit=run['code_commit']))
    control.status(run,'registered');control.note(run,hypothesis=value['hypothesis'],decision='Launch '+kind,code_commit=run['code_commit'])
    return run


def timing(rows):
    by_step={r['step']:r for r in rows if r.get('event')=='train'}
    if not all(s in by_step for s in (25,100)):raise ValueError('Missing original timing prefix')
    early,last=by_step[25]['train_seconds'],by_step[100]['train_seconds']
    if not (math.isfinite(early) and math.isfinite(last) and 0<early<last):
        raise ValueError('Invalid continuous prefix timestamps')
    return dict(first100_seconds=by_step[100]['train_seconds'],
                after25_through100_seconds=by_step[100]['train_seconds']-by_step[25]['train_seconds'],
                first100_steps=100,all_logical_ranks=8)


def speed_report(candidate,reference):
    a,b=timing(candidate),timing(reference)
    names=('first100_seconds','after25_through100_seconds')
    ratios={key:a[key]/b[key] for key in names}
    return dict(passed=all(r<1. for r in ratios.values()),candidate=a,reference=b,ratios=ratios,
                criterion='Both first100 and updates26-100 actual-model durations must be below the saved incumbent prefix.',
                limitation='Single measured original prefix, not a sampling-error estimate or convergence result; loss pruning remains elapsed-time fixed-text only.')


def train(directory,mode):
    import torch
    cfg=read(directory/'run.json');value=cfg['recipe'];validate(value)
    if stopped():raise RuntimeError('Explicit study STOP')
    if control.sources()!=cfg['source_hashes']:raise RuntimeError('Source changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    if mode=='numerical':
        from wr_reproduction.backend import install
        install('pr393')
        from wr_reproduction import ks3_fp8_gemm,ks3_moe_fp8
        ks3_fp8_gemm.TILE_MODE=value['fp8_native_gemm_tuning']
        ks3_moe_fp8.GEOMETRY_BACKWARD_ROWS=value['geometry_backward_rows']
        from wr_reproduction.check_ks3_prepared_native import check
        result=dict(passed=True,mode=mode,**check(activations=True,dual=True,native_gather=True,gs32=True,charts=1024))
        atomic(directory/'calibration.json',result);append(directory/'events.jsonl',dict(event='calibration_complete',**result))
        return
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.probe_gradient_extension import ObservedRunner
    from wr_reproduction import probe_train
    with LocalCollectives():
        r=ObservedRunner('pr393',directory/'events.jsonl',True,'fp8',directory)
        from wr_reproduction.matched_architecture import audit_model
        inventory=audit_model(r.model,'ks3_experts');assert inventory['active_conventional_mlp_instances']==5
        r.log(event='feedforward_inventory',**inventory);r.warmup()
        if mode=='discovery':
            probe_train.train(r);return
        from wr_reproduction.probe_precision_calibration import run
        run(r)
        calibration=read(directory/'calibration.json')
        ref_id=value['precision_reevaluation']['speed_reference_run_id']
        ref_run=read(STATE/'runs'/ref_id/'run.json')
        assert ref_run['seed']==cfg['seed']==169
        assert ref_run['backend_hash']==cfg['backend_hash']
        speed=speed_report(events(directory/'events.jsonl'),events(STATE/'runs'/ref_id/'events.jsonl'))
        speed['reference_run_id']=ref_id;speed['reference_code_commit']=ref_run['code_commit']
        projected=calibration['cold_probe_seconds']+6*calibration['warm_probe_seconds']
        passed=bool(calibration['passed'] and calibration['continuation_parity']['all_parameters_bitwise_equal']
                    and speed['passed'] and projected/900<=.03)
        gate=dict(passed=passed,mode='speed_state',state_calibration=calibration,speed=speed,
                  projected_seven_probe_seconds=projected,projected_fraction_of900=projected/900,
                  overhead_limitation='Projection includes measured probe preparation/scoring/restoration; screen and sparse diagnostics also remain on the actual training clock.',
                  not_a_screening_reference=True,full_convergence_verified=False)
        # Preserve the reusable helper's complete report under its own name.
        atomic(directory/'probe-state-calibration.json',calibration)
        atomic(directory/'calibration.json',gate);r.log(event='precision_speed_state_gate',**gate)


def require_gate(mode,value):
    gate=read(RESULTS/('fp8-'+mode+'-gate.json'),{})
    if not gate.get('passed'):raise RuntimeError('Missing passed FP8 '+mode+' gate')
    if gate['recipe']!=value:raise RuntimeError('Recipe changed since FP8 '+mode+' gate')
    if gate['source_hashes']!=control.sources():raise RuntimeError('Source changed since FP8 '+mode+' gate')


def work(request):
    cfg=read(request);mode=cfg['mode'];value=cfg['recipe']
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopped():return
        if mode!='numerical':require_gate('numerical',value)
        if mode=='discovery':require_gate('speed_state',value)
        run=create_run(value,169,'discovery' if mode=='discovery' else 'calibration',cfg['unit'],
                       None if mode=='discovery' else mode)
        result=control.execute(run)
        if mode!='discovery':
            gate=dict(utc=now(),run_id=run['run_id'],code_commit=run['code_commit'],recipe=value,
                source_hashes=run['source_hashes'],passed=result['status']=='calibration_complete'
                and bool((result.get('calibration') or {}).get('passed')),result=result)
            atomic(RESULTS/('fp8-'+mode+'-gate.json'),gate);atomic(STATE/('fp8-'+mode+'-gate.json'),gate)
        if result['qualifies']:
            control.create_run=create_run;control.verification(run,result,cfg['unit'])


def launch(path,mode):
    if stopped():raise RuntimeError('Explicit study STOP')
    value=read(path);validate(value)
    unit='wr-probe-precision-'+uuid.uuid4().hex[:8];request=STATE/'requests'/(unit+'.json')
    atomic(request,dict(utc=now(),unit=unit,mode=mode,recipe=value))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
        '--property=KillMode=control-group','--property=TimeoutStopSec=15',
        '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),
        str(control.PYTHON),'-m','wr_reproduction.probe_precision_trial','work',str(request)],check=True)
    print(unit,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','work','train'])
    parser.add_argument('path',type=Path);parser.add_argument('--mode',choices=['numerical','speed_state','discovery'],default='numerical')
    a=parser.parse_args()
    if a.action=='launch':launch(a.path,a.mode)
    elif a.action=='train':train(a.path,a.mode)
    else:
        try:work(a.path)
        except BaseException:
            value=dict(utc=now(),error=traceback.format_exc());atomic(STATE/'precision-blocker.json',value)
            publish('precision-status.json',dict(status='error',**value));raise
