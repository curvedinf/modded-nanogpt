"""KS3-only BF16 gather fusion: numerical, actual-model speed/state and loss gates.

Reuses the frozen probe math, actual-model calibration and process owner.
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
    expected=dict(architecture='ks3_experts',precision='bf16',charts=1024,rank=1,
                  training_gs32=True,weight_packing='loop',bf16_kernel_mode='tuned',
                  ks3_bf16_gradient_gather='fused-v2-exact-sum')
    for key,want in expected.items():
        if value.get(key)!=want:raise ValueError('Unsupported BF16 gather scope: '+key)
    if value.get('pairwise'):raise ValueError('Independent incumbent rank1 experts required')


def create_run(value,seed,kind,unit,calibration=None,frozen=None):
    if stopped():raise RuntimeError('Explicit study STOP')
    validate(value);fence=assert_fence(value)
    dirty=control.git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
    if dirty:raise RuntimeError('Commit implementation/config before launch: '+dirty)
    code=control.sources()
    if frozen and code!=frozen['source_hashes']:raise RuntimeError('Verification source changed')
    rid=value['name']+'-'+kind+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(control.PYTHON),'-m','wr_reproduction.probe_bf16_gather_trial','train',str(directory),
             '--mode',calibration or 'discovery']
    head=control.git('rev-parse','HEAD')
    run=dict(study='probe-study',run_id=rid,directory=str(directory),seed=seed,kind=kind,recipe=value,
             command=command,unit=unit,created_utc=now(),source_hashes=code,
             code_commit=frozen['code_commit'] if frozen else head,launch_head_commit=head,
             environment=environment(),protocol_path=str(PROTOCOL),fence_path=str(FENCE),
             backend_hash=fence['backend_hash'],backend_fence=fence,probe_protocol=read(PROTOCOL),
             validation_data=base.validation_data(),continuous_from_initialization=not calibration,
             comparative_pruning_exempt=kind=='verification',stop_files=[str(p) for p in STOP_FILES],
             calibration=calibration,kernel_scope='KS3 BF16 gradient gather only; all products, five original dense MLPs and shared backend unchanged')
    atomic(directory/'run.json',run);atomic(directory/'config.json',value);atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind=kind,seed=seed,code_commit=run['code_commit']))
    control.status(run,'registered');control.note(run,hypothesis=value['hypothesis'],decision='Launch '+kind,code_commit=run['code_commit'])
    return run


from wr_reproduction.probe_precision_trial import speed_report


def train(directory,mode):
    import torch
    cfg=read(directory/'run.json');value=cfg['recipe'];validate(value)
    if stopped():raise RuntimeError('Explicit study STOP')
    if control.sources()!=cfg['source_hashes']:raise RuntimeError('Source changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    if mode=='numerical':
        from wr_reproduction.check_ks3_bf16_gather import check
        result=dict(mode=mode,**check())
        atomic(directory/'calibration.json',result);append(directory/'events.jsonl',dict(event='calibration_complete',**result))
        return
    from wr_reproduction.ks3_bf16_gather import install
    install()
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.probe_gradient_extension import ObservedRunner
    from wr_reproduction import probe_train
    with LocalCollectives():
        r=ObservedRunner('pr393',directory/'events.jsonl',True,'bf16',directory)
        from wr_reproduction.matched_architecture import audit_model
        inventory=audit_model(r.model,'ks3_experts');assert inventory['active_conventional_mlp_instances']==5
        r.log(event='feedforward_inventory',**inventory);r.warmup()
        if mode=='discovery':
            probe_train.train(r);return
        from wr_reproduction.probe_precision_calibration import run
        run(r)
        calibration=read(directory/'calibration.json')
        ref_id=value['gather_fusion']['speed_reference_run_id']
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
        atomic(directory/'calibration.json',gate);r.log(event='bf16_gather_speed_state_gate',**gate)


def require_gate(mode,value):
    gate=read(RESULTS/('bf16-gather-'+mode+'-gate.json'),{})
    if not gate.get('passed'):raise RuntimeError('Missing passed BF16 gather '+mode+' gate')
    if gate['recipe']!=value:raise RuntimeError('Recipe changed since BF16 gather '+mode+' gate')
    if gate['source_hashes']!=control.sources():raise RuntimeError('Source changed since BF16 gather '+mode+' gate')


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
            atomic(RESULTS/('bf16-gather-'+mode+'-gate.json'),gate);atomic(STATE/('bf16-gather-'+mode+'-gate.json'),gate)
        if result['qualifies']:
            control.create_run=create_run;control.verification(run,result,cfg['unit'])


def launch(path,mode):
    if stopped():raise RuntimeError('Explicit study STOP')
    value=read(path);validate(value)
    unit='wr-probe-bf16-gather-'+uuid.uuid4().hex[:8];request=STATE/'requests'/(unit+'.json')
    atomic(request,dict(utc=now(),unit=unit,mode=mode,recipe=value))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
        '--property=KillMode=control-group','--property=TimeoutStopSec=15',
        '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),
        str(control.PYTHON),'-m','wr_reproduction.probe_bf16_gather_trial','work',str(request)],check=True)
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
            value=dict(utc=now(),error=traceback.format_exc());atomic(STATE/'bf16-gather-blocker.json',value)
            publish('bf16-gather-status.json',dict(status='error',**value));raise
