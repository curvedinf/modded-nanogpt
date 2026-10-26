"""Small KS3-only requests using frozen V2 CE and an audited gradient extension."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import *


def create_run(value, seed, kind, unit, calibration=None, frozen=None):
    if stopped():raise RuntimeError('Explicit probe-study STOP')
    fence=assert_fence(value)
    dirty=control.git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
    if dirty:raise RuntimeError('Commit owned implementation/config before launch: '+dirty)
    code=control.sources()
    if frozen and code!=frozen['source_hashes']:raise RuntimeError('Verification source changed')
    if value.get('gradient_diagnostics_extension')!='active-slot-pre-optimizer-v1':
        raise ValueError('Explicit diagnostic extension required in recipe')
    rid=value['name']+'-'+kind+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(control.PYTHON),'-m','wr_reproduction.probe_gradient_extension','--run',str(directory)]
    if calibration:command+=['--gate']
    head=control.git('rev-parse','HEAD')
    run=dict(study='probe-study',run_id=rid,directory=str(directory),seed=seed,kind=kind,recipe=value,
             command=command,unit=unit,created_utc=now(),source_hashes=code,
             code_commit=frozen['code_commit'] if frozen else head,launch_head_commit=head,
             environment=environment(),protocol_path=str(PROTOCOL),fence_path=str(FENCE),
             backend_hash=fence['backend_hash'],backend_fence=fence,probe_protocol=read(PROTOCOL),
             validation_data=base.validation_data(),continuous_from_initialization=not calibration,
             comparative_pruning_exempt=kind=='verification',stop_files=[str(p) for p in STOP_FILES],
             calibration=calibration,diagnostic_extension=dict(name='active-slot-pre-optimizer-v1',
             shared_math_unchanged=True,frozen_probe_code_unchanged=True,
             additional_cost_on_continuous_clock=True,source_files={k:v for k,v in code.items()
                if k in ('wr_reproduction/ks3_update_observer.py','wr_reproduction/probe_gradient_extension.py','wr_reproduction/probe_directed_controller.py')}))
    atomic(directory/'run.json',run);atomic(directory/'config.json',value)
    atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind=kind,seed=seed,code_commit=run['code_commit']))
    control.status(run,'registered');control.note(run,hypothesis=value['hypothesis'],decision='Launch '+kind,code_commit=run['code_commit'])
    return run


def work(request):
    cfg=read(request)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopped():return
        if cfg['kind']=='discovery':
            gate=read(RESULTS/'gradient-extension-gate.json',{})
            if not gate.get('passed'):raise RuntimeError('Gradient extension GPU gate required')
            for path,expected in gate['extension_source_hashes'].items():
                if sha(ROOT/path)!=expected:raise RuntimeError('Extension changed since GPU gate: '+path)
        run=create_run(cfg['recipe'],cfg.get('seed',169),cfg['kind'],cfg['unit'],cfg.get('calibration'))
        result=control.execute(run)
        if cfg.get('calibration'):
            value=dict(utc=now(),run_id=run['run_id'],code_commit=run['code_commit'],
                       passed=result['status']=='calibration_complete' and bool(result.get('calibration',{}).get('passed')),
                       extension_source_hashes=run['diagnostic_extension']['source_files'],result=result)
            atomic(RESULTS/'gradient-extension-gate.json',value);atomic(STATE/'gradient-extension-gate.json',value)
        if result['qualifies']:
            # The existing verification state machine now uses the same extension
            # entry point and exact hashes for all three frozen fresh seeds.
            control.create_run=create_run
            control.verification(run,result,cfg['unit'])


def launch(recipe_path, gate=False):
    if stopped():raise RuntimeError('Explicit probe-study STOP')
    unit='wr-probe-directed-'+uuid.uuid4().hex[:8]
    path=STATE/'requests'/(unit+'.json');value=read(recipe_path)
    atomic(path,dict(unit=unit,recipe=value,kind='calibration' if gate else 'discovery',
                    calibration='gradient-extension' if gate else None,seed=169,utc=now()))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
                    '--property=KillMode=control-group','--property=TimeoutStopSec=15',
                    '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),
                    str(control.PYTHON),'-m','wr_reproduction.probe_directed_controller','work',str(path)],check=True)
    print(unit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','work']);p.add_argument('path',type=Path);p.add_argument('--gate',action='store_true');a=p.parse_args()
    if a.action=='launch':launch(a.path,a.gate)
    else:
        try:work(a.path)
        except BaseException:
            value=dict(utc=now(),error=traceback.format_exc());atomic(STATE/'directed-blocker.json',value);publish('directed-status.json',dict(status='error',**value));raise
