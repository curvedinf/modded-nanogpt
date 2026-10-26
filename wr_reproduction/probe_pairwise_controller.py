"""One explicit pairwise request; existing lease, STOP, registry and verification."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import *


PAIR_FILES=('wr_reproduction/ks3_pairwise.py','wr_reproduction/probe_pairwise_train.py',
            'wr_reproduction/probe_pairwise_controller.py','wr_reproduction/check_ks3_pairwise.py')


def create_run(value,seed,kind,unit,calibration=None,frozen=None):
    if stopped():raise RuntimeError('Explicit study STOP')
    fence=assert_fence(value)
    if value['architecture']!='ks3_experts' or value.get('rank')!=1 or value.get('precision')!='bf16':
        raise ValueError('This pairwise cohort retains all five original dense MLPs and rank1 BF16 experts')
    dirty=control.git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
    if dirty:raise RuntimeError('Commit owned implementation/config before launch: '+dirty)
    code=control.sources()
    if frozen and code!=frozen['source_hashes']:raise RuntimeError('Verification source changed')
    rid=value['name']+'-'+kind+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    module='wr_reproduction.check_ks3_pairwise' if calibration=='pairwise-numerical' else 'wr_reproduction.probe_pairwise_train'
    command=[str(control.PYTHON),'-m',module,'--run',str(directory)]
    if calibration=='pairwise-graph':command+=['--gate']
    head=control.git('rev-parse','HEAD')
    run=dict(study='probe-study',run_id=rid,directory=str(directory),seed=seed,kind=kind,recipe=value,
             command=command,unit=unit,created_utc=now(),source_hashes=code,
             code_commit=frozen['code_commit'] if frozen else head,launch_head_commit=head,
             environment=environment(),protocol_path=str(PROTOCOL),fence_path=str(FENCE),
             backend_hash=fence['backend_hash'],backend_fence=fence,probe_protocol=read(PROTOCOL),
             validation_data=base.validation_data(),continuous_from_initialization=not calibration,
             comparative_pruning_exempt=kind=='verification',stop_files=[str(p) for p in STOP_FILES],
             calibration=calibration,pairwise_extension=dict(source_files={p:code[p] for p in PAIR_FILES if p in code},
             auxiliary_oracle_hashes={p:sha(ROOT/p) for p in ('ks3_research/pairwise_reference.py','ks3_research/check_pairwise_reference.py')},
             shared_math_unchanged=True,frozen_probe_code_unchanged=True,additional_cost_on_continuous_clock=True))
    atomic(directory/'run.json',run);atomic(directory/'config.json',value);atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind=kind,seed=seed,code_commit=run['code_commit']))
    control.status(run,'registered');control.note(run,hypothesis=value['hypothesis'],decision='Launch '+kind,code_commit=run['code_commit'])
    return run


def require_gate(name,files,recipe):
    gate=read(RESULTS/(name+'-gate.json'),{})
    if not gate.get('passed'):raise RuntimeError('Missing passed '+name+' gate')
    if gate['result']['calibration'].get('pair_placement','before-GS32')!=recipe['pairwise'].get('placement','before-GS32'):
        raise RuntimeError('Pair placement changed since '+name+' gate')
    for path in files:
        if sha(ROOT/path)!=gate['source_files'].get(path):raise RuntimeError('Source changed since '+name+' gate: '+path)


def work(request):
    cfg=read(request)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopped():return
        mode=cfg.get('calibration')
        if mode!='pairwise-numerical':require_gate('pairwise-numerical',('wr_reproduction/ks3_pairwise.py','wr_reproduction/check_ks3_pairwise.py'),cfg['recipe'])
        if not mode:require_gate('pairwise-graph',PAIR_FILES,cfg['recipe'])
        run=create_run(cfg['recipe'],cfg.get('seed',169),cfg['kind'],cfg['unit'],mode)
        result=control.execute(run)
        if mode:
            value=dict(utc=now(),run_id=run['run_id'],code_commit=run['code_commit'],
                       passed=result['status']=='calibration_complete' and bool((result.get('calibration') or {}).get('passed')),
                       source_files=run['pairwise_extension']['source_files'],result=result)
            atomic(RESULTS/(mode+'-gate.json'),value);atomic(STATE/(mode+'-gate.json'),value)
        if result['qualifies']:
            control.create_run=create_run;control.verification(run,result,cfg['unit'])


def launch(recipe_path,mode=None):
    if stopped():raise RuntimeError('Explicit study STOP')
    if mode not in (None,'pairwise-numerical','pairwise-graph'):raise ValueError(mode)
    unit='wr-probe-pairwise-'+uuid.uuid4().hex[:8];path=STATE/'requests'/(unit+'.json')
    atomic(path,dict(unit=unit,recipe=read(recipe_path),kind='calibration' if mode else 'discovery',calibration=mode,seed=169,utc=now()))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
                    '--property=KillMode=control-group','--property=TimeoutStopSec=15',
                    '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),
                    str(control.PYTHON),'-m','wr_reproduction.probe_pairwise_controller','work',str(path)],check=True)
    print(unit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','work']);p.add_argument('path',type=Path)
    p.add_argument('--mode',choices=['pairwise-numerical','pairwise-graph']);a=p.parse_args()
    if a.action=='launch':launch(a.path,a.mode)
    else:
        try:work(a.path)
        except BaseException:
            value=dict(utc=now(),error=traceback.format_exc());atomic(STATE/'pairwise-blocker.json',value)
            publish('pairwise-status.json',dict(status='error',**value));raise
