"""One GPU lease, append-only evidence, process-group and service STOP ownership."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
import traceback
import uuid
from wr_reproduction.probe_study import *
PYTHON=ROOT/'.venv-mi300/bin/python'

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def recipe(architecture):
    value=read(ROOT/'configs/wr_reproduction/research_bf16_shared_fp8.json') if architecture!='mlp' else {}
    return dict(value,**SHARED,architecture=architecture,name='probe-'+architecture.replace('_','-'),
                hypothesis='Acquire one exempt fixed-probe reference under the frozen shared original-PR393 recipe.')

def sources():
    paths=list((ROOT/'wr_reproduction').rglob('*.py'))+list((ROOT/'ks3_research/vendor').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def status(run,state,**extra):
    value=dict(utc=now(),run_id=run['run_id'],status=state,**extra)
    atomic(Path(run['directory'])/'status.json',value);publish('latest-run.json',dict(run=run,**value))
    append(STATE/'registry.jsonl',dict(event='status',**{k:v for k,v in value.items() if k!='utc'}))

def note(run,**extra):
    for path in (STATE/'notes.jsonl',RESULTS/'notes.jsonl',Path(run['directory'])/'notes.jsonl'):
        append(path,dict(run_id=run['run_id'],**extra))

def create_run(value,seed,kind,unit,calibration=None,frozen=None):
    if stopped():raise RuntimeError('Explicit probe-study STOP')
    fence=read(FENCE) if calibration else assert_fence(value)
    if not calibration:
        dirty=git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
        if dirty:raise RuntimeError('Commit owned source/config before reference or candidate launch: '+dirty)
    protocol=read(PROTOCOL) or initial_protocol()
    code=sources()
    if frozen and code!=frozen['source_hashes']:raise RuntimeError('Verification code changed')
    rid=value['name']+'-'+kind+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(PYTHON),'-m','wr_reproduction.probe_train','--run',str(directory)]
    if calibration:command+=['--calibration',calibration]
    head=git('rev-parse','HEAD')
    run=dict(study='probe-study',run_id=rid,directory=str(directory),seed=seed,kind=kind,recipe=value,
             command=command,unit=unit,created_utc=now(),source_hashes=code,code_commit=frozen['code_commit'] if frozen else head,
             launch_head_commit=head,environment=environment(),protocol_path=str(PROTOCOL),fence_path=str(FENCE),
             backend_hash=fence['backend_hash'] if fence else 'provisional-calibration-not-screenable',
             backend_fence=fence,probe_protocol=protocol,validation_data=base.validation_data(),
             continuous_from_initialization=kind in ('control','reference','discovery','verification'),
             comparative_pruning_exempt=kind in ('control','reference','verification'),
             stop_files=[str(p) for p in STOP_FILES],calibration=calibration)
    atomic(directory/'run.json',run);atomic(directory/'config.json',value)
    atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind=kind,seed=seed,code_commit=run['code_commit']))
    status(run,'registered');note(run,hypothesis=value['hypothesis'],decision='Launch '+kind,code_commit=run['code_commit'])
    return run

def terminate(process):
    # Kill the complete process group even if its leader has already exited.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()

def qualifies(result,run):
    c=read(RESULTS/'control_set.json');v=result.get('validation')
    return bool(c and v and result['status']=='complete' and run['recipe']['architecture']!='mlp'
                and math.isfinite(v['loss']) and v['loss']<c['best']['loss'] and v['train_seconds']<=BUDGET
                and v['evaluation_signature']==c['evaluation_signature'] and run['backend_hash']==c['backend_hash'])

def execute(run):
    if stopped():raise RuntimeError('Explicit probe-study STOP')
    if not run.get('calibration'):assert_fence(run['recipe'])
    directory=Path(run['directory']);env=dict(os.environ,**run['environment'],OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',TRAIN_SEED=str(run['seed']))
    with (directory/'console.log').open('ab',buffering=0) as out:
        p=subprocess.Popen(run['command'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        publish('active.json',dict(run_id=run['run_id'],trainer_pid=p.pid,controller_pid=os.getpid(),unit=run['unit'],utc=now()))
        status(run,'running',trainer_pid=p.pid)
        next_report=time.monotonic()+120;stop=False
        try:
            while p.poll() is None:
                if stopped():stop=True;terminate(p);break
                if time.monotonic()>=next_report:
                    rows=events(directory/'events.jsonl');latest=rows[-1] if rows else None
                    publish('heartbeat.json',dict(utc=now(),run_id=run['run_id'],latest_event=latest))
                    live=read(STATE/'latest-run.json',{});live.update(heartbeat_utc=now(),latest_event=latest);publish('latest-run.json',live)
                    next_report=time.monotonic()+120
                time.sleep(.5)
        finally:terminate(p)
    rows=events(directory/'events.jsonl');vals=[r for r in rows if r.get('event')=='validation']
    val=vals[-1] if vals else None
    prune=next((r for r in rows if r.get('event') in ('screening_abort','research_pruned')),None)
    calibration=read(directory/'calibration.json')
    outcome='stopped' if stop else 'error' if p.returncode else 'pruned' if prune else 'complete' if val else 'calibration_complete' if calibration else 'incomplete'
    if val and not math.isfinite(val['loss']):val=dict(val,loss=None);outcome='error'
    result=dict(run_id=run['run_id'],status=outcome,seed=run['seed'],code_commit=run['code_commit'],recipe=run['recipe'],
                backend_hash=run['backend_hash'],validation=val,pruning=prune,calibration=calibration,exit_code=p.returncode,
                reserved_confirmation=next((r for r in rows if r.get('event')=='reserved_confirmation'),None),
                events_sha256=sha(directory/'events.jsonl') if rows else None,completed_utc=now())
    result['qualifies']=qualifies(result,run)
    atomic(directory/'result.json',result);atomic(RESULTS/(run['run_id']+'-result.json'),result)
    if rows:shutil.copyfile(directory/'events.jsonl',RESULTS/(run['run_id']+'-events.jsonl'))
    status(run,outcome,result=result);note(run,result=result,decision='Three fresh frozen seeds required' if result['qualifies'] else 'Preserve outcome and inspect next authorized hypothesis')
    publish('active.json',dict(status='idle',previous_run_id=run['run_id'],utc=now()))
    return result

def verification(run,result,unit):
    path=STATE/'verification.json';saved=read(path,dict(groups=[]));identity=digest(dict(recipe=run['recipe'],source_hashes=run['source_hashes']))
    group=next((g for g in saved['groups'] if g['identity']==identity),None)
    if group is None:
        group=dict(identity=identity,discovery=run['run_id'],status='queued',seeds=[dict(seed=s,status='queued') for s in VERIFY_SEEDS])
        saved['groups'].append(group);atomic(path,saved)
    if group['status']=='complete':return
    for item in group['seeds']:
        if stopped():return
        if item.get('run_id'):
            outcome=read(STATE/'runs'/item['run_id']/'result.json')
            if not outcome:raise RuntimeError('Interrupted verification needs reconciliation')
        else:
            fresh=create_run(run['recipe'],item['seed'],'verification',unit,frozen=run)
            item.update(run_id=fresh['run_id'],status='running');group['status']='running';atomic(path,saved);outcome=execute(fresh)
        item.update(status=outcome['status'],result=outcome);atomic(path,saved)
        if outcome['status']!='complete':raise RuntimeError('Verification error/stop requires diagnosis')
    scores=[x['result']['validation']['loss'] for x in group['seeds']]
    group.update(status='complete',statistics=dict(losses=scores,mean=statistics.mean(scores),sample_std=statistics.stdev(scores),
        minimum=min(scores),maximum=max(scores),control_loss=read(RESULTS/'control_set.json')['best']['loss']),completed_utc=now())
    atomic(path,saved);atomic(RESULTS/('verification-'+identity+'.json'),group)
    note(run,result=group['statistics'],decision='Three fresh seeds complete; continue authorized placement study')

def references(unit):
    progress=read(STATE/'references.json',dict(arms=[]))
    for architecture,kind in (('mlp','control'),('ks3_experts','reference')):
        item=next((x for x in progress['arms'] if x['architecture']==architecture),None)
        if item:
            run=read(STATE/'runs'/item['run_id']/'run.json');result=read(STATE/'runs'/item['run_id']/'result.json')
            if not result:raise RuntimeError('Interrupted reference must be reconciled; never silently repeat')
        else:
            run=create_run(recipe(architecture),169,kind,unit)
            item=dict(architecture=architecture,run_id=run['run_id'],status='running');progress['arms'].append(item)
            atomic(STATE/'references.json',progress);result=execute(run)
        item.update(status=result['status'],result=result);atomic(STATE/'references.json',progress)
        if result['status']!='complete':raise RuntimeError('Reference failed: '+run['run_id'])
        if architecture=='mlp' and not (RESULTS/'control_set.json').exists():
            v=result['validation'];assert v['train_seconds']<=BUDGET
            control=dict(utc=now(),immutable=True,one_seed_intentional=True,best=dict(run_id=run['run_id'],loss=v['loss'],train_seconds=v['train_seconds']),
                         evaluation_signature=v['evaluation_signature'],backend_hash=run['backend_hash'],
                         historical_controls_separate=True,probe_overhead_on_clock=True)
            atomic(RESULTS/'control_set.json',control);atomic(STATE/'control_set.json',control)
        if result['qualifies']:verification(run,result,unit)
    atomic(STATE/'phase.json',dict(phase='references_complete_placement_batch_next',utc=now()))

def work(request):
    cfg=read(request)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopped():return
        if cfg['mode']=='references':references(cfg['unit'])
        else:
            run=create_run(cfg['recipe'],cfg.get('seed',169),cfg['kind'],cfg['unit'],cfg.get('calibration'))
            result=execute(run)
            if result['qualifies']:verification(run,result,cfg['unit'])

def launch(mode,*,value=None,kind='discovery',calibration=None):
    if stopped():raise RuntimeError('Explicit probe-study STOP')
    unit='wr-probe-'+mode+'-'+uuid.uuid4().hex[:8];path=STATE/'requests'/(unit+'.json')
    atomic(path,dict(mode=mode,unit=unit,recipe=value,kind=kind,calibration=calibration,seed=169,utc=now(),environment=environment()))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
                    '--property=KillMode=control-group','--property=TimeoutStopSec=15',
                    '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),str(PYTHON),'-m','wr_reproduction.probe_controller','work',str(path)],check=True)
    print(unit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['work','calibration','references']);p.add_argument('request',nargs='?',type=Path)
    p.add_argument('--architecture',default='mlp');p.add_argument('--calibration-mode',default='probe',choices=['baseline','probe','precision']);a=p.parse_args()
    if a.action=='work':
        try:work(a.request)
        except BaseException:
            atomic(STATE/'blocker.json',dict(utc=now(),error=traceback.format_exc()));raise
    elif a.action=='calibration':launch('calibration',value=recipe(a.architecture),kind='calibration',calibration=a.calibration_mode)
    else:launch('references')
