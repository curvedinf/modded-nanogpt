"""Detached sequential controls and research for the fenced matched study."""
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
from wr_reproduction.matched_study import (ROOT,STATE,RESULTS,FENCE,BUDGET,STOP_FILES,
    Metadata,assert_fence,stopped,atomic,append,read,sha,digest,now,events)

PYTHON=ROOT/'.venv-mi300/bin/python'

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def status(run,value,**extra):
    row=dict(run_id=run['run_id'],status=value,utc=now(),**extra)
    atomic(Path(run['directory'])/'status.json',row)
    atomic(STATE/'latest-run.json',dict(run=run,**row))
    append(STATE/'registry.jsonl',dict(event='status',**{k:v for k,v in row.items() if k!='utc'}))

def note(run,**extra):
    for path in (STATE/'notes.jsonl',RESULTS/'notes.jsonl',Path(run['directory'])/'notes.jsonl'):
        append(path,dict(run_id=run['run_id'],**extra))

def source_hashes():
    paths=list((ROOT/'wr_reproduction').rglob('*.py'))+list((ROOT/'ks3_research/vendor').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def create_run(recipe,seed,kind,unit,frozen=None):
    if stopped():raise RuntimeError('Explicit matched-study STOP')
    fence=assert_fence(recipe);metadata=Metadata();hashes=source_hashes()
    if frozen and hashes!=frozen['source_hashes']:raise RuntimeError('Verification source hashes differ from frozen discovery')
    rid=recipe['name']+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(PYTHON),'-m','wr_reproduction.train','--version','pr393',
             '--architecture',recipe['architecture'],'--max-train-seconds',str(BUDGET),
             '--research-run',str(directory),'--out',str(directory/'events.jsonl')]
    if recipe['architecture']!='mlp':command+=['--ks3-precision',recipe['precision']]
    if kind=='timing':command+=['--timing-updates','100']
    if kind=='correctness':
        oracle=('wr_reproduction.check_ks3_feedforward_bf16_direct' if recipe.get('dense_precision','bf16')=='bf16' and recipe.get('dense_training_gs32',True) is False else
                'wr_reproduction.check_ks3_feedforward_fp8' if recipe.get('dense_precision')=='fp8' else
                'wr_reproduction.check_ks3_feedforward' if recipe['rank']==1 else 'wr_reproduction.check_ks3_feedforward_rank')
        command=[str(PYTHON),'-m',oracle,'--out',str(directory/'events.jsonl'),
                 '--features',str(recipe.get('dense_charts',recipe['charts'])*recipe['rank'])]
        if recipe['rank']!=1:command+=['--rank',str(recipe['rank'])]
        if recipe.get('dense_precision')=='fp8' and recipe.get('dense_training_gs32',True) is False:
            command+=['--dense-native-pack']
    dirty=git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
    if dirty:raise RuntimeError('Commit owned code/config before launch: '+dirty)
    run=dict(study='matched-study',run_id=rid,directory=str(directory),created_utc=now(),kind=kind,
             seed=seed,recipe=recipe,continuous_from_initialization=kind in ('control','discovery','verification'),
             code_commit=frozen['code_commit'] if frozen else git('rev-parse','HEAD'),
             launch_head_commit=git('rev-parse','HEAD'),source_hashes=hashes,backend_hash=fence['backend_hash'],
             backend_fence=fence,command=command,unit=unit,validation_data=metadata.data,
             expected_evaluation_signature=metadata.validation(595),stop_files=[str(p) for p in STOP_FILES],
             comparative_pruning_exempt=kind in ('control','verification'),
             reference_policy='Only registered matched-study continuous curves; no historical curve import')
    atomic(directory/'run.json',run);atomic(directory/'config.json',recipe)
    atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),seed=seed,kind=kind,code_commit=run['code_commit']))
    status(run,'registered');note(run,hypothesis=recipe['hypothesis'],decision='Launch '+kind,seed=seed,code_commit=run['code_commit'])
    return run

def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        except ProcessLookupError:pass

def qualifies(result,run):
    control=read(RESULTS/'control_set.json')
    val=result.get('validation')
    return bool(control and result['status']=='complete' and val and math.isfinite(val['loss'])
                and val['train_seconds']<=BUDGET and val['loss']<control['best']['loss']
                and val['evaluation_signature']==control['evaluation_signature']
                and run['backend_hash']==control['backend_hash'] and run['recipe']['architecture']!='mlp')

def execute(run):
    if stopped():raise RuntimeError('Explicit matched-study STOP')
    assert_fence(run['recipe']);directory=Path(run['directory'])
    env=dict(os.environ,TRAIN_SEED=str(run['seed']),OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in env:raise RuntimeError('Forbidden inherited override: '+key)
    with (directory/'console.log').open('ab',buffering=0) as out:
        process=subprocess.Popen(run['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                 stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        active=dict(run_id=run['run_id'],controller_pid=os.getpid(),trainer_pid=process.pid,unit=run['unit'],utc=now())
        atomic(STATE/'active.json',active);status(run,'running',trainer_pid=process.pid)
        was_stopped=False
        try:
            while process.poll() is None:
                if stopped():was_stopped=True;terminate(process);break
                time.sleep(.5)
        finally:terminate(process)
    rows=events(directory/'events.jsonl')
    vals=[r for r in rows if r.get('event')=='validation']
    val=vals[-1] if vals else None
    prune=next((r for r in rows if r.get('event') in ('screening_abort','research_pruned')),None)
    timing=next((r for r in rows if r.get('event')=='timing_complete'),None)
    correctness=next((r for r in rows if r.get('event')=='kernel_correctness'),None)
    state='stopped' if was_stopped else ('error' if process.returncode else ('pruned' if prune else ('complete' if val else ('timing_complete' if timing else 'incomplete'))))
    if correctness and correctness.get('passed') and not process.returncode and not was_stopped:state='correctness_passed'
    if val and not math.isfinite(val['loss']):val=dict(val,loss=None);state='error'
    result=dict(run_id=run['run_id'],status=state,seed=run['seed'],code_commit=run['code_commit'],
                recipe=run['recipe'],backend_hash=run['backend_hash'],validation=val,pruning=prune,timing=timing,
                correctness=correctness,exit_code=process.returncode,events_sha256=sha(directory/'events.jsonl') if rows else None,completed_utc=now())
    result['qualifies']=qualifies(result,run)
    atomic(directory/'result.json',result);atomic(RESULTS/(run['run_id']+'-result.json'),result)
    if rows:shutil.copyfile(directory/'events.jsonl',RESULTS/(run['run_id']+'-events.jsonl'))
    status(run,state,result=result);note(run,result=result,decision='Queue three frozen fresh seeds' if result['qualifies'] else 'Record and continue requested sequence')
    atomic(STATE/'active.json',dict(status='idle',previous_run_id=run['run_id'],utc=now()))
    append(RESULTS/'milestones.jsonl',dict(milestone='run_result',run_id=run['run_id'],result_hash=digest(result),code_commit=run['code_commit']))
    return result

def freeze_controls(results):
    if len(results)!=3 or any(r['status']!='complete' for r in results):raise RuntimeError('Three complete matched controls required')
    points=[]
    for r in results:
        v=r['validation']
        if v['train_seconds']>BUDGET:raise RuntimeError('Control budget overrun; preserve result and audit timing before reference freeze')
        points.append(dict(run_id=r['run_id'],seed=r['seed'],loss=v['loss'],train_seconds=v['train_seconds'],code_commit=r['code_commit'],events_sha256=r['events_sha256']))
    signatures=[r['validation']['evaluation_signature'] for r in results]
    if not all(s==signatures[0] for s in signatures):raise RuntimeError('Controls have incompatible evaluation signatures')
    if signatures[0]['attention_windows']!=[640,1408]:raise RuntimeError('Wrong evaluation state')
    value=dict(utc=now(),immutable=True,nominal_training_seconds=BUDGET,backend_hash=results[0]['backend_hash'],
               controls=points,best=min(points,key=lambda p:p['loss']),evaluation_signature=signatures[0],
               mean=statistics.mean(p['loss'] for p in points),sample_std=statistics.stdev(p['loss'] for p in points),
               objective='Strictly lower full-validation loss than the best measured matched MLP within nominal900 timed seconds')
    for path in (RESULTS/'control_set.json',STATE/'control_set.json'):
        if path.exists():raise RuntimeError('Control set already frozen')
        atomic(path,value)
    atomic(STATE/'phase.json',dict(phase='controls_complete_awaiting_verified_full_feedforward',utc=now()))
    return value

def verification(run,result,unit):
    """Three fresh runs from a frozen recipe/source; never comparative-pruned."""
    path=STATE/'verification.json';saved=read(path,dict(groups=[]))
    identity=digest(dict(recipe=run['recipe'],source_hashes=run['source_hashes']))
    group=next((g for g in saved['groups'] if g['recipe_id']==identity),None)
    if group is None:
        group=dict(recipe_id=identity,discovery_run=run,discovery_result=result,status='queued',
                   seeds=[dict(seed=s,status='queued') for s in (269,270,271)])
        saved['groups'].append(group);atomic(path,saved)
    if group['status']=='complete':return
    group['status']='running';atomic(path,saved)
    for item in group['seeds']:
        if stopped():return
        if item.get('run_id'):
            outcome=read(STATE/'runs'/item['run_id']/'result.json')
            if not outcome:raise RuntimeError('Interrupted verification needs reconciliation: '+item['run_id'])
        else:
            fresh=create_run(run['recipe'],item['seed'],'verification',unit,frozen=run)
            item.update(run_id=fresh['run_id'],status='running');atomic(path,saved)
            outcome=execute(fresh)
        item.update(status=outcome['status'],result=outcome);atomic(path,saved)
        if outcome['status']=='stopped':return
    completed=[dict(seed=s['seed'],run_id=s['run_id'],loss=s['result']['validation']['loss'],
                    train_seconds=s['result']['validation']['train_seconds']) for s in group['seeds'] if s['status']=='complete']
    group['status']='complete' if len(completed)==3 else 'complete_with_errors'
    control=read(RESULTS/'control_set.json')
    group['statistics']=dict(scores=completed,count=len(completed),
         mean=statistics.mean(v['loss'] for v in completed) if completed else None,
         sample_std=statistics.stdev(v['loss'] for v in completed) if len(completed)>1 else None,
         best_matched_mlp_loss=control['best']['loss'],matched_mlp_mean=control['mean'],matched_mlp_sample_std=control['sample_std'],
         count_beating_best_control=sum(v['loss']<control['best']['loss'] and v['train_seconds']<=BUDGET for v in completed))
    group['completed_utc']=now();atomic(path,saved)
    atomic(RESULTS/('verification-'+identity+'.json'),group)
    note(run,result=group['statistics'],decision='Three-seed report complete; continue KS3 architecture research')

def trial(recipe,seed,unit,kind='discovery'):
    if kind=='discovery' and not read(RESULTS/'control_set.json'):raise RuntimeError('Matched controls must complete first')
    run=create_run(recipe,seed,kind,unit);result=execute(run)
    if result['qualifies']:verification(run,result,unit)
    return result

def initial_research(unit):
    """First all-KS3, then the three requested edge-layer hybrids, resumably."""
    if not read(RESULTS/'control_set.json'):raise RuntimeError('Matched control set is not frozen')
    phase=read(STATE/'initial-research.json',dict(correctness=None,arms=[]))
    first=read(ROOT/'configs/wr_reproduction/matched_ks3_all.json')
    if phase['correctness'] is None:
        result=trial(first,172,unit,'correctness')
        phase['correctness']=result;atomic(STATE/'initial-research.json',phase)
    if phase['correctness']['status']!='correctness_passed':raise RuntimeError('Dense KS3 oracle did not pass')
    for architecture in ('ks3_all','ks3_first_mlp','ks3_last_mlp','ks3_both_mlp'):
        if stopped():return
        item=next((x for x in phase['arms'] if x['architecture']==architecture),None)
        if item:
            run=read(STATE/'runs'/item['run_id']/'run.json')
            result=read(STATE/'runs'/item['run_id']/'result.json')
            if not result:raise RuntimeError('Interrupted architecture run needs reconciliation: '+item['run_id'])
        else:
            recipe=read(ROOT/('configs/wr_reproduction/matched_'+architecture+'.json'))
            run=create_run(recipe,168,'discovery',unit)
            item=dict(architecture=architecture,run_id=run['run_id'],status='running')
            phase['arms'].append(item);atomic(STATE/'initial-research.json',phase)
            result=execute(run)
        item.update(status=result['status'],result=result);atomic(STATE/'initial-research.json',phase)
        if result['status'] in ('error','incomplete','stopped'):raise RuntimeError('Architecture trial needs diagnosis: '+result['run_id'])
        if result['qualifies']:verification(run,result,unit)
    atomic(STATE/'phase.json',dict(phase='initial_architectures_complete_continue_meaningful_ks3_ablations',utc=now()))

def controls(unit):
    recipe=read(ROOT/'configs/wr_reproduction/matched_mlp.json')
    progress=read(STATE/'controls.json',dict(seeds=[dict(seed=s,status='queued') for s in (169,170,171)]))
    for item in progress['seeds']:
        if stopped():return
        if item.get('run_id'):
            result=read(STATE/'runs'/item['run_id']/'result.json')
            if not result:raise RuntimeError('Interrupted run needs reconciliation: '+item['run_id'])
        else:
            run=create_run(recipe,item['seed'],'control',unit)
            item.update(run_id=run['run_id'],status='running');atomic(STATE/'controls.json',progress)
            result=execute(run)
        item.update(status=result['status'],result=result);atomic(STATE/'controls.json',progress)
        if result['status']!='complete':raise RuntimeError('Control did not complete: '+item['run_id'])
    freeze_controls([x['result'] for x in progress['seeds']])

def work(request):
    request=read(request)
    STATE.mkdir(parents=True,exist_ok=True)
    transition=None
    if request['mode']=='after-controls':
        transition=(STATE/'transition.lock').open('a')
        fcntl.flock(transition,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(STATE/'transition.json',dict(status='waiting_for_all_matched_controls',unit=request['unit'],utc=now()))
        while not (RESULTS/'control_set.json').exists():
            if stopped():return
            progress=read(STATE/'controls.json',dict(seeds=[]))
            if any(s['status'] in ('error','incomplete','stopped') for s in progress['seeds']):
                raise RuntimeError('Control failure blocks automatic research transition')
            time.sleep(1)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if request['mode']!='after-controls':raise
                if stopped():return
                time.sleep(.5)
        if stopped():return
        if request['mode']=='controls':controls(request['unit'])
        elif request['mode'] in ('initial-research','after-controls'):
            atomic(STATE/'transition.json',dict(status='gpu_oracle_then_all_ks3_and_edge_hybrids',unit=request['unit'],utc=now()))
            initial_research(request['unit'])
        else:
            trial(request['recipe'],request['seed'],request['unit'],request.get('kind','discovery'))

def launch(mode,recipe=None,seed=168,kind='discovery'):
    if stopped():raise RuntimeError('Explicit matched-study STOP')
    STATE.mkdir(parents=True,exist_ok=True)
    if mode!='after-controls':
        with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    unit='wr-matched-'+mode+'-'+uuid.uuid4().hex[:10]
    request=STATE/'requests'/(unit+'.json')
    atomic(request,dict(mode=mode,recipe=read(recipe) if recipe else None,seed=seed,kind=kind,unit=unit,utc=now()))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
                    '--property=KillMode=control-group','--property=TimeoutStopSec=15',
                    '--property=PartOf=wr-ks3-rebaseline.service',str(PYTHON),'-m',
                    'wr_reproduction.matched_controller','work',str(request)],check=True)
    print(unit,flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    l=sub.add_parser('launch');l.add_argument('mode',choices=['controls','trial','initial-research','after-controls']);l.add_argument('--recipe',type=Path);l.add_argument('--seed',type=int,default=168)
    w=sub.add_parser('work');w.add_argument('request',type=Path)
    a=p.parse_args()
    if a.action=='launch':launch(a.mode,a.recipe,a.seed)
    else:
        try:work(a.request)
        except BaseException:
            append(STATE/'controller-errors.jsonl',dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
