"""Detached KS3 run registry, watchdog, timestamped notes, and independent status CLI."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

if __package__:
    from .screening import ReferenceCurves, assess, load_policy, point
else:
    from screening import ReferenceCurves, assess, load_policy, point

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT/'logs/ks3-autoresearch'
SELF = Path(__file__).resolve()
TARGET = 3.68
SERVICE = 'ks3-codex-research.service'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


@contextmanager
def lock(name):
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/name).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def events(path):
    try:
        lines=Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    result=[]
    for line in lines:
        try:
            value=json.loads(line)
            if isinstance(value,dict):
                result.append(value)
        except json.JSONDecodeError:
            pass
    return result


def note(run_id, phase, message):
    message=' '.join(message.split())[:1000]
    entry=dict(datetime_utc=now(),run_id=run_id,phase=phase,note=message)
    with lock('notes.lock'):
        for path in (STATE/'notes.jsonl',STATE/'runs'/run_id/'notes.jsonl'):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as handle:
                handle.write(json.dumps(entry)+'\n')
        with (STATE/'notes.log').open('a') as handle:
            handle.write(f"{entry['datetime_utc']} {run_id} [{phase}] {message}\n")
    return entry


def unit_status(unit):
    result=subprocess.run(['systemctl','show',unit,'--property=ActiveState,SubState,MainPID,Result'],
                          capture_output=True,text=True)
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)


def alive_unit(unit):
    info=unit_status(unit)
    return info.get('ActiveState') in ('activating','active','reloading')


def stop_marker():
    return next((p for p in (STATE/'STOP', ROOT/'logs/control393/STOP') if p.exists()), None)


def require_control_cohort(config):
    if config.get('architecture') != 'pr393-moe-ks3':
        return
    cohort = read(ROOT/'configs/control393/control_set.json', {})
    controls = cohort.get('controls', [])
    if (cohort.get('status') != 'completed' or len(controls) != 3
            or len({c.get('seed') for c in controls}) != 3):
        raise RuntimeError('Three independent completed PR393 controls are required')
    for control in controls:
        directory = Path(control['directory'])/'artifacts'
        result = read(directory/'result.json', {})
        recipe = read(directory/'config.json', {})
        if not (result.get('exit_code') == 0 and result.get('full_validation') is True
                and result.get('validation_tokens') == 10485760
                and result.get('val_loss', math.inf) <= cohort['target']
                and recipe.get('architecture') == 'pr393-moe'
                and recipe.get('seed') == control['seed']):
            raise RuntimeError('Each PR393 control must have a successful measured full target')
    if config.get('screen_seconds'):
        raise RuntimeError('KS3 trials cannot inherit bounded control-tuning termination')


def require_attention_correction(config,kind):
    correction=read(ROOT/'configs/control393/attention_correction.json',{})
    if (kind=='trial' and config.get('architecture')=='pr393-moe-ks3'
            and correction.get('status')=='verified'):
        if config.get('attention_mask_semantics')!=correction['corrected_training_semantics']:
            raise RuntimeError('Known grouped-attention mask defect: full trials require the verified corrected attention semantics; see configs/control393/attention_correction.json')


def start(options):
    marker = stop_marker()
    if marker is not None:
        raise RuntimeError(f'Research is stopped by {marker}; explicit user resumption is required')
    config=Path(options.config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    config_data = read(config)
    if config_data.get('architecture') == 'pr393-moe-ks3' and config_data.get('target_val_loss') != TARGET:
        raise RuntimeError('Current MoE KS3 target is 3.68; update the stale candidate configuration')
    require_control_cohort(config_data)
    require_attention_correction(config_data,options.kind)
    active_controls = read(ROOT/'configs/control393/control_set.json', {})
    if active_controls.get('status') == 'completed' and config_data.get('architecture') != 'pr393-moe-ks3':
        raise RuntimeError('The active study requires the matched PR393 MoE KS3 architecture')
    policy=load_policy(ROOT/'configs/ks3/screening_policy.json')
    with lock('registry.lock'):
        registry=read(STATE/'registry.json',dict(next_id=1,active_run=None))
        active=registry.get('active_run')
        if active:
            old=read(STATE/'runs'/active/'run.json',{})
            if old.get('unit') and alive_unit(old['unit']):
                raise RuntimeError(f'{active} is still live; monitor or cancel it before starting another')
        number=registry['next_id']
        label=re.sub('[^a-z0-9_-]+','-',options.label.lower()).strip('-')[:48] or 'candidate'
        run_id=f'run-{number:06d}-{label}'
        directory=STATE/'runs'/run_id
        directory.mkdir(parents=True,exist_ok=False)
        unit=f'ks3-research-run-{number:06d}.service'
        atomic(directory/'candidate.json',config_data)
        atomic(directory/'screening_policy.json',policy)
        launcher='run_control393.sh' if config_data.get('architecture')=='pr393-moe-ks3' else 'run_ks3.sh'
        active_controls=read(ROOT/'configs/control393/control_set.json',{})
        if active_controls.get('status')=='completed' and launcher!='run_control393.sh':
            raise RuntimeError('The active study requires the matched PR393 MoE KS3 architecture')
        command=[str(ROOT/launcher),'--config',str(directory/'candidate.json'),
                 '--out',str(directory/'artifacts'),'--seed',str(options.seed)]
        if options.kind!='trial':
            command.append('--'+options.kind)
        record=dict(run_id=run_id,unit=unit,created_utc=now(),kind=options.kind,label=label,
                    seed=options.seed,hypothesis=options.hypothesis,config_source=str(config),
                    directory=str(directory),command=command)
        atomic(directory/'run.json',record)
        atomic(directory/'status.json',dict(run_id=run_id,state='starting',updated_utc=now()))
        registry.update(next_id=number+1,active_run=run_id,latest_run=run_id)
        atomic(STATE/'registry.json',registry)
        atomic(STATE/'latest.json',record)
        if options.kind=='trial':
            atomic(STATE/'latest_trial.json',record)
        temporary=STATE/'latest.new'
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(directory)
        temporary.replace(STATE/'latest')
    note(run_id,'start',options.hypothesis)
    subprocess.run(['systemd-run','--quiet','--unit',unit,'--service-type=exec',
                    '--property=WorkingDirectory='+str(ROOT),'--property=KillMode=control-group',
                    '--property=TimeoutStopSec=30','--property=StandardInput=null',
                    '/usr/bin/python3',str(SELF),'worker','--run-id',run_id],check=True)
    return record


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()


def worker(run_id):
    directory=STATE/'runs'/run_id
    record=read(directory/'run.json')
    policy=load_policy(directory/'screening_policy.json')
    candidate_config=read(directory/'candidate.json',{})
    curves=ReferenceCurves(ROOT,STATE,run_id)
    seen=0
    warmup_reached=False
    warmup_observation=None
    cancellation=None
    started=now()
    with (directory/'console.log').open('a',buffering=1) as console:
        process=subprocess.Popen(record['command'],cwd=ROOT,stdin=subprocess.DEVNULL,
                                 stdout=console,stderr=subprocess.STDOUT,start_new_session=True,
                                 env={**os.environ,'KS3_REGISTERED_RUN_ID':run_id})
        pid_record=dict(worker_pid=os.getpid(),launcher_pid=process.pid,process_group_id=process.pid,
                        started_utc=started,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        atomic(directory/'process.json',pid_record)
        while process.poll() is None:
            metrics=events(directory/'artifacts/metrics.jsonl')
            reference=curves.refresh()
            snapshot=curves.snapshot()
            reference_file=directory/f"references-{snapshot['sha256']}.json"
            if not reference_file.exists():
                atomic(reference_file,snapshot)
            for metric in metrics[seen:]:
                candidate=point(metric,candidate_config,run_id)
                if record['kind']=='trial' and candidate is not None:
                    assessment=assess(candidate,reference,warmup_reached,policy)
                    if assessment['warmup_reached'] and not warmup_reached:
                        warmup_observation=candidate
                    warmup_reached=assessment['warmup_reached']
                    check=dict(datetime_utc=now(),step=metric.get('step'),
                               training_tokens=metric.get('training_tokens'),wall_seconds=metric.get('wall_seconds'),
                               reference_sha256=snapshot['sha256'],
                               **assessment)
                    with (directory/'pruning.jsonl').open('a') as handle:
                        handle.write(json.dumps(check,allow_nan=False)+'\n')
                    if assessment['abort']:
                        cancellation=dict(reason='nonfinite_loss' if assessment['reason']=='nonfinite_loss'
                                          else 'inferior_to_current_best',evidence=check)
                        break
            seen=len(metrics)
            atomic(directory/'screening_state.json',dict(updated_utc=now(),policy_version=policy['version'],
                   warmup_reached=warmup_reached,warmup_observation=warmup_observation,events_seen=seen,
                   reference_sha256=snapshot['sha256']))
            requested=read(directory/'cancel.json')
            if requested:
                cancellation=requested
            marker = stop_marker()
            if marker is not None:
                cancellation=dict(reason='user_stop', marker=str(marker))
            current=dict(run_id=run_id,state='cancelling' if cancellation else 'running',
                         kind=record['kind'],updated_utc=now(),started_utc=started,unit=record['unit'],
                         process=pid_record,latest_metric=metrics[-1] if metrics else None,
                         screening=dict(policy_version=policy['version'],warmup_reached=warmup_reached),
                         latest_training=next((m for m in reversed(metrics) if m.get('event')=='training'),None),
                         latest_validation=next((m for m in reversed(metrics) if m.get('event')=='validation'),None))
            atomic(directory/'status.json',current)
            atomic(STATE/'status.json',current)
            if cancellation:
                atomic(directory/'cancellation.json',dict(cancellation, recorded_utc=now()))
                terminate(process)
                break
            time.sleep(policy['watchdog_poll_seconds'])
        code=process.wait()
    result=read(directory/'artifacts/result.json',{})
    final=dict(run_id=run_id,state='cancelled' if cancellation else 'completed' if code==0 else 'failed',
               kind=record['kind'],updated_utc=now(),started_utc=started,finished_utc=now(),
               unit=record['unit'],exit_code=code,cancellation=cancellation,result=result,process=pid_record)
    atomic(directory/'status.json',final)
    atomic(STATE/'status.json',final)
    with lock('registry.lock'):
        registry=read(STATE/'registry.json')
        if registry.get('active_run')==run_id:
            registry['active_run']=None
        registry['last_finished_run']=run_id
        atomic(STATE/'registry.json',registry)
        complete=(code==0 and record['kind']=='trial' and result.get('full_validation') is True
                  and result.get('validation_tokens')==10485760 and result.get('val_loss',math.inf)<=TARGET)
        wall=result.get('complete_wall_seconds',math.inf)
        best=read(STATE/'best.json',{})
        active_controls=read(ROOT/'configs/control393/control_set.json',{})
        control_wall=(active_controls['best_complete_wall_seconds'] if active_controls.get('status')=='completed'
                      else 2091.5553867050003)
        best_target=read(STATE/'best_target.json',{})
        if complete:
            measured=dict(run_id=run_id,datetime_utc=now(),val_loss=result['val_loss'],
                          complete_wall_seconds=wall,time_to_target_seconds=result.get('time_to_target_seconds'),
                          target=TARGET,directory=str(directory),seed=record['seed'],
                          faster_than_control=wall<control_wall)
            if wall<best_target.get('complete_wall_seconds',math.inf):
                atomic(STATE/'best_target.json',measured)
            if wall<min(best.get('complete_wall_seconds',math.inf),control_wall):
                atomic(STATE/'best.json',measured)
        with (STATE/'ledger.jsonl').open('a') as handle:
            handle.write(json.dumps(final)+'\n')
    note(run_id,'result',f"{final['state']}; exit={code}; validation={result.get('val_loss')}; full={result.get('full_validation',False)}; validation tokens={result.get('validation_tokens')}; complete wall={result.get('complete_wall_seconds')}; reason={(cancellation or {}).get('reason',result.get('reason'))}.")
    return final


def status():
    latest=read(STATE/'latest.json',{})
    result=dict(datetime_utc=now(),supervisor=read(STATE/'supervisor.json',{}),
                codex_service=unit_status(SERVICE),registry=read(STATE/'registry.json',{}),
                latest=latest,run=read(STATE/'status.json',{}),best=read(STATE/'best.json'),
                best_target=read(STATE/'best_target.json'),
                notes=str(STATE/'notes.log'))
    if latest.get('unit'):
        result['run_service']=unit_status(latest['unit'])
        result['run_live']=alive_unit(latest['unit'])
    return result


def brief_result(result):
    if not isinstance(result,dict):return result
    # Display only; immutable result artifacts retain all per-case evidence.
    brief={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
    first=result.get('first_target_hit')
    if isinstance(first,dict):
        brief['first_target_hit']={k:first.get(k) for k in ('step','val_loss','val_tokens','wall_seconds','weights')}
    for name,value in result.items():
        if name.endswith('_probe') and isinstance(value,dict):
            summary={k:value[k] for k in ('passed','failed_case','error_type','error') if k in value}
            summary['checks']=len(value.get('checks',[]))
            timing=value.get('real_document_benchmark',value.get('benchmark',{}))
            summary.update({k:timing[k] for k in ('speedup','incumbent_speedup') if k in timing})
            brief[name]=summary
    return brief


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    start_parser=commands.add_parser('start')
    start_parser.add_argument('--config',required=True)
    start_parser.add_argument('--label',required=True)
    start_parser.add_argument('--hypothesis',required=True)
    start_parser.add_argument('--kind',choices=('trial','pilot','smoke'),default='trial')
    start_parser.add_argument('--seed',type=int,default=0)
    commands.add_parser('status').add_argument('--brief',action='store_true')
    wait=commands.add_parser('wait')
    wait.add_argument('--seconds',type=int,default=120)
    wait.add_argument('--brief',action='store_true')
    worker_parser=commands.add_parser('worker')
    worker_parser.add_argument('--run-id',required=True)
    cancel=commands.add_parser('cancel')
    cancel.add_argument('--reason',required=True)
    notes=commands.add_parser('note')
    notes.add_argument('--run-id',required=True)
    notes.add_argument('--phase',default='decision')
    notes.add_argument('--text',required=True)
    commands.add_parser('stop')
    options=parser.parse_args()
    if options.action=='start':
        value=start(options)
    elif options.action=='worker':
        value=worker(options.run_id)
    elif options.action in ('status','wait'):
        if options.action=='wait':
            deadline=time.monotonic()+max(1,min(options.seconds,3600))
            while time.monotonic()<deadline:
                latest=read(STATE/'latest.json',{})
                if not latest.get('unit') or not alive_unit(latest['unit']):
                    break
                time.sleep(min(5,max(0,deadline-time.monotonic())))
        value=status()
    elif options.action=='note':
        value=note(options.run_id,options.phase,options.text)
    elif options.action in ('cancel','stop'):
        if options.action=='stop':
            (STATE/'STOP').write_text(now()+'\n')
        registry=read(STATE/'registry.json',{})
        active=registry.get('active_run')
        if active:
            atomic(STATE/'runs'/active/'cancel.json',dict(datetime_utc=now(),reason=getattr(options,'reason','user_stop')))
        if options.action=='stop':
            subprocess.run(['systemctl','stop',SERVICE],check=True)
        value=dict(cancel_requested=active,stopped=options.action=='stop')
    if getattr(options,'brief',False):
        run=value.get('run',{})
        value=dict(datetime_utc=value['datetime_utc'],
                   codex_state=value['codex_service'].get('ActiveState'),
                   codex_pid=value['supervisor'].get('codex_pid'),
                   run_id=run.get('run_id'),state=run.get('state'),live=value.get('run_live',False),
                   training=run.get('latest_training'),validation=run.get('latest_validation'),
                   screening=run.get('screening'),
                   result=brief_result(run.get('result')),cancellation=run.get('cancellation'),best=value.get('best'),
                   best_target=value.get('best_target'))
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
