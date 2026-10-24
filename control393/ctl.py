"""Independent detached registry for the new controls and bounded tuning."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'logs/control393'
def now():return datetime.now(timezone.utc).isoformat()
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def note(r,phase,text):
    e=dict(datetime_utc=now(),run_id=r,phase=phase,note=text)
    with (STATE/'notes.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
def live(unit):return subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0
def start(a):
    STATE.mkdir(exist_ok=True)
    with (STATE/'registry.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        old=read(STATE/'latest.json',{})
        if old and live(old['unit']):raise RuntimeError('Previous control experiment still active')
        n=read(STATE/'registry.json',{'next_id':1})['next_id']
        rid=f'{n:03d}-{a.label}';d=STATE/'runs'/rid;d.mkdir(parents=True)
        c=read(Path(a.config).resolve());save(d/'candidate.json',c)
        used=sum(read(p,{}).get('elapsed_seconds',0) for p in (STATE/'runs').glob('*/status.json')
                 if read(p,{}).get('kind') in ('pilot','smoke','screen'))
        # Reserve one minute for the separately logged numerical unit checks.
        remaining=1800-60-used
        if a.kind!='control' and remaining<15:raise RuntimeError('30-minute short-test budget exhausted')
        r=dict(run_id=rid,unit=f'control393-{n:03d}.service',directory=str(d),kind=a.kind,
               config=str(d/'candidate.json'),seed=a.seed,created_utc=now(),hypothesis=a.hypothesis,
               test_budget_remaining_seconds=remaining)
        save(d/'run.json',r);save(STATE/'latest.json',r);save(STATE/'registry.json',{'next_id':n+1})
        tmp=STATE/'latest.new';tmp.unlink(missing_ok=True);tmp.symlink_to(d);tmp.replace(STATE/'latest')
        note(rid,'start',a.hypothesis)
        subprocess.run(['systemd-run','--quiet','--unit',r['unit'],'--property=WorkingDirectory='+str(ROOT),
            '--property=KillMode=control-group','--property=TimeoutStopSec=20','/usr/bin/python3',__file__,'worker','--run-id',rid],check=True)
        return r
def worker(rid):
    d=STATE/'runs'/rid;r=read(d/'run.json');beg=time.monotonic()
    command=[str(ROOT/'run_control393.sh'),'--config',r['config'],'--out',str(d/'artifacts'),'--seed',str(r['seed'])]
    if r['kind'] in ('pilot','smoke'):command+=['--'+r['kind']]
    cancelled=None
    with (d/'console.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            elapsed=time.monotonic()-beg
            if (STATE/'STOP').exists():cancelled='user_stop'
            if r['kind']!='control' and elapsed>=r['test_budget_remaining_seconds']:cancelled='short_test_budget_exhausted'
            s=dict(run_id=rid,kind=r['kind'],state='running',updated_utc=now(),elapsed_seconds=elapsed,launcher_pid=p.pid)
            m=d/'artifacts/metrics.jsonl'
            if m.exists():
                rows=[]
                for line in m.read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
                s['training']=next((x for x in reversed(rows) if x.get('event')=='training'),None)
                s['validation']=next((x for x in reversed(rows) if x.get('event')=='validation'),None)
            save(d/'status.json',s);save(STATE/'status.json',s)
            if cancelled:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(3)
        code=p.wait()
    result=read(d/'artifacts/result.json',{})
    s.update(state='cancelled' if cancelled else 'completed' if code==0 else 'failed',exit_code=code,
             result=result,cancellation=cancelled,elapsed_seconds=time.monotonic()-beg,finished_utc=now())
    save(d/'status.json',s);save(STATE/'status.json',s)
    note(rid,'result',f"{s['state']}; loss={result.get('val_loss')}; full={result.get('full_validation')}; complete wall={result.get('complete_wall_seconds')}; {cancelled or result.get('reason')}")
    return s
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','worker','status','note']);p.add_argument('--config');p.add_argument('--kind',choices=['pilot','smoke','screen','control'],default='control');p.add_argument('--seed',type=int,default=0);p.add_argument('--label');p.add_argument('--hypothesis');p.add_argument('--run-id');p.add_argument('--text')
    a=p.parse_args()
    if a.action=='start':v=start(a)
    elif a.action=='worker':v=worker(a.run_id)
    elif a.action=='note':v=note(a.run_id,'decision',a.text)
    else:
        v=read(STATE/'status.json',{});v['latest']=read(STATE/'latest.json',{})
        if v.get('result'):v['result']={k:x for k,x in v['result'].items() if k not in ('validations','first_target_hit')}
    print(json.dumps(v,indent=2))
if __name__=='__main__':main()
