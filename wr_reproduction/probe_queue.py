"""Resumable requested five-site batch after instrumented references/replication.

No old queue is read. This worker deliberately stops the finite placement batch
for inspection before any combinations or new optimizer hypotheses are chosen.
The detached research manager remains responsible for continued hypotheses.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid
from wr_reproduction.probe_study import *
from wr_reproduction.probe_controller import create_run,execute,verification,PYTHON,git,sources,status,note,terminate


def kernel_gate(unit):
    saved=read(RESULTS/'placement-kernel-gate.json')
    if saved and saved.get('passed'):return
    rid='probe-dense2816-numerical-'+now().replace('-','').replace(':','').split('.')[0]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(PYTHON),'-m','wr_reproduction.check_ks3_feedforward','--features','2816','--out',str(directory/'events.jsonl')]
    run=dict(study='probe-study',run_id=rid,directory=str(directory),kind='correctness',seed=172,
             recipe=read(ROOT/'configs/wr_reproduction/probe_dense_0.json'),command=command,unit=unit,
             code_commit=git('rev-parse','HEAD'),source_hashes=sources(),backend_hash=read(FENCE)['backend_hash'],
             continuous_from_initialization=False,stop_files=[str(p) for p in STOP_FILES],created_utc=now())
    atomic(directory/'run.json',run);atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind='correctness',seed=172,code_commit=run['code_commit']))
    with (directory/'console.log').open('ab',buffering=0) as out:
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        status(run,'running',trainer_pid=process.pid)
        publish('active.json',dict(run_id=rid,trainer_pid=process.pid,controller_pid=os.getpid(),unit=unit,utc=now()))
        try:
            while process.poll() is None:
                if stopped():terminate(process);break
                time.sleep(.5)
        finally:terminate(process)
    rows=events(directory/'events.jsonl');result=rows[-1] if rows else {}
    result.update(run_id=rid,code_commit=run['code_commit'],exit_code=process.returncode,passed=bool(result.get('passed') and not process.returncode and not stopped()))
    atomic(directory/'result.json',result);atomic(RESULTS/(rid+'-result.json'),result)
    atomic(RESULTS/'placement-kernel-gate.json',result);status(run,'correctness_passed' if result['passed'] else 'error',result=result)
    note(run,hypothesis='Current original-width2816 dense kernel output and all-gradient parity',result=result,decision='Proceed to five isolated sites only on pass')
    if not result['passed']:raise RuntimeError('Dense numerical gate failed')


def work(unit):
    while read(STATE/'phase.json',{}).get('phase')!='references_complete_placement_batch_next':
        if stopped():return
        progress=read(STATE/'references.json',dict(arms=[]))
        if any(x['status'] in ('error','stopped','incomplete') for x in progress['arms']):
            raise RuntimeError('Reference error blocks placement batch')
        time.sleep(2)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        while True:
            if stopped():return
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.5)
        kernel_gate(unit)
        progress=read(STATE/'placement-batch.json',dict(arms=[]))
        for layer in (0,2,4,8,10):
            if stopped():return
            item=next((x for x in progress['arms'] if x['layer']==layer),None)
            if item:
                run=read(STATE/'runs'/item['run_id']/'run.json');result=read(STATE/'runs'/item['run_id']/'result.json')
                if not result:raise RuntimeError('Interrupted placement requires registry reconciliation')
            else:
                gate=read(RESULTS/f'placement-layer{layer}-cpu.json')
                if not gate or not gate['passed']:raise RuntimeError('Missing placement CPU gate')
                numerical=read(RESULTS/'placement-kernel-gate.json')
                if not numerical or not numerical['passed']:raise RuntimeError('Missing original-width numerical gate')
                value=read(ROOT/f'configs/wr_reproduction/probe_dense_{layer}.json')
                run=create_run(value,169,'discovery',unit)
                item=dict(layer=layer,run_id=run['run_id'],status='running');progress['arms'].append(item)
                atomic(STATE/'placement-batch.json',progress);result=execute(run)
            item.update(status=result['status'],result=result);atomic(STATE/'placement-batch.json',progress)
            if result['status'] not in ('complete','pruned'):raise RuntimeError('Placement needs diagnosis: '+run['run_id'])
            if result['qualifies']:verification(run,result,unit)
        summary=dict(utc=now(),status='batch_complete_inspect_before_combining',arms=progress['arms'],
                     control=read(RESULTS/'control_set.json'),decision='Compare all five isolated sites and their diagnostics before any placement combination.')
        atomic(RESULTS/'placement-batch.json',summary);atomic(STATE/'phase.json',dict(phase=summary['status'],utc=now()))


def launch():
    if stopped():raise RuntimeError('Explicit probe-study STOP')
    unit='wr-probe-placements-'+uuid.uuid4().hex[:8]
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
        '--property=KillMode=control-group','--property=TimeoutStopSec=15',
        '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),str(PYTHON),'-m','wr_reproduction.probe_queue','work','--unit',unit],check=True)
    atomic(STATE/'placement-controller.json',dict(unit=unit,status='waiting_for_references_and_required_replication',utc=now()))
    print(unit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','work']);p.add_argument('--unit');a=p.parse_args()
    if a.action=='launch':launch()
    else:
        try:work(a.unit)
        except BaseException:
            atomic(STATE/'placement-blocker.json',dict(utc=now(),error=traceback.format_exc()));raise
