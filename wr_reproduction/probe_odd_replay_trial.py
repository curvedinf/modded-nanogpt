"""One bounded diagnostic of the first original Adam/sparse event; no new controls."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import (ROOT, STATE, RESULTS, PROTOCOL, FENCE,
    STOP_FILES, read, atomic, append, now, stopped, assert_fence, environment,
    systemd_environment, publish)

KIND = 'first-adam-and-sparse-event-replay-v1'


def validate(value):
    assert_fence(value)
    baseline = control.recipe('ks3_experts')
    omit = {'name', 'hypothesis', 'hypothesis_class', 'odd_replay_diagnostic'}
    if {k:v for k,v in value.items() if k not in omit} != {k:v for k,v in baseline.items() if k not in omit}:
        raise ValueError('Replay diagnostic requires the exact original expert-only recipe')
    if value.get('odd_replay_diagnostic') != KIND: raise ValueError('Explicit diagnostic identifier required')


def create_run(value, unit):
    from wr_reproduction import research
    if stopped(): raise RuntimeError('Explicit study STOP')
    validate(value); fence = assert_fence(value)
    dirty = control.git('status', '--porcelain', '--', 'wr_reproduction', 'configs/wr_reproduction', 'ks3_research/vendor')
    if dirty: raise RuntimeError('Commit implementation/config before launch: '+dirty)
    rid = value['name']+'-calibration-'+now().replace('-', '').replace(':', '').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory = STATE/'runs'/rid; directory.mkdir(parents=True); head = control.git('rev-parse', 'HEAD')
    run = dict(study='probe-study', run_id=rid, directory=str(directory), seed=169, kind='calibration',
        recipe=value, unit=unit, created_utc=now(),
        command=[str(control.PYTHON), '-m', 'wr_reproduction.probe_odd_replay_trial', 'train', str(directory)],
        code_commit=head, launch_head_commit=head, source_hashes=control.sources(), environment=environment(),
        protocol_path=str(PROTOCOL), fence_path=str(FENCE), backend_hash=fence['backend_hash'], backend_fence=fence,
        probe_protocol=read(PROTOCOL), validation_data=research.validation_data(), continuous_from_initialization=False,
        comparative_pruning_exempt=False, stop_files=[str(p) for p in STOP_FILES], calibration='odd-replay',
        not_a_screening_reference=True, odd_replay_diagnostic=KIND,
        diagnostic_bound_seconds_after_warmup=360., total_update_executions=7,
        scope='One original update, then three identical-state replays of the first Adam/sparse event and its next update. Full24warmup. Two no-probe baselines and one with frozen probe; exact touched sparse row restoration. No loss/time trial, reference curve, full or reserved validation. Frozen shared math/settings unchanged.')
    atomic(directory/'run.json', run); atomic(directory/'config.json', value); atomic(RESULTS/(rid+'-config.json'), run)
    append(STATE/'registry.jsonl', dict(event='registered', run_id=rid, directory=str(directory), kind='calibration',
        seed=169, code_commit=head, not_a_screening_reference=True))
    control.status(run, 'registered'); control.note(run, hypothesis=value['hypothesis'],
        decision='Bounded first-event replay diagnostic; known early-trajectory ambiguity, no new loss/time reference', code_commit=head)
    return run


def train(directory):
    import torch
    cfg = read(directory/'run.json'); validate(cfg['recipe'])
    if cfg.get('kind') != 'calibration' or not cfg.get('not_a_screening_reference'): raise RuntimeError('Diagnostic-only runner')
    if stopped(): raise RuntimeError('Explicit study STOP')
    if control.sources() != cfg['source_hashes']: raise RuntimeError('Sources changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ: raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(1)
    from wr_reproduction.probe_odd_replay import cpu_check, diagnostic
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction.train import Runner
    atomic(directory/'cpu-checks.json',cpu_check())
    with LocalCollectives():
        r=Runner('pr393',directory/'events.jsonl',True,'bf16',directory)
        r.warmup(); diagnostic(r)


def work(request):
    cfg=read(request)
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopped(): return
        run=create_run(cfg['recipe'],cfg['unit']); result=control.execute(run)
        summary=dict(utc=now(),run_id=run['run_id'],code_commit=run['code_commit'],
            passed=result['status']=='calibration_complete' and bool((result.get('calibration') or {}).get('passed')),
            result=result,not_a_screening_reference=True)
        for root in (STATE,RESULTS): atomic(root/'odd-replay-diagnostic.json',summary)


def launch(path):
    if stopped(): raise RuntimeError('Explicit study STOP')
    value=read(path);validate(value)
    unit='wr-probe-odd-replay-'+uuid.uuid4().hex[:8];request=STATE/'requests'/(unit+'.json')
    atomic(request,dict(utc=now(),unit=unit,recipe=value))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
        '--property=KillMode=control-group','--property=TimeoutStopSec=15','--property=RuntimeMaxSec=900',
        '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),str(control.PYTHON),
        '-m','wr_reproduction.probe_odd_replay_trial','work',str(request)],check=True)
    print(unit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','work','train']);p.add_argument('path',type=Path)
    args=p.parse_args()
    try:globals()[args.action](args.path)
    except BaseException:
        error=dict(utc=now(),error=traceback.format_exc())
        atomic(STATE/'odd-replay-blocker.json',error);publish('directed-status.json',dict(status='error',**error));raise
