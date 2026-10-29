"""One fresh-seed replication of an immutable, loss-qualifying experiment."""
import argparse,datetime,hashlib,json,os,shlex,shutil,signal,subprocess,time
from pathlib import Path
from experiments.h100_ks3.research import ROOT,RESULTS,STATE,append,collect,utc
from experiments.h100_ks3.references import local_control_reference


def qualify_original(result, reference_seconds, allow_slower_than_reference=False):
    assert result['status']=='complete' and result['source_hashes_unchanged']
    assert result['validation_loss']<=3.28
    assert result['timing_reference_eligible']
    assert allow_slower_than_reference or result['official_seconds']<reference_seconds


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('original',type=Path);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--reference',choices=('local','published'),default='local')
    ap.add_argument('--allow-slower-than-reference',action='store_true',
                    help='Explicitly requested replication of a loss-compliant recipe; keep the time reference unchanged.')
    a=ap.parse_args();original=a.original.resolve()
    result=json.loads((original/'result.json').read_text())
    reference=local_control_reference() if a.reference=='local' else dict(kind='published_pr393',n=12,seconds=31.285)
    reference_seconds=reference['seconds']
    qualify_original(result,reference_seconds,a.allow_slower_than_reference)
    settings=json.loads((original/'settings.json').read_text())
    assert settings['seed']!=a.seed
    assert not any((p/'STOP').exists() for p in (STATE,RESULTS))
    # Each seed must be new, including earlier failed/stopped attempts.
    for line in (RESULTS/'registry.jsonl').read_text().splitlines():
        prior=json.loads(line)
        assert not (prior.get('verification_of')==original.name and prior.get('seed')==a.seed)
    manifest=json.loads((original/'execution_manifest.json').read_text())
    assert all(hashlib.sha256((original/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    run=STATE/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run.mkdir(parents=True,exist_ok=False)
    for directory in ('source','overlay'):
        shutil.copytree(original/directory,run/directory,ignore=shutil.ignore_patterns('__pycache__','logs'))
    for name in ('launch.sh','execution_manifest.json'):
        shutil.copy2(original/name,run/name)
    assert all(hashlib.sha256((run/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    settings.update(utc=utc(),seed=a.seed,verification_of=original.name,
                    verification_reference=a.reference,verification_reference_seconds=reference_seconds,
                    verification_reference_details=reference,
                    verification_allow_slower_than_reference=a.allow_slower_than_reference,
                    comparative_pruning=False,hypothesis='Fresh independent seed of user-selected frozen loss-compliant configuration; full budget and validation, no comparative pruning. Time is reported against the unchanged reference.')
    settings['environment']['TRAIN_SEED']=str(a.seed)
    (run/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    launch=run/'launch.sh';custom=settings['environment']
    (run/'reproduce.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in custom.items())+'exec '+shlex.quote(str(launch))+'\n')
    environment={k:v for k,v in os.environ.items() if not k.startswith('KS3_')}
    p=subprocess.Popen([str(launch)],cwd=ROOT,env=dict(environment,**custom),stdout=(run/'launcher.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    (run/'launcher.pid').write_text(str(p.pid)+'\n')
    append('registry.jsonl',dict(kind='start',run_id=run.name,status='running',pid=p.pid,**settings))
    append('notes.jsonl',dict(kind='verification',run_id=run.name,verification_of=original.name,seed=a.seed,code_commit=settings['code_commit'],manifest_sha256=result['manifest_sha256']))
    print(run,flush=True)
    def stop(*_):
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        p.wait();append('registry.jsonl',dict(kind='stop',run_id=run.name,status='stopped'))
        raise SystemExit('Replication stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while p.poll() is None:
        if any((d/'STOP').exists() for d in (STATE,RESULTS)):stop()
        time.sleep(2)
    if not (run/'exit_code.txt').exists():(run/'exit_code.txt').write_text(str(p.returncode)+'\n')
    collect(run)


if __name__=='__main__':main()
