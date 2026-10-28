"""Eleven new, unpruned, exact-PR393 controls plus the preserved original."""
import argparse,datetime,hashlib,json,math,os,re,shutil,signal,statistics,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=ROOT/'logs/wr-reproduction/h100-pr393-baseline/20261028T180502Z'
STATE=ROOT/'logs/wr-reproduction/h100-control-calibration'
OUT=ROOT/'research_results/wr_reproduction/h100-control-calibration'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def event(value):
    with (OUT/'registry.jsonl').open('a') as f:f.write(json.dumps(dict(utc=utc(),**value))+'\n')
def parse(text):
    rows=re.findall(r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:([\d.]+)ms',text)
    assert len(rows)==1,rows
    step,loss,ms=rows[0];assert int(step)==1053
    return dict(steps=int(step),validation_loss=float(loss),official_seconds=float(ms)/1000)
def describe(values):
    sd=statistics.stdev(values) if len(values)>1 else None
    return dict(n=len(values),mean=statistics.mean(values),sample_std=sd,
                standard_error=sd/math.sqrt(len(values)) if sd is not None else None,
                minimum=min(values),maximum=max(values))
def complete():
    return [json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/result.json'))
            if json.loads(p.read_text()).get('status')=='complete']
def summary():
    rows=complete();old=json.loads((ORIGINAL/'result.json').read_text())
    original=dict(run_id=ORIGINAL.name,seed='original random default',official_seconds=old['official_training_time_seconds'],validation_loss=old['validation_loss_logged'])
    report=dict(utc=utc(),requested_fresh_runs=11,completed_fresh_runs=len(rows),
                original=original,fresh_runs=rows,published_pr393_seconds=31.285,
                validation_tokens=10485760,source_changes=[],
                loss_precision='Original unchanged control prints each loss to four decimal places.')
    for key,data in [('fresh_only',rows),('combined_with_original',[original,*rows])]:
        if data:report[key]=dict(seconds=describe([r['official_seconds'] for r in data]),loss=describe([r['validation_loss'] for r in data]))
    write(OUT/'summary.json',report);return report
def checkpoint(message):
    subprocess.run(['git','add',str(OUT.relative_to(ROOT))],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
    push=subprocess.run(['git','push','origin','master'],cwd=ROOT,text=True,capture_output=True)
    print(push.stdout+push.stderr,flush=True)
    if push.returncode:event(dict(kind='push_failed',returncode=push.returncode,error=push.stderr))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--total',type=int,default=11);a=ap.parse_args()
    assert a.total==11,'The user requested exactly eleven new complete controls'
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'runs').mkdir(exist_ok=True);STATE.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ORIGINAL/'source_manifest.json').read_text())
    verify=lambda run:all(hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest for name,digest in manifest['files'].items())
    assert verify(ORIGINAL)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan=dict(utc=utc(),original_run=ORIGINAL.name,requested_new_runs=11,seeds=list(range(601,612)),
              upstream_commit=manifest['commit'],controller_commit=code,source_changes=[],comparative_pruning=False,
              purpose='User-authorized estimate of local control time/loss distribution; preserve and include original separately.')
    if not (OUT/'plan.json').exists():write(OUT/'plan.json',plan)
    summary()
    for index in range(len(complete()),11):
        stops=(STATE/'STOP',OUT/'STOP',ROOT/'logs/wr-reproduction/h100-research/STOP',ROOT/'research_results/wr_reproduction/h100-research/STOP')
        if any(p.exists() for p in stops):raise SystemExit('New-study STOP present')
        seed=601+index;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run=STATE/stamp;run.mkdir(exist_ok=False)
        shutil.copytree(ORIGINAL/'source',run/'source',ignore=shutil.ignore_patterns('__pycache__','logs'))
        shutil.copy2(ORIGINAL/'source_manifest.json',run/'source_manifest.json');assert verify(run)
        write(run/'execution_manifest.json',{'source/'+p:h for p,h in manifest['files'].items()})
        launch=(ORIGINAL/'launch.sh').read_text().replace('unset NUM_SCHEDULED_ITERATIONS TRAIN_SEED','unset NUM_SCHEDULED_ITERATIONS PYTHONPATH')
        (run/'launch.sh').write_text(launch);(run/'launch.sh').chmod(0o755)
        settings=dict(utc=utc(),run_id=stamp,seed=seed,control_index=index+1,controller_commit=code,
                      upstream_commit=manifest['commit'],source_changes=[],environment={'TRAIN_SEED':str(seed)},
                      validation_tokens=10485760,scheduled_steps=981,growth_steps=52,extension_steps=20,
                      comparative_pruning=False,original_control=ORIGINAL.name)
        write(run/'settings.json',settings)
        env={k:v for k,v in os.environ.items() if not k.startswith('KS3_') and k not in ('NUM_SCHEDULED_ITERATIONS','PYTHONPATH','TRITON_PTXAS_PATH')}
        env['TRAIN_SEED']=str(seed)
        p=subprocess.Popen([str(run/'launch.sh')],cwd=ROOT,env=env,stdout=(run/'launcher.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
        (run/'launcher.pid').write_text(str(p.pid)+'\n');event(dict(kind='start',status='running',pid=p.pid,**settings));print(json.dumps(dict(start=stamp,index=index+1,seed=seed)),flush=True)
        def stop(*_):
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            p.wait();event(dict(kind='stop',run_id=stamp,seed=seed,status='stopped'));raise SystemExit('Control calibration stopped')
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        while p.poll() is None:
            if any(q.exists() for q in stops):stop()
            time.sleep(1)
        assert verify(run),'Control source changed during execution'
        result=dict(run_id=stamp,seed=seed,control_index=index+1,status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,
                    source_hashes_unchanged=True,controller_commit=code,upstream_commit=manifest['commit'],validation_tokens=10485760)
        if p.returncode==0:result.update(parse((run/'console.log').read_text()))
        write(run/'result.json',result);dest=OUT/'runs'/stamp;dest.mkdir()
        for name in ('result.json','settings.json','execution_manifest.json','console.log','launch.sh','gpu-environment.txt','packages.txt'):
            if (run/name).exists():shutil.copy2(run/name,dest/name)
        event(dict(kind='result',**result));report=summary();print(json.dumps(dict(result=result,completed=report['completed_fresh_runs'],combined=report['combined_with_original'])),flush=True)
        subprocess.run([sys.executable,'-m','experiments.h100_ks3.archive_run',str(run),str(OUT/'archives'/f'{stamp}.tar.gz')],cwd=ROOT,check=True)
        checkpoint(f'Record PR393 control {index+1}/11 and update calibration average')
        if p.returncode:raise SystemExit('Control failed; preserve evidence and inspect before continuing')
    print(json.dumps(summary(),indent=2),flush=True)


if __name__=='__main__':main()
