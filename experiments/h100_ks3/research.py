"""One isolated, registered H100 experiment; stop-aware, no automatic reruns."""
import argparse,datetime,hashlib,json,os,re,shlex,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'research_results/wr_reproduction/h100-research'
STATE=ROOT/'logs/wr-reproduction/h100-research'

def terminal_comparison(result):
    # These short runs currently provide terminal full validation only. Never
    # manufacture a curve from training losses or compare unequal eval modes.
    from wr_reproduction.probe_study import decision
    signature=dict(protocol='PR393-full-final-canonical-tail-window-v1',tokens=10485760)
    def point(row):
        return dict(run_id=row['run_id'],loss=row['validation_loss'],wall_seconds=row['official_seconds'],
                    signature=signature,signature_hash=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),chunk_losses=[])
    references=[]
    registry=RESULTS/'registry.jsonl'
    if registry.exists():
        for line in registry.read_text().splitlines():
            row=json.loads(line)
            if row.get('timing_reference_eligible') and row.get('status')=='complete' and 'validation_loss' in row:
                references.append(point(row))
    return decision(point(result),references,False,max_age=3.)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def append(name,data):
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/name).open('a') as f:f.write(json.dumps({'utc':utc(),**data})+'\n')
def collect(run):
    settings=json.loads((run/'settings.json').read_text())
    text=(run/'console.log').read_text(errors='replace')
    found=re.findall(r'\[wr-result\] steps:(\d+) loss:([\d.]+) seconds:([\d.]+)',text)
    if not found:
        found=[(n,loss,str(float(ms)/1000)) for n,loss,ms in re.findall(r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:([\d.]+)ms',text)]
    code=int((run/'exit_code.txt').read_text())
    manifest=json.loads((run/'execution_manifest.json').read_text())
    unchanged=all(hashlib.sha256((run/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    result=dict(run_id=run.name,status='complete' if code==0 else 'failed',exit_code=code,settings=settings,
                code_commit=settings['code_commit'],source_hashes_unchanged=unchanged,
                manifest_sha256=hashlib.sha256((run/'execution_manifest.json').read_bytes()).hexdigest(),
                timing_reference_eligible=not (settings.get('profile',False) or settings.get('kind')=='full-profile' or settings.get('timing_reference_eligible') is False),log_directory=str(run))
    if found:
        step,loss,seconds=found[-1]
        result.update(steps=int(step),validation_loss=float(loss),official_seconds=float(seconds),validation_tokens=10485760,
                      target_met=float(loss)<=3.28)
        if result['timing_reference_eligible']:
            result['terminal_comparison']=terminal_comparison(result)
    elif code==0:result['status']='timing_screen' if settings.get('screen_updates') else 'missing_validation'
    (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    out=RESULTS/'runs'/run.name;out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ['settings.json','execution_manifest.json','reproduce.sh','full-profile-summary.json']:
        if (run/name).exists():(out/name).write_bytes((run/name).read_bytes())
    (out/'console-tail.txt').write_text('\n'.join(text.splitlines()[-60:])+'\n')
    append('registry.jsonl',result)
    append('notes.jsonl',dict(kind='result',run_id=run.name,result={k:v for k,v in result.items() if k not in ('settings',)},hypothesis=settings.get('hypothesis')))
    print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',default='separate-dx-prepared-128-gate')
    ap.add_argument('--hypothesis');ap.add_argument('--seed',type=int,default=413)
    ap.add_argument('--steps',type=int,default=981);ap.add_argument('--screen-updates',type=int,default=0)
    ap.add_argument('--drop-dense-layer',type=int,choices=(0,2,4));ap.add_argument('--dense-width',type=int,default=2816)
    ap.add_argument('--drop-parallel-mlp',action='store_true')
    ap.add_argument('--first-batch-units',type=int,choices=(4,8,12,16),default=8)
    ap.add_argument('--env',action='append',default=[]);ap.add_argument('--profile',action='store_true')
    ap.add_argument('--collect',type=Path);ap.add_argument('--adopt',type=Path);a=ap.parse_args()
    if a.collect:collect(a.collect);return
    if a.adopt:
        run=a.adopt;pid=int((run/'launcher.pid').read_text());settings=json.loads((run/'settings.json').read_text())
        append('registry.jsonl',dict(kind='adopt',run_id=run.name,status='running',pid=pid,**settings))
        while not (run/'exit_code.txt').exists():
            if any((d/'STOP').exists() for d in (STATE,RESULTS)):
                os.killpg(pid,signal.SIGTERM)
                append('registry.jsonl',dict(kind='stop',run_id=run.name,status='stopped'))
                return
            time.sleep(2)
        collect(run);return
    if not a.hypothesis:ap.error('--hypothesis is required for a new experiment')
    if any((p/'STOP').exists() for p in (STATE,RESULTS)):raise SystemExit('New study STOP is present')
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments/h100_ks3','wr_reproduction','ks3_research'],cwd=ROOT,text=True)
    if dirty:raise SystemExit('Commit code before launching an immutable experiment: '+dirty)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cmd=[sys.executable,'-m','experiments.h100_ks3.prepare','--study','h100-research','--scheduled-steps',str(a.steps),'--screen-updates',str(a.screen_updates)]
    cmd.extend(['--dense-width',str(a.dense_width),'--first-batch-units',str(a.first_batch_units)])
    if a.drop_dense_layer is not None:cmd.extend(['--drop-dense-layer',str(a.drop_dense_layer)])
    if a.drop_parallel_mlp:cmd.append('--drop-parallel-mlp')
    if a.profile:cmd.append('--profile')
    launch=Path(subprocess.check_output(cmd,cwd=ROOT,text=True).strip());run=launch.parent
    custom=dict(TRAIN_SEED=str(a.seed),KS3_H100_OPT=a.variant,KS3_H100_TRANSPORT='codes',KS3_H100_PACK_ROWS='16')
    custom.update(dict(x.split('=',1) for x in a.env))
    settings=dict(utc=utc(),code_commit=commit,hypothesis=a.hypothesis,variant=a.variant,seed=a.seed,scheduled_steps=a.steps,
                  drop_dense_layer=a.drop_dense_layer,drop_parallel_mlp=a.drop_parallel_mlp,dense_width=a.dense_width,first_batch_units=a.first_batch_units,
                  environment=custom,profile=a.profile,screen_updates=a.screen_updates,validation_protocol='PR393 full final canonical mask, window extension and tail shipment')
    (run/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    (run/'reproduce.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in custom.items())+'exec '+shlex.quote(str(launch))+'\n')
    p=subprocess.Popen([str(launch)],cwd=ROOT,env=dict(os.environ,**custom),stdout=(run/'launcher.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    (run/'launcher.pid').write_text(str(p.pid)+'\n')
    append('registry.jsonl',dict(kind='start',run_id=run.name,status='running',pid=p.pid,**settings))
    append('notes.jsonl',dict(kind='hypothesis',run_id=run.name,hypothesis=a.hypothesis,environment=custom))
    print(run,flush=True)
    def stop(*_):
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        p.wait();append('registry.jsonl',dict(kind='stop',run_id=run.name,status='stopped'));raise SystemExit('Experiment stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while p.poll() is None:
        if any((d/'STOP').exists() for d in (STATE,RESULTS)):stop()
        time.sleep(2)
    if not (run/'exit_code.txt').exists():(run/'exit_code.txt').write_text(str(p.returncode)+'\n')
    collect(run)
if __name__=='__main__':main()
