"""Run the fixed clean PR checkout and preserve every requested seed."""
import argparse,csv,datetime,hashlib,json,math,os,re,shutil,signal,statistics,subprocess,time
from pathlib import Path
from experiments.h100_ks3.research import ROOT,RESULTS,STATE,append,summarize,utc
from experiments.h100_ks3.verification_summary import student_cdf

BRANCH=Path('/root/modded-nanogpt-ks3-pr')
RECORDS=BRANCH/'records/track_1_short/2026-10-29_KS3_1287'
STUDY=RESULTS/'clean-port-1287'
FIELDS=['seed','steps','seconds','loss','source_commit','source_sha256','log']


def source_manifest():
    files=[BRANCH/'train_gpt.py',BRANCH/'run.sh',BRANCH/'requirements.txt']
    files+=list((BRANCH/'track_1_short').rglob('*.py'))
    files+=list((BRANCH/'track_1_short').glob('*.npy'))
    return {str(p.relative_to(BRANCH)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=BRANCH,text=True).strip()


def save(rows):
    RECORDS.mkdir(parents=True,exist_ok=True)
    with (RECORDS/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    losses=[float(r['loss']) for r in rows];times=[float(r['seconds']) for r in rows]
    n=len(rows);loss_sd=statistics.stdev(losses) if n>1 else None;time_sd=statistics.stdev(times) if n>1 else None
    statistic=(statistics.mean(losses)-3.28)/(loss_sd/math.sqrt(n)) if loss_sd else None
    probability=student_cdf(statistic,n-1) if statistic is not None else (float(statistics.mean(losses)>=3.28) if n>1 else None)
    report={'n':n,'steps':1287,'validation_tokens':10485760,'loss_threshold':3.28,
            'loss_mean':statistics.mean(losses),'loss_sample_std':loss_sd,
            'seconds_mean':statistics.mean(times),'seconds_sample_std':time_sd,
            'one_sided_student_t':statistic,'one_sided_p':probability,
            'seeds_meeting_loss':sum(v<=3.28 for v in losses),
            'reference_seconds':31.07925,'reference_n':8,
            'reference_evidence':'https://github.com/curvedinf/modded-nanogpt/blob/109d6a7/research_results/wr_reproduction/h100-control-calibration/summary.json',
            'qualifies_against_reference':statistics.mean(losses)<=3.28 and probability is not None and probability<.01 and statistics.mean(times)<31.07925,
            'pre_port_seeds':[],
            'port_check':'https://github.com/curvedinf/modded-nanogpt/blob/master/research_results/wr_reproduction/h100-research/clean-port/equivalence.json'}
    (RECORDS/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def publish(message):
    git('add',str(RECORDS.relative_to(BRANCH)))
    if git('diff','--cached','--name-only'):
        git('commit','-m',message)
        subprocess.run(['git','push','origin','ks3-pr393-h100'],cwd=BRANCH,check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='711,723,724,725,726,727,728,729,730,731,732,733');a=ap.parse_args()
    seeds=[int(s) for s in a.seeds.split(',')]
    STUDY.mkdir(exist_ok=True)
    metadata=STUDY/'branch-source.json'
    manifest=source_manifest();digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if metadata.exists():
        meta=json.loads(metadata.read_text());assert meta['manifest']==manifest,'Clean branch code changed'
    else:
        meta={'utc':utc(),'source_commit':git('rev-parse','HEAD'),'source_sha256':digest,'manifest':manifest}
        metadata.write_text(json.dumps(meta,indent=2)+'\n')
    if (RECORDS/'results.csv').exists():
        with (RECORDS/'results.csv').open() as f:rows=list(csv.DictReader(f))
    else:
        rows=[]
    current=None
    def stopped():return any((d/'STOP').exists() for d in (STATE,RESULTS,STUDY))
    def stop(*_):
        if current is not None and current.poll() is None:
            os.killpg(current.pid,signal.SIGTERM);current.wait()
        raise SystemExit('Seed series stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for seed in seeds:
        if seed in {int(r['seed']) for r in rows}:continue
        if stopped():stop()
        assert source_manifest()==meta['manifest'],'Clean branch code changed'
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run=STATE/f'pr1287-seed-{seed}-{stamp}';run.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith('KS3_')}
        env.update(TRAIN_SEED=str(seed),NUM_SCHEDULED_ITERATIONS='981',DATA_PATH=str(ROOT),CUDA_HOME='/usr/local/cuda',
                   LOCAL_KERNELS='devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned',TRITON_PTXAS_PATH='/root/venvs/h100-ptxas-13.3/ptxas',
                   PATH='/root/venvs/pr393-h100/bin:'+env.get('PATH',''),LD_LIBRARY_PATH='/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH',''))
        before=set((BRANCH/'logs').glob('*.txt'))
        settings={'seed':seed,'source_commit':meta['source_commit'],'source_sha256':meta['source_sha256'],'branch':'ks3-pr393-h100','steps':1287,'comparative_pruning':False}
        (run/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        with (run/'console.log').open('w') as stream:
            current=subprocess.Popen(['bash','run.sh'],cwd=BRANCH,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            (run/'launcher.pid').write_text(str(current.pid)+'\n')
            append('registry.jsonl',{'kind':'branch_seed_start','status':'running','run_id':run.name,'pid':current.pid,'settings':settings})
            print(json.dumps({'started':seed,'run':str(run)}),flush=True)
            while current.poll() is None:
                if stopped():stop()
                time.sleep(1)
        (run/'exit_code.txt').write_text(str(current.returncode)+'\n')
        console=(run/'console.log').read_text(errors='replace')
        found=re.findall(r'\[wr-result\] steps:(\d+) loss:([\d.]+) seconds:([\d.]+)',console)
        if current.returncode or not found:
            append('registry.jsonl',{'kind':'branch_seed_failed','status':'failed','run_id':run.name,'exit_code':current.returncode,'settings':settings})
            raise SystemExit(f'Seed{seed} did not finish; inspect {run}')
        assert source_manifest()==meta['manifest'],'Clean branch code changed during run'
        steps,loss,seconds=found[-1];assert int(steps)==1287
        logs=set((BRANCH/'logs').glob('*.txt'))-before;assert len(logs)==1,logs
        filename=f'seed_{seed}.txt';shutil.copyfile(logs.pop(),RECORDS/filename)
        row=dict(seed=seed,steps=int(steps),seconds=float(seconds),loss=float(loss),source_commit=meta['source_commit'],source_sha256=meta['source_sha256'],log=filename)
        rows.append(row);report=save(rows)
        result={'run_id':run.name,'status':'complete','exit_code':0,'settings':settings,'code_commit':meta['source_commit'],'manifest_sha256':meta['source_sha256'],'source_hashes_unchanged':True,'timing_reference_eligible':True,'steps':1287,'official_seconds':float(seconds),'validation_loss':float(loss),'validation_tokens':10485760,'target_met':float(loss)<=3.28}
        (run/'result.json').write_text(json.dumps(result,indent=2)+'\n');out=RESULTS/'runs'/run.name;out.mkdir();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        append('registry.jsonl',result);append('notes.jsonl',{'kind':'branch_seed_result','result':row,'decision':'Keep1287-update recipe frozen; continue only if this full loss is strictly below3.28.'});summarize()
        publish(f'Record frozen KS3 seed {seed} ({len(rows)}/12)')
        print(json.dumps({'completed':row,'summary':report}),flush=True)
        if float(loss)>=3.28:
            report['status']='stopped_seed_loss_at_least_3.28'
            report['failed_seed']=seed
            (RECORDS/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            (STUDY/'STOP').write_text(f'Seed{seed} full CE{loss}>=3.28; user stopping rule.\n')
            append('notes.jsonl',{'kind':'series_stop','run_id':run.name,'seed':seed,'loss':float(loss),'reason':'User directs every run strictly below3.28.'})
            publish(f'Stop1287-update series after seed{seed} missed3.28')
            return
    print(json.dumps({'series_complete':True,'summary':save(rows)}),flush=True)


if __name__=='__main__':main()
