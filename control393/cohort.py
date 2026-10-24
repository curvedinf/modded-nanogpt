"""Three frozen independent controls; only successful completion enables KS3 handoff."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
from control393.ctl import ROOT,STATE,read,save,start,note,now
from control393.tune import wait

def source_hashes():
    paths=[ROOT/'train_control393.py',ROOT/'run_control393.sh',
           *sorted((ROOT/'control393').rglob('*.py')),*sorted((ROOT/'mi300').rglob('*.py')),
           *sorted((ROOT/'ks3_research').rglob('*.py'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def main():
    save(STATE/'cohort.json',dict(state='waiting_for_tuning',updated_utc=now(),seeds=[11,22,33]))
    while read(STATE/'tuning.json',{}).get('state')!='frozen':
        if (STATE/'STOP').exists():return
        if subprocess.run(['systemctl','is-active','--quiet','control393-tuning.service']).returncode:
            raise RuntimeError('Tuning ended without frozen recipe; do not launch controls')
        time.sleep(5)
    config_path=ROOT/'configs/control393/frozen.json'
    config=read(config_path)
    config.update(first_validation_seconds=360.,validation_interval_seconds=120.)
    save(config_path,config)
    config_digest=hashlib.sha256(config_path.read_bytes()).hexdigest()
    frozen=source_hashes();save(STATE/'frozen-source-sha256.json',frozen)
    controls=[]
    for seed in [11,22,33]:
        if (STATE/'STOP').exists():return
        if source_hashes()!=frozen:raise RuntimeError('Execution source changed during frozen controls')
        if hashlib.sha256(config_path.read_bytes()).hexdigest()!=config_digest:
            raise RuntimeError('Frozen recipe changed during controls')
        r=start(SimpleNamespace(config=str(ROOT/'configs/control393/frozen.json'),kind='control',
            label=f'pr393-moe-control-seed{seed}',seed=seed,
            hypothesis=f'Frozen full PR393 MoE control, independent seed{seed}. Same recipe/source as the other controls; train until full10485760-token CE<=3.793633 with no fixed-step/time termination or KS3 ablation pruning.'))
        save(STATE/'cohort.json',dict(state='running_controls',updated_utc=now(),active=r['run_id'],completed=controls,seeds=[11,22,33]))
        s=wait(r);v=s['result']
        if not(v.get('full_validation') and v.get('validation_tokens')==10485760 and v.get('val_loss',99)<=3.793633):
            raise RuntimeError('Control failed the full target; no manager handoff')
        if source_hashes()!=frozen:raise RuntimeError('Source changed during a control')
        if hashlib.sha256(config_path.read_bytes()).hexdigest()!=config_digest:
            raise RuntimeError('Frozen recipe changed during a control')
        controls.append(dict(run_id=r['run_id'],directory=r['directory'],seed=seed,
            val_loss=v['val_loss'],complete_wall_seconds=v['complete_wall_seconds'],
            time_to_target_seconds=v['time_to_target_seconds'],validation_tokens=v['validation_tokens']))
        note(r['run_id'],'decision',f"Control seed{seed} reached full CE{v['val_loss']:.9f}, complete wall{v['complete_wall_seconds']:.6f}s. Preserve result/curve/source; proceed only with the same frozen recipe.")
    walls=[r['complete_wall_seconds'] for r in controls];losses=[r['val_loss'] for r in controls]
    result=dict(status='completed',created_utc=now(),architecture='pr393-moe',target=3.793633,
        validation_tokens=10485760,controls=controls,mean_complete_wall_seconds=statistics.mean(walls),
        median_complete_wall_seconds=statistics.median(walls),std_complete_wall_seconds=statistics.stdev(walls),
        best_complete_wall_seconds=min(walls),mean_val_loss=statistics.mean(losses),
        frozen_config=str(config_path),frozen_config_sha256=config_digest,source_sha256=frozen,
        best_control=min(controls,key=lambda x:x['complete_wall_seconds']),
        short_test_budget_seconds=1800,short_test_seconds=read(STATE/'tuning.json')['short_test_seconds_including_reserve'])
    save(ROOT/'configs/control393/control_set.json',result)
    save(STATE/'cohort.json',dict(state='controls_complete',updated_utc=now(),summary=result))
    subprocess.run(['/usr/bin/python3','-m','control393.handoff'],cwd=ROOT,check=True)

if __name__=='__main__':main()
