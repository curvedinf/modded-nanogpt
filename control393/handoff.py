"""Activate the matched MoE KS3 study only after all three new controls exist."""
import json
from pathlib import Path
import subprocess
from control393.ctl import ROOT,STATE,read,save,note,now

def main():
    cohort=read(ROOT/'configs/control393/control_set.json',{})
    assert cohort.get('status')=='completed' and len(cohort.get('controls',[]))==3
    for c in cohort['controls']:
        v=read(Path(c['directory'])/'artifacts/result.json',{})
        assert v.get('full_validation') and v.get('validation_tokens')==10485760 and v.get('val_loss',99)<=3.793633 and v.get('exit_code')==0
    if (STATE/'STOP').exists():raise RuntimeError('New user stop prevents handoff')
    old_stop=ROOT/'logs/ks3-autoresearch/STOP'
    expected=read(STATE/'authorized-handoff.json',{}).get('superseded_stop')
    if old_stop.exists():
        actual=dict(mtime_ns=old_stop.stat().st_mtime_ns,content=old_stop.read_text())
        if actual!=expected:raise RuntimeError('A new research STOP marker prevents handoff')
    config=read(ROOT/'configs/control393/frozen.json')
    config.update(architecture='pr393-moe-ks3',charts=1024,rank=1,geometry_lr_mul=.125,
        control_wall_seconds=cohort['best_complete_wall_seconds'],
        objective='Optimize matched MoE KS3 loss versus elapsed time against the completed PR393 MoE control cohort.',
        research_note='Same PR393 router/topology, dense branches, CPLM, normalization, optimizer family and selected systems settings; replace only expert MLPs with rank1 KS3 experts. Verify before full trial.')
    save(ROOT/'configs/control393/ks3_initial.json',config)
    state=ROOT/'logs/ks3-autoresearch';archive=state/'historical-pre-pr393';archive.mkdir(exist_ok=True)
    for name in ['best.json','best_target.json']:
        p=state/name
        if p.exists():p.replace(archive/name)
    supervisor=read(state/'supervisor.json',{})
    save(archive/'supervisor-before-handoff.json',supervisor)
    supervisor.update(session_id=None,state='new_pr393_moe_study',codex_pid=None,updated_utc=now())
    save(state/'supervisor.json',supervisor)
    prompt=(ROOT/'configs/control393/manager_prompt.txt').read_text()
    (ROOT/'configs/ks3/autonomous_prompt.txt').write_text(prompt)
    # The old stop preceded the explicit request to resume after the new cohort.
    (state/'STOP').unlink(missing_ok=True)
    note('cohort','handoff',f"All3 frozen MoE controls completed. Fastest complete time{cohort['best_complete_wall_seconds']:.6f}s; mean{cohort['mean_complete_wall_seconds']:.6f}s. Activate matched MoE KS3 study; historical architectures excluded from reference curves.")
    subprocess.run(['systemctl','enable','--now','ks3-codex-research.service'],check=True)
    save(STATE/'handoff.json',dict(state='manager_started',datetime_utc=now(),control_set=str(ROOT/'configs/control393/control_set.json')))

if __name__=='__main__':main()
