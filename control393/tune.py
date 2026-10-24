"""Bounded recipe selection: four equal-wall screens, then independent-seed finalists."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from control393.ctl import ROOT,STATE,read,save,start,live,note,now

def wait(record):
    while live(record['unit']):
        if (STATE/'STOP').exists():raise RuntimeError('User stop')
        time.sleep(3)
    status=read(Path(record['directory'])/'status.json',{})
    if status.get('state')!='completed':raise RuntimeError(f"{record['run_id']} failed: inspect its evidence")
    return status

def main():
    last=read(STATE/'latest.json');wait(last)
    trials=[]
    for name in ['a128-g32','b128-g64','c64-g64','d128-g64-lr09']:
        record=start(SimpleNamespace(config=str(ROOT/'configs/control393'/f'{name}.json'),
            kind='screen',label=name+'-s0',seed=0,hypothesis=f'Bounded equal-wall control tuning: {name}; 180s training deadline then the same262144-token canonical CPLM validation. Full control target claims require10M tokens.'))
        status=wait(record);loss=status['result']['val_loss']
        trials.append(dict(name=name,seed=0,run_id=record['run_id'],val_loss=loss,wall_seconds=status['result']['complete_wall_seconds']))
        note(record['run_id'],'decision',f'Partial canonical validation={loss:.8f}; completed fixed-wall screen. Compare with other recipe screens; no full-target claim.')
        save(STATE/'tuning.json',dict(updated_utc=now(),trials=trials))
    finalists=sorted(trials,key=lambda x:x['val_loss'])[:2]
    for candidate in finalists:
        name=candidate['name']
        record=start(SimpleNamespace(config=str(ROOT/'configs/control393'/f'{name}.json'),
            kind='screen',label=name+'-s1',seed=1,hypothesis=f'Independent-seed short repeat of finalist {name}; same180-second deadline/262144-token validation. Check whether the observed recipe advantage repeats before freezing controls.'))
        status=wait(record);loss=status['result']['val_loss']
        trials.append(dict(name=name,seed=1,run_id=record['run_id'],val_loss=loss,wall_seconds=status['result']['complete_wall_seconds']))
        note(record['run_id'],'decision',f'Finalist seed1 partial CE={loss:.8f}; retain both seed scores for selection, not a full-target claim.')
        save(STATE/'tuning.json',dict(updated_utc=now(),trials=trials))
    means={c['name']:sum(t['val_loss'] for t in trials if t['name']==c['name'])/2 for c in finalists}
    winner=min(means,key=means.get)
    config=read(ROOT/'configs/control393'/f'{winner}.json')
    for key in ['screen_seconds','screen_validation_tokens','screen_name']:config.pop(key,None)
    config.update(control_recipe_frozen_utc=now(),selected_short_recipe=winner,control_seeds=[11,22,33],
                  first_validation_seconds=360.,validation_interval_seconds=120.)
    save(ROOT/'configs/control393/frozen.json',config)
    used=60+sum(read(p,{}).get('elapsed_seconds',0) for p in (STATE/'runs').glob('*/status.json')
               if read(p,{}).get('kind') in ('pilot','smoke','screen'))
    assert used<=1800
    save(STATE/'tuning.json',dict(updated_utc=now(),state='frozen',trials=trials,finalist_means=means,
                                selected=winner,short_test_seconds_including_reserve=used))
    note('cohort','selection',f'Freeze {winner}: two-seed short-screen mean CE={means[winner]:.8f}; cumulative short tests+numerical reserve {used:.1f}/1800s. Independent full controls use seeds11,22,33 and identical frozen source/config.')

if __name__=='__main__':main()
