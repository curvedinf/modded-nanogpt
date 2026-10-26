"""Read-only precision/status report; reserved text is never a tuning signal."""
import argparse
from pathlib import Path
from wr_reproduction.probe_study import *


def report():
    registered=[r for r in events(STATE/'registry.jsonl') if r.get('event')=='registered']
    points=[];runs=[]
    for row in registered:
        directory=STATE/'runs'/row['run_id'];cfg=read(directory/'run.json',{})
        if cfg.get('study')!='probe-study' or not cfg.get('continuous_from_initialization'):continue
        series=[]
        for event in events(directory/'events.jsonl'):
            p=point(event,row['run_id'])
            if p:series.append(p)
        points.extend(series)
        result=read(directory/'result.json')
        runs.append(dict(run_id=row['run_id'],kind=cfg['kind'],architecture=cfg['recipe']['architecture'],
                         code_commit=cfg['code_commit'],status=read(directory/'status.json',{}).get('status'),
                         probes=series,validation=result.get('validation') if result else None))
    comparisons=[]
    for run in runs:
        for p in run['probes']:
            d=decision(p,points,False,max_age=read(PROTOCOL)['max_reference_age_seconds'])
            comparisons.append(dict(run_id=run['run_id'],kind=run['kind'],step=p['step'],loss=p['loss'],
                                    checkpoint_elapsed_seconds=p['wall_seconds'],diagnostic_only=True,**d))
    report=dict(utc=now(),study='probe-study',runs=runs,comparisons=comparisons,
                unresolved_precision=[dict(run_id=r['run_id'],step=r['step'],uncertainty=r['paired_text_uncertainty'])
                    for r in comparisons if r.get('paired_text_uncertainty',{}).get('resolves_001') is False],
                interpretation='Actual paired-block uncertainty at nonfuture compatible times. Not runtime-repeat noise; no interpolation. Exempt references/verification never stopped by this read-only report.',
                reserved_text_excluded=True)
    atomic(STATE/'analysis.json',report)
    return report

if __name__=='__main__':
    value=report()
    print(__import__('json').dumps(dict(utc=value['utc'],runs=[dict(run_id=r['run_id'],status=r['status'],probes=len(r['probes']),validation=r['validation']) for r in value['runs']],
                                      comparisons=len(value['comparisons']),unresolved_precision=len(value['unresolved_precision']))))
