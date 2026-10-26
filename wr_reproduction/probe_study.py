"""Isolated fixed-text protocol and CPU policy. Historical curves are never loaded."""
import math
import os
import statistics
from decimal import Decimal
from pathlib import Path
from wr_reproduction.research import ROOT,atomic,append,read,sha,digest,now,events
from wr_reproduction import research as base
from wr_reproduction.matched_study import SHARED

ROOT_STATE=ROOT/'logs/wr-reproduction/probe-study'
ROOT_RESULTS=ROOT/'research_results/wr_reproduction/probe-study'
STATE=Path(os.environ.get('WR_PROBE_STATE',ROOT_STATE)).resolve()
RESULTS=Path(os.environ.get('WR_PROBE_RESULTS',ROOT_RESULTS)).resolve()
MANAGER=Path(os.environ.get('WR_PROBE_MANAGER',ROOT/'logs/wr-reproduction/probe-manager'))
PUBLIC_STATE=Path(os.environ.get('WR_PROBE_PUBLIC_STATE',ROOT_STATE)).resolve()
STOP_FILES=list(dict.fromkeys(p/'STOP' for p in (ROOT_STATE,ROOT_RESULTS,MANAGER,STATE,RESULTS,PUBLIC_STATE)))
FENCE=Path(os.environ.get('WR_PROBE_FENCE',ROOT/'configs/wr_reproduction/probe_backend.json')).resolve()
PROTOCOL=Path(os.environ.get('WR_PROBE_PROTOCOL',ROOT/'configs/wr_reproduction/probe_protocol.json')).resolve()
VERIFY_SEEDS=tuple(int(s) for s in os.environ.get('WR_PROBE_VERIFY_SEEDS','269,270,271').split(','))
if len(VERIFY_SEEDS)!=3 or len(set(VERIFY_SEEDS))!=3 or 169 in VERIFY_SEEDS:
    raise ValueError('Exactly three distinct fresh verification seeds are required')
BUDGET=900.
DEFINITION='fixed-heldout-full-vocabulary-cplm-bf16-raw-v1'

def stopped():return any(p.exists() for p in STOP_FILES)

def environment():
    """Explicitly carry cohort paths through systemd; it does not inherit the CLI environment."""
    return dict(WR_PROBE_STATE=str(STATE),WR_PROBE_RESULTS=str(RESULTS),WR_PROBE_MANAGER=str(MANAGER),
                WR_PROBE_PUBLIC_STATE=str(PUBLIC_STATE),WR_PROBE_FENCE=str(FENCE),WR_PROBE_PROTOCOL=str(PROTOCOL),
                WR_PROBE_VERIFY_SEEDS=','.join(map(str,VERIFY_SEEDS)))

def systemd_environment():return ['--setenv='+k+'='+v for k,v in environment().items()]

def publish(name,value):
    """Keep the isolated cohort authoritative and mirror only live status to the parent."""
    record=dict(value,study_state=str(STATE),study_results=str(RESULTS),probe_protocol_path=str(PROTOCOL))
    for directory in dict.fromkeys((STATE,PUBLIC_STATE)):atomic(directory/name,record)

def initial_protocol():
    return dict(version=1,loss_definition=DEFINITION,tokens=65536,offset=0,chunk_tokens=8192,
                uncertainty_block_tokens=8192,reserved_offset=12582912,reserved_tokens=262144,
                interval_seconds=60.,max_reference_age_seconds=3.,attention_windows=[640,1408],
                yarn_sha256='f578f7187981f9a8220c90ea99eaaa35616aa87bf99b49cfdd2307f3628f3612',
                segmentation='original-unaligned-BOS-within-each-fixed8192-token-chunk',
                precision='bf16',weights='raw',canonical_mask=False,tail_shipping=False,
                final_window_extension=False,mtp=False,prefix=False,moe_aux=False,
                copy_eval_block=8192,softcap=[23.,5.,7.5],
                clock='continuous-perf-counter-setup-evaluation-restoration-included',
                uncertainty='paired-nonoverlapping8192-token-block-student95-halfwidth; correlated-block caveat; not runtime noise')

def freeze(protocol):
    if read(RESULTS/'calibration-gates.json',{}).get('status')!='passed':raise RuntimeError('Both architecture probe gates must pass before freezing')
    old=read(ROOT/'configs/wr_reproduction/matched_backend.json')
    for p,h in old['source_hashes'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Original shared backend changed: '+p)
    paths=[ROOT/'wr_reproduction'/name for name in ('probe_study.py','probe_train.py','probe_evaluation.py','probe_diagnostics.py','probe_controller.py','probe_queue.py','matched_study.py')]
    payload=dict(source_hashes=dict(old['source_hashes'],**{str(p.relative_to(ROOT)):sha(p) for p in paths}),
                 settings=SHARED,protocol=protocol,old_backend_hash=old['backend_hash'])
    value=dict(utc=now(),backend_hash=digest(payload),**payload,
               audit='All old fenced files byte-identical; only new measurement/control modules added')
    for path,data in ((PROTOCOL,protocol),(FENCE,value)):
        if path.exists():
            previous=read(path)
            if path==FENCE:assert previous['backend_hash']==value['backend_hash'],'Frozen instrumentation changed'
            else:assert previous==data,'Frozen protocol changed'
        else:atomic(path,data)
    atomic(RESULTS/'backend-audit.json',value)
    return value

def assert_fence(recipe):
    fence=read(FENCE)
    if not fence:raise RuntimeError('Probe protocol not frozen')
    for p,h in fence['source_hashes'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Probe/backend fence violation: '+p)
    for k,v in fence['settings'].items():
        if recipe.get(k)!=v:raise RuntimeError('Shared setting fence violation: '+k)
    return fence

class Metadata(base.Metadata):
    def validation(self,step):
        value=super().validation(step)
        value.update(probe_backend_hash=read(FENCE)['backend_hash'])
        return value

def signature(protocol,data,backend_hash,*,reserved=False):
    fields={k:v for k,v in protocol.items() if k not in ('interval_seconds','max_reference_age_seconds','uncertainty')}
    fields.update(validation_data=data,backend_hash=backend_hash,measurement=DEFINITION,
                  purpose='reserved-confirmation' if reserved else 'screening')
    if reserved:fields.update(offset=protocol['reserved_offset'],tokens=protocol['reserved_tokens'])
    fields.update(input_token_offset_range=[fields['offset'],fields['offset']+fields['tokens']],
                  target_token_offset_range=[fields['offset']+1,fields['offset']+fields['tokens']+1],
                  interval_convention='half-open-shard-token-offsets-after1024-byte-header')
    return fields

def paired(a,b):
    if len(a)!=len(b) or len(a)<2:return dict(available=False)
    differences=[x-y for x,y in zip(a,b)]
    se=statistics.stdev(differences)/math.sqrt(len(differences))
    df=len(differences)-1
    critical={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,15:2.131,20:2.086,30:2.042,60:2.001,120:1.980}
    t=critical[max((k for k in critical if k<=df),default=1)] if df<120 else 1.980
    return dict(available=True,blocks=len(differences),mean_difference=statistics.mean(differences),
                standard_error=se,normal95_halfwidth=1.96*se,student95_halfwidth=t*se,resolves_001=t*se<=.01,
                limitation='Text sampling uncertainty under independent-block approximation; adjacent blocks may correlate. Not inference nondeterminism.')

def point(row,run_id):
    if row.get('event')!='probe' or row.get('measurement')!=DEFINITION:return None
    sig=row.get('probe_signature')
    if not isinstance(sig,dict) or sig.get('measurement')!=DEFINITION or sig.get('purpose')!='screening':return None
    if row.get('signature_hash')!=digest(sig):return None
    loss,wall=row.get('loss'),row.get('checkpoint_elapsed_seconds')
    if not isinstance(loss,(int,float)) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<0:return None
    return dict(run_id=run_id,loss=loss,wall_seconds=wall,signature=sig,signature_hash=digest(sig),
                chunk_losses=row.get('chunk_losses',[]),utc=row.get('utc'),step=row.get('step'))

def decision(candidate,references,latched,*,max_age=3.,stop=False,exempt=False):
    result=dict(abort=False,below5_latched=latched,signature_hash=candidate['signature_hash'],
                candidate_elapsed_seconds=candidate['wall_seconds'])
    if stop:return dict(result,abort=True,reason='explicit_study_stop')
    if not math.isfinite(candidate['loss']):return dict(result,abort=True,reason='nonfinite_probe')
    latched=latched or candidate['loss']<5.
    result['below5_latched']=latched
    if exempt:return dict(result,reason='instrumented_reference_or_frozen_verification_exempt',comparative_pruning_exempt=True)
    if not latched:return dict(result,reason='await_strict_below5')
    latest={}
    for p in references:
        if p['run_id']==candidate['run_id'] or p['signature']!=candidate['signature'] or not math.isfinite(p['loss']):continue
        age=candidate['wall_seconds']-p['wall_seconds']
        if not 0<=age<=max_age:continue
        old=latest.get(p['run_id'])
        if old is None or p['wall_seconds']>old['wall_seconds']:latest[p['run_id']]=p
    if not latest:return dict(result,reason='no_comparable_nonfuture_probe',max_reference_age_seconds=max_age)
    best=min(latest.values(),key=lambda p:p['loss'])
    gap=candidate['loss']-best['loss']
    # Decimal avoids rejecting the exact inclusive boundary through binary subtraction.
    abort=Decimal(str(candidate['loss']))-Decimal(str(best['loss']))>=Decimal('.01')
    result.update(abort=abort,reason='first_one_sided_gap_at_least_001' if abort else 'within_one_sided_margin',
                  gap=gap,reference_run_id=best['run_id'],reference_elapsed_seconds=best['wall_seconds'],
                  reference_age_seconds=candidate['wall_seconds']-best['wall_seconds'],reference_utc=best.get('utc'),
                  reference_loss=best['loss'],reference_signature_hash=best['signature_hash'],
                  reference_evidence=best.get('evidence'),paired_text_uncertainty=paired(candidate['chunk_losses'],best['chunk_losses']))
    return result

class Screen:
    def __init__(self,run,state=STATE):
        self.directory=Path(run);self.config=read(self.directory/'run.json');self.state=Path(state)
        self.run_id=self.config['run_id'];self.latched=False
        # Recover the gate solely from this run's real fixed-text observations.
        for row in events(self.directory/'events.jsonl'):
            p=point(row,self.run_id)
            if p and math.isfinite(p['loss']) and p['loss']<5:self.latched=True
    def references(self):
        result=[]
        registered={r['run_id'] for r in events(self.state/'registry.jsonl') if r.get('event')=='registered'}
        for rid in registered-{self.run_id}:
            directory=self.state/'runs'/rid;cfg=read(directory/'run.json',{})
            if cfg.get('study')!='probe-study' or not cfg.get('continuous_from_initialization') or cfg.get('kind') not in ('control','reference','discovery','verification'):continue
            path=directory/'events.jsonl'
            rows=events(path)
            for n,row in enumerate(rows,1):
                p=point(row,rid)
                if p and math.isfinite(p['loss']):
                    p['evidence']=dict(path=str(path),line=n,row_sha256=digest(row));result.append(p)
        return result
    def observe(self,row):
        p=point(row,self.run_id)
        if p is None:raise ValueError('Only signed fixed-text probes can enter screening')
        result=decision(p,self.references(),self.latched,
                        max_age=self.config['probe_protocol']['max_reference_age_seconds'],
                        stop=any(Path(p).exists() for p in self.config['stop_files']),
                        exempt=self.config.get('comparative_pruning_exempt',False))
        self.latched=result['below5_latched']
        append(self.directory/'screening.jsonl',result)
        return result
