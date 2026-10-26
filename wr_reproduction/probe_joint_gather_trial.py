"""One directed KS3-only joint-rate trial using the tested exact BF16 kernel.

Scalar hypotheses use full graph warmup and the frozen held-out screen, without
repeating a100-update kernel gate. Reuse requires all prior implementation
hashes unchanged; this small control entry point is the sole allowed addition.
"""
import argparse
import copy
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_bf16_gather_trial as kernel_trial
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import *

SELF='wr_reproduction/probe_joint_gather_trial.py'


def normalized(value):
    result=copy.deepcopy(value)
    for key in ('name','hypothesis','hypothesis_class','joint_rate_evidence'):
        result.pop(key,None)
    result['geometry']['lr_mul']=.125
    result.pop('projection_lr_multiplier',None)
    return result


def compatible_gate(mode,value):
    kernel_trial.validate(value)
    if value['geometry']['lr_mul']!=.1328125 or value.get('projection_lr_multiplier')!=1.03125:
        raise ValueError('Only the explicit small joint-rate hypothesis is authorized here')
    gate=read(RESULTS/('bf16-gather-'+mode+'-gate.json'),{})
    if not gate.get('passed'):raise RuntimeError('Missing passed BF16 gather '+mode+' gate')
    previous=gate['recipe']
    if previous['geometry']['lr_mul']!=.125 or previous.get('projection_lr_multiplier',1.)!=1.:
        raise RuntimeError('Unexpected kernel gate origin')
    if normalized(value)!=normalized(previous):
        raise RuntimeError('More than the explicit KS3 optimizer rates changed')
    current=control.sources();frozen=gate['source_hashes']
    for path,digest in frozen.items():
        if current.get(path)!=digest:raise RuntimeError('Prior implementation changed: '+path)
    if set(current)-set(frozen)!={SELF}:
        raise RuntimeError('Unexpected additions since numerical/model gate')
    return dict(mode=mode,run_id=gate['run_id'],code_commit=gate['code_commit'],
                recipe_scope='same kernel and architecture; geometry/projection rates only',
                prior_sources_unchanged=True,sole_new_control_source=SELF,
                new_rate_recipe_was_not_the_calibration_recipe=True)


def work(request):
    value=read(request)
    if value['mode']!='discovery':raise ValueError('No redundant scalar timing/calibration run')
    for mode in ('numerical','speed_state'):compatible_gate(mode,value['recipe'])
    # The original lease, owner, trainer entry point and verification machinery
    # remain in the kernel trial. Only gate compatibility changes for this case.
    kernel_trial.require_gate=compatible_gate
    kernel_trial.work(request)


def launch(path):
    if stopped():raise RuntimeError('Explicit study STOP')
    value=read(path)
    reuse=[compatible_gate(mode,value) for mode in ('numerical','speed_state')]
    unit='wr-probe-joint-gather-'+uuid.uuid4().hex[:8]
    request=STATE/'requests'/(unit+'.json')
    atomic(request,dict(utc=now(),unit=unit,mode='discovery',recipe=value,kernel_gate_reuse=reuse))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
        '--property=KillMode=control-group','--property=TimeoutStopSec=15',
        '--property=PartOf=wr-ks3-probe-research.service',*systemd_environment(),
        str(control.PYTHON),'-m','wr_reproduction.probe_joint_gather_trial','work',str(request)],check=True)
    print(unit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','work']);p.add_argument('path',type=Path);a=p.parse_args()
    if a.action=='launch':launch(a.path)
    else:
        try:work(a.path)
        except BaseException:
            value=dict(utc=now(),error=traceback.format_exc());atomic(STATE/'joint-gather-blocker.json',value)
            publish('joint-gather-status.json',dict(status='error',**value));raise
