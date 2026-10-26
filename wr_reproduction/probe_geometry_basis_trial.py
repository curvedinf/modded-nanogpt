"""One explicit geometry-coordinate hypothesis, isolated from frozen probing."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
import uuid
from wr_reproduction import probe_controller as control
from wr_reproduction.probe_study import *

KIND='cubic-chebyshev-normalized-to-hermite-v1'


def validate(value):
    assert_fence(value)
    baseline=read(ROOT/'configs/wr_reproduction/probe_ks3_bf16_gather_exact.json')
    omitted={'name','hypothesis','hypothesis_class','geometry_parameterization','basis_evidence'}
    if {k:v for k,v in value.items() if k not in omitted}!={k:v for k,v in baseline.items() if k not in omitted}:
        raise ValueError('Only the explicit KS3 geometry coordinate change is allowed')
    if value.get('geometry_parameterization')!=KIND:raise ValueError('Unsupported geometry parameterization')


def create_run(value,seed,kind,unit,calibration=None,frozen=None):
    if stopped():raise RuntimeError('Explicit study STOP')
    validate(value);fence=assert_fence(value)
    dirty=control.git('status','--porcelain','--','wr_reproduction','configs/wr_reproduction','ks3_research/vendor')
    if dirty:raise RuntimeError('Commit owned implementation/config before launch: '+dirty)
    code=control.sources()
    if frozen and code!=frozen['source_hashes']:raise RuntimeError('Verification source changed')
    rid=value['name']+'-'+kind+'-'+now().replace('-','').replace(':','').split('.')[0]+'-'+uuid.uuid4().hex[:6]
    directory=STATE/'runs'/rid;directory.mkdir(parents=True)
    command=[str(control.PYTHON),'-m','wr_reproduction.probe_geometry_basis_trial','train',str(directory),'--mode',calibration or 'discovery']
    head=control.git('rev-parse','HEAD')
    run=dict(study='probe-study',run_id=rid,directory=str(directory),seed=seed,kind=kind,recipe=value,
        command=command,unit=unit,created_utc=now(),source_hashes=code,
        code_commit=frozen['code_commit'] if frozen else head,launch_head_commit=head,
        environment=environment(),protocol_path=str(PROTOCOL),fence_path=str(FENCE),
        backend_hash=fence['backend_hash'],backend_fence=fence,probe_protocol=read(PROTOCOL),
        validation_data=base.validation_data(),continuous_from_initialization=not calibration,
        comparative_pruning_exempt=kind=='verification',stop_files=[str(p) for p in STOP_FILES],
        calibration=calibration,geometry_parameterization=KIND,
        scope='Original independent rank1 experts with changed cubic optimizer coordinates; original coefficient quantization/nonlinearity/products, exact BF16 gather and five original dense MLPs retained')
    atomic(directory/'run.json',run);atomic(directory/'config.json',value);atomic(RESULTS/(rid+'-config.json'),run)
    append(STATE/'registry.jsonl',dict(event='registered',run_id=rid,directory=str(directory),kind=kind,seed=seed,code_commit=run['code_commit']))
    control.status(run,'registered');control.note(run,hypothesis=value['hypothesis'],decision='Launch '+kind,code_commit=run['code_commit'])
    return run


def runner_class():
    import torch
    from wr_reproduction.probe_gradient_extension import ObservedRunner
    from wr_reproduction.ks3_update_observer import UpdateObserver,specifications
    class BasisObserver(UpdateObserver):
        def capture(self,do_adam,allow_missing=False):
            if self.specs is None:
                self.specs=specifications(self.model)
                item=next(x for x in self.specs if x['key']=='routed.coefficients')
                self.specs.remove(item);groups=item['indices'].reshape(-1,6)
                for i,name in enumerate(('T0','T1','T2','T3','warp_T2','warp_T3')):
                    self.specs.append(dict(item,key='routed.'+name,component=name,
                        indices=groups[:,i],total_elements=item['total_elements']//6))
            return super().capture(do_adam,allow_missing)
    class BasisRunner(ObservedRunner):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.tm.step_optimizers=self.update_observer.original
            self.update_observer=BasisObserver(self.model,self.tm,self.log)
            p=self.model.ks3_geometry;cfg=self.tm.optimizer.param_cfgs[p]
            assert p.shape==(5,8,1024,8) and p.dtype==torch.float32
            assert torch.count_nonzero(p[...,:6])==0
            assert cfg.optim=='adam' and cfg.lr_mul==.125 and cfg.wd_mul==0 and list(cfg.adam_betas)==[.9,.99]
            self.log(event='geometry_basis_architecture',parameter_columns=['T0','T1','T2','T3','warp_T2','warp_T3','raw_scale','raw_shift'],
                initial_function_unchanged=True,independent_rank_one_paths=True,parameter_count_unchanged=True,
                original_coefficient_GS32_and_FP16_nonlinearity=True,five_original_dense_MLPs=True,
                no_graph_owned_buffers_added=True,not_a_new_function_class=True)
        def log(self,**record):
            record['geometry_parameterization']=KIND
            if record.get('event')=='terminal_geometry':
                record['component_label_mapping']={'c0':'theta_T0','c1':'theta_T1','c2':'theta_T2','c3':'theta_T3','c4':'warp_T2','c5':'warp_T3'}
                record['artifact_coordinates']='optimizer parameters; first four entries are not physical Hermite coefficients'
            return super().log(**record)
    return BasisRunner


def diagnostics():
    """Small same-coordinate basis/effect audit during the existing first chunk."""
    import torch
    from contextlib import contextmanager
    from unittest.mock import patch
    from wr_reproduction import ks3_moe as core,probe_diagnostics as diagnostic
    from wr_reproduction.ks3_update_observer import evenly
    from wr_reproduction.ks3_geometry_basis import transform
    original_report=diagnostic.activation_report
    @contextmanager
    def observe(enabled,output):
        if not enabled:
            yield;return
        original=core.forward
        def capture(*a,**kw):
            result=original(*a,**kw);y,x,c,h=result
            rows,src,tiles,up,down,theta,*rest=a;m,n=c.shape
            index=evenly(c.numel(),c.device);row=index//n;col=index%n
            valid=src[row]<rows.shape[0];expert=tiles[row//128].long()
            coord=c.reshape(-1)[index].clamp(-65504,65504);t=4*torch.tanh(coord/4);u=t/4
            g=transform(theta)[expert,col];cs=g[:,:6]
            scale=(cs.abs().amax(-1)/127).clamp(2**-24,65504).half().float()
            cs=(cs/scale[:,None]).round().clamp(-127,127)*scale[:,None]
            c0,c1,c2,c3,c4,c5=cs.unbind(-1)
            z=torch.tanh((torch.nn.functional.softplus(g[:,6])+1e-4)*(t+.5*torch.tanh(g[:,7])))
            old=torch.stack((2*t**3-3*t**2+1,-2*t**3+3*t**2,t**3-2*t**2+t,t**3-t**2),-1)
            new=torch.stack((torch.ones_like(u),u,2*u*u-1,4*u**3-3*u),-1)
            spine=((2*c0-2*c1+c2+c3)*t+(-3*c0+3*c1-2*c2-c3))*t*t+c2*t+c0
            warp=c4*(2*z*z-1)+c5*(4*z*z*z-3*z)
            weight=valid.float();den=weight.sum().clamp_min(1)
            rms=lambda v:((v.square()*weight[:,None]).sum(0)/den).sqrt()
            effects=torch.stack((coord,spine,warp),-1)
            values=torch.stack((diagnostic.moments(c),diagnostic.moments(h),diagnostic.moments(y)))
            output.append(dict(branch='routed',values=values,basis_rms=torch.cat((rms(old),rms(new),rms(effects)))))
            return result
        with patch.object(core,'forward',capture):yield
    def report(rows):
        result=original_report(rows)
        for row,out in zip(rows,result,strict=True):
            if 'basis_rms' in row:
                values=row['basis_rms'].cpu().tolist()
                out['geometry_basis_effect']=dict(original_Hermite_basis_rms_same_coordinates=values[:4],
                    normalized_Chebyshev_basis_rms=values[4:8],coordinate_rms=values[8],
                    smooth_quantized_cubic_deformation_rms=values[9],smooth_quantized_warp_deformation_rms=values[10],
                    scope='4096 deterministic sampled feature entries on first probe chunk; padding excluded. Smooth FP32 deformations at quantized physical coefficients, not measured packed activation deltas or post/router effects.')
        return result
    @contextmanager
    def context():
        with patch.object(diagnostic,'activations',observe),patch.object(diagnostic,'activation_report',report):yield
    return context()


def graph_gate(r):
    import time
    import torch
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_continuation import baseline_and_restore,compare
    from wr_reproduction.probe_calibration import state,oracle
    from wr_reproduction.probe_precision_calibration import audit
    from wr_reproduction.probe_evaluation import FixedProbe
    from wr_reproduction import ks3_moe as core
    from wr_reproduction.matched_architecture import audit_model
    m=r.model;cfg=r.research_config;m.limit_yarn_rebuild(r.max_tokens)
    m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size,0,1))
    for sampler in r.samplers:sampler.set_prefix_table(m.prefix_table.cpu().numpy())
    batches=r.loaders(list(range(12)));seconds=[]
    for step in range(8):
        if stopped():raise RuntimeError('Explicit study STOP')
        torch.cuda.synchronize();start= time.perf_counter()
        loss=r.step(step,batches,next_step=step+1,prefetch=True);r.tail.tick(step)
        torch.cuda.synchronize();seconds.append(time.perf_counter()-start);assert torch.isfinite(loss).item()
    assert m.ks3_geometry[...,:4].abs().max()>0
    assert core.EXECUTED_MOE_BACKWARDS==set(range(5))
    assert audit_model(m,'ks3_experts')['active_conventional_mlp_instances']==5
    r.tm.advance_schedule(8);reference=baseline_and_restore(r,batches,8)
    probe=FixedProbe(r,cfg['probe_protocol'],cfg['backend_hash'])
    before=state(r);cold=probe.observe(8,time.perf_counter());restoration=audit(before,state(r))
    independent=oracle(probe);error=max(abs(a-b) for a,b in zip(cold['chunk_losses'],independent));assert error<1e-6
    warm=probe.observe(8,time.perf_counter());assert cold['chunk_losses']==warm['chunk_losses']
    r.update_observer.arm(2)
    loss=r.step(8,batches,next_step=9,prefetch=True);r.tail.tick(8);torch.cuda.synchronize()
    parity=compare(r,loss.item(),reference);del reference
    assert parity['all_parameters_bitwise_equal'],parity['maximum_parameter_relative_rms']
    assert any(r.tables.pending) and any(r.gradients.pending)
    before=state(r);pending=probe.observe(9,time.perf_counter());pending_restoration=audit(before,state(r))
    loss=r.step(9,batches,prefetch=False);r.tail.tick(9);torch.cuda.synchronize();assert torch.isfinite(loss).item()
    reports=r.update_observer.reports;assert len(reports)==2
    for name in ('T0','T1','T2','T3'):
        row=reports[-1]['measurements']['routed.'+name]
        assert row['optimizer_applied'] and row['gradient_sample_rms']>0
    projected=cold['probe_seconds']+6*warm['probe_seconds']+7*r.update_observer.total_seconds
    assert projected/900<.03,projected
    value=dict(passed=True,mode='graph',full_graph_warmup=True,original_logical_ranks=8,
        actual_prefix_updates=8,actual_prefix_update_seconds=seconds,not_a_speed_or_convergence_win=True,
        geometry_parameterization=KIND,routed_backward_sites=sorted(core.EXECUTED_MOE_BACKWARDS),
        five_original_dense_MLPs_retained=True,parameter_count_unchanged=True,
        fixed_probe_oracle_max_abs=error,state_restoration=restoration,pending_state_restoration=pending_restoration,
        continuation_parity={k:v for k,v in parity.items() if k!='parameters'},odd_continuation_loss=loss.item(),
        cold_probe_seconds=cold['probe_seconds'],warm_probe_seconds=warm['probe_seconds'],pending_probe_seconds=pending['probe_seconds'],
        projected_seven_probe_and_diagnostic_seconds=projected,projected_fraction_of900=projected/900,
        projection_limitation='Screen/ordinary instrumentation also remains on clock; inspect actual run overhead.',
        gradient_measurements=reports,activation_effect_diagnostics=cold['ks3_activation_diagnostics'],
        signature_hash=cold['signature_hash'],not_a_screening_reference=True,full_convergence_verified=False)
    atomic(r.research_run/'calibration.json',value);r.log(event='calibration_complete',**value)


def train(directory,mode):
    import torch
    cfg=read(directory/'run.json');validate(cfg['recipe'])
    if stopped():raise RuntimeError('Explicit study STOP')
    if control.sources()!=cfg['source_hashes']:raise RuntimeError('Source changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    if mode=='numerical':
        from wr_reproduction.check_ks3_geometry_basis import check
        value=check();atomic(directory/'calibration.json',value);append(directory/'events.jsonl',dict(event='calibration_complete',**value));return
    from wr_reproduction.ks3_geometry_basis import install
    install()
    from wr_reproduction.local_collectives import LocalCollectives
    from wr_reproduction import probe_train
    with LocalCollectives(),diagnostics():
        r=runner_class()('pr393',directory/'events.jsonl',True,'bf16',directory)
        from wr_reproduction.matched_architecture import audit_model
        r.log(event='feedforward_inventory',**audit_model(r.model,'ks3_experts'));r.warmup()
        if mode=='graph':graph_gate(r)
        else:probe_train.train(r)


def require_gate(mode,value):
    gate=read(RESULTS/('geometry-basis-'+mode+'-gate.json'),{})
    if not gate.get('passed'):raise RuntimeError('Missing passed geometry basis '+mode+' gate')
    if gate['recipe']!=value or gate['source_hashes']!=control.sources():raise RuntimeError('Recipe or source changed since gate')


def work(request):
    cfg=read(request);mode=cfg['mode'];value=cfg['recipe']
    with (ROOT/'logs/wr-reproduction/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopped():return
        if mode!='numerical':require_gate('numerical',value)
        if mode=='discovery':require_gate('graph',value)
        run=create_run(value,169,'discovery' if mode=='discovery' else 'calibration',cfg['unit'],None if mode=='discovery' else mode)
        result=control.execute(run)
        if mode!='discovery':
            gate=dict(utc=now(),run_id=run['run_id'],code_commit=run['code_commit'],recipe=value,source_hashes=run['source_hashes'],
                passed=result['status']=='calibration_complete' and bool((result.get('calibration') or {}).get('passed')),result=result)
            atomic(RESULTS/('geometry-basis-'+mode+'-gate.json'),gate);atomic(STATE/('geometry-basis-'+mode+'-gate.json'),gate)
        if result['qualifies']:
            control.create_run=create_run;control.verification(run,result,cfg['unit'])


def launch(path,mode):
    if stopped():raise RuntimeError('Explicit study STOP')
    value=read(path);validate(value)
    if mode!='numerical':require_gate('numerical',value)
    if mode=='discovery':require_gate('graph',value)
    unit='wr-probe-geometry-basis-'+uuid.uuid4().hex[:8];request=STATE/'requests'/(unit+'.json')
    atomic(request,dict(utc=now(),unit=unit,mode=mode,recipe=value))
    subprocess.run(['systemd-run','--unit='+unit,'--working-directory='+str(ROOT),
        '--property=KillMode=control-group','--property=TimeoutStopSec=15','--property=PartOf=wr-ks3-probe-research.service',
        *systemd_environment(),str(control.PYTHON),'-m','wr_reproduction.probe_geometry_basis_trial','work',str(request)],check=True)
    print(unit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','work','train']);p.add_argument('path',type=Path)
    p.add_argument('--mode',choices=['numerical','graph','discovery'],default='numerical');a=p.parse_args()
    if a.action=='launch':launch(a.path,a.mode)
    elif a.action=='train':train(a.path,a.mode)
    else:
        try:work(a.path)
        except BaseException:
            value=dict(utc=now(),error=traceback.format_exc());atomic(STATE/'geometry-basis-blocker.json',value)
            publish('geometry-basis-status.json',dict(status='error',**value));raise
