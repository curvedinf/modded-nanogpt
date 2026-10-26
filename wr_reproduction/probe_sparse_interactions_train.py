"""Bounded sparse three-partner interactions on the frozen continuous-clock V2 protocol."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import time
from unittest.mock import patch
import torch
import triton as tr
from wr_reproduction import ks3_moe as core,ks3_sparse_interactions as local,probe_train
from wr_reproduction.ks3_update_observer import UpdateObserver,specifications,evenly
from wr_reproduction.probe_gradient_extension import ObservedRunner
from wr_reproduction.probe_study import read,atomic,stopped,assert_fence


class SparseInteractionsObserver(UpdateObserver):
    def capture(self,do_adam,allow_missing=False):
        if self.specs is None:
            self.specs=specifications(self.model)
            item=next(x for x in self.specs if x["key"]=="routed.coefficients")
            self.specs.remove(item);groups=item["indices"].reshape(-1,6)
            for i,name in enumerate(("spine_c0","spine_c1","spine_c2","spine_c3","warp_c4","warp_c5")):
                self.specs.append(dict(item,key="routed."+name,component=name,
                    indices=groups[:,i],total_elements=item["total_elements"]//6))
            p=self.model.ks3_interaction
            indices=evenly(p.numel()//3,p.device)
            for k,partner in enumerate((1,2,4)):
                self.specs.append(dict(key='routed.interaction_xor'+str(partner),parameter='ks3_interaction',tensor=p,
                    indices=indices*3+k,total_elements=p.numel()//3,branch='routed',
                    component='bounded_interaction_raw_xor'+str(partner),active_slots=None))
        return super().capture(do_adam,allow_missing)


class SparseInteractionsRunner(ObservedRunner):
    def __init__(self,version,out,ks3=False,ks3_precision='bf16',research_run=None):
        cfg=read(research_run/'run.json');recipe=cfg['recipe'];settings=recipe['sparse_interactions']
        if settings['kind']!='bounded-sparse-three-partner-interactions-v1' or settings['initial_raw_coefficient']!=0:
            raise ValueError('Unsupported sparse-interactions initialization/parameterization')
        if recipe['architecture']!='ks3_experts' or recipe['rank']!=1 or not recipe['training_gs32']:
            raise ValueError('Only original dense MLPs plus GS32 independent rank1 experts are covered')
        local.configure(settings)
        base=core.install
        with patch.object(core,'install',lambda *a,**kw:local.install(base,settings,*a,**kw)):
            super().__init__(version,out,ks3,ks3_precision,research_run)
        self.tm.step_optimizers=self.update_observer.original
        self.update_observer=SparseInteractionsObserver(self.model,self.tm,self.log)
        parameter=self.model.ks3_interaction;pcfg=self.tm.optimizer.param_cfgs[parameter]
        assert parameter.dtype==torch.float32 and parameter.shape==(5,8,1024,3)
        assert torch.count_nonzero(parameter)==0
        assert pcfg.optim=='adam' and pcfg.lr_mul==settings['lr_mul'] and pcfg.wd_mul==0
        assert list(pcfg.adam_betas)==[.9,.99]
        self.log(event='sparse_interactions_architecture',independent_rank_one_paths=True,cross_coordinate_interaction=True,partner_offsets=[1,2,4],
            bound=settings['bound'],additional_parameters=parameter.numel(),initial_coefficients_zero=True,
            geometry_lr=recipe['geometry']['lr_mul'],interaction_lr=settings['lr_mul'],
            five_original_dense_MLPs_retained=True,shared_backend_unchanged=True,
            residual_placement=settings.get('placement','before-GS32'),
            absolute_residual_bound=3.,independent_rank_one_base_preserved=True,
            basis='u(c_i)*u(c_j), u(c)=tanh(clamp(c,-65504,65504)/4); j=i xor1/2/4',
            formula='h_i=BF16(independent_GS32_BF16_h_i+sum_k tanh(raw_ik)*u(c_i)*u(c_(i xor2^k))))')


@contextmanager
def diagnostics():
    """Read actual sparse interaction activation effect on one probe chunk, on its clock."""
    from wr_reproduction import probe_diagnostics as diagnostic
    original_report=diagnostic.activation_report
    @contextmanager
    def observe(enabled,output):
        if not enabled:
            yield;return
        original=local.forward
        def capture(*a,**kw):
            result=original(*a,**kw);y,x,c,h=result
            _,src,tiles,up,down,g,raw,bound,*rest=a
            m,n=c.shape;bm=64 if core.BF16_TUNED_KERNELS else 16
            baseline=torch.empty_like(h)
            core._geometry[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,baseline,tiles,m,n,BM=bm,RANK=1,GS32=True,num_warps=4)
            index=evenly(c.numel(),c.device)
            index=index[src[index//n]<a[0].shape[0]]
            base_sample=baseline.reshape(-1)[index].float()
            ci=c.reshape(-1)[index].clamp(-65504,65504)
            expert=tiles[(index//n)//128].long()
            ui=(ci/4.).tanh()
            terms=[]
            for k,partner in enumerate((1,2,4)):
                cj=c.reshape(-1)[index^partner].clamp(-65504,65504)
                terms.append(bound*raw[expert,index%n,k].tanh()*ui*(cj/4.).tanh())
            term=torch.stack(terms).sum(0)
            delta=h.reshape(-1)[index].float()-baseline.reshape(-1)[index].float()
            values=torch.stack((diagnostic.moments(c),diagnostic.moments(h),diagnostic.moments(y)))
            effect=torch.stack((diagnostic.moments(term),diagnostic.moments(delta),
                diagnostic.moments(base_sample)))
            partner_effect=torch.stack([diagnostic.moments(value) for value in terms])
            zero_base=base_sample==0
            zero_activation=torch.stack((zero_base.float().mean(),((delta!=0)&zero_base).float().mean(),
                ((delta!=0)&zero_base).sum().float()/zero_base.sum().clamp_min(1)))
            output.append(dict(branch='routed',values=values,residual_effect=effect,partner_effect=partner_effect,zero_activation=zero_activation))
            return result
        with patch.object(local,'forward',capture):yield
    def report(rows):
        result=original_report(rows)
        for row,out in zip(rows,result,strict=True):
            if 'residual_effect' in row:
                values=row['residual_effect'].cpu().tolist()
                keys=('mean','rms','max_abs','nonzero_fraction')
                out['sparse_interactions_term_after_independent_GS32']=dict(zip(keys,values[0]))
                out['residual_placement']='after-independent-GS32'
                out['per_partner_term_before_BF16']={str(partner):dict(zip(keys,v)) for partner,v in zip((1,2,4),row['partner_effect'].cpu().tolist())}
                out['independent_feature_sample']=dict(zip(keys,values[2]))
                out['zero_independent_feature_activation']=dict(zip(('zero_base_fraction','newly_activated_fraction','fraction_of_zero_base_activated'),row['zero_activation'].cpu().tolist()))
                out['relative_feature_change_rms']=values[1][1]/max(values[2][1],1e-12)
                out['packed_feature_change_due_to_residual']=dict(zip(keys,values[1]))
                out['residual_effect_scope']='Same live coordinates/geometry with and without sparse interactions, first probe chunk; final BF16 core features precede gate and post multiplier'
        return result
    with patch.object(diagnostic,'activations',observe),patch.object(diagnostic,'activation_report',report):yield


def gate(r):
    """Eight actual updates, fixed probe and even/odd continuation after24 warmups."""
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from wr_reproduction.probe_continuation import baseline_and_restore,compare
    from wr_reproduction.probe_calibration import state,oracle
    from wr_reproduction.probe_precision_calibration import audit
    from wr_reproduction.probe_evaluation import FixedProbe
    from wr_reproduction.matched_architecture import audit_model
    m=r.model;cfg=r.research_config
    m.limit_yarn_rebuild(r.max_tokens)
    m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size,0,1))
    for sampler in r.samplers:sampler.set_prefix_table(m.prefix_table.cpu().numpy())
    batches=r.loaders(list(range(12)))
    seconds=[]
    for step in range(8):
        if stopped():raise RuntimeError('Explicit study STOP')
        torch.cuda.synchronize();t=time.perf_counter()
        loss=r.step(step,batches,next_step=step+1,prefetch=True);r.tail.tick(step)
        torch.cuda.synchronize();seconds.append(time.perf_counter()-t)
        assert torch.isfinite(loss).item()
    learned=m.ks3_interaction.abs().amax(dim=(1,2))
    assert (learned>0).all(),'A routed site has no sparse-interactions coefficient learning'
    assert core.EXECUTED_MOE_BACKWARDS==set(range(5)),core.EXECUTED_MOE_BACKWARDS
    inventory=audit_model(m,'ks3_experts');assert inventory['active_conventional_mlp_instances']==5
    r.tm.advance_schedule(8)
    reference=baseline_and_restore(r,batches,8)
    probe=FixedProbe(r,cfg['probe_protocol'],cfg['backend_hash'])
    before=state(r);cold=probe.observe(8,time.perf_counter());restoration=audit(before,state(r))
    independent=oracle(probe)
    oracle_error=max(abs(a-b) for a,b in zip(cold['chunk_losses'],independent));assert oracle_error<1e-6
    warm=probe.observe(8,time.perf_counter())
    assert cold['chunk_losses']==warm['chunk_losses']
    r.update_observer.arm(2)
    loss=r.step(8,batches,next_step=9,prefetch=True);r.tail.tick(8)
    torch.cuda.synchronize();parity=compare(r,loss.item(),reference);del reference
    assert parity['all_parameters_bitwise_equal'],parity['maximum_parameter_relative_rms']
    assert any(r.tables.pending) and any(r.gradients.pending)
    pending_before=state(r);pending=probe.observe(9,time.perf_counter());pending_restoration=audit(pending_before,state(r))
    loss=r.step(9,batches,prefetch=False);r.tail.tick(9)
    torch.cuda.synchronize();assert torch.isfinite(loss).item()
    rows=r.update_observer.reports;assert len(rows)==2
    gradients=next(row for row in rows if row['do_adam'])['measurements']
    for partner in (1,2,4):
        item=gradients['routed.interaction_xor'+str(partner)]
        assert item['optimizer_applied'] and item['gradient_sample_rms']>0
    projected=cold['probe_seconds']+6*warm['probe_seconds']+7*r.update_observer.total_seconds
    assert projected/900<.03,('Sparse-interaction probe/diagnostic overhead exceeds budget',projected)
    result=dict(passed=True,mode='sparse-interactions-graph',full_graph_warmup=True,original_logical_ranks=8,
        residual_placement=cfg['recipe']['sparse_interactions'].get('placement','before-GS32'),
        actual_prefix_updates=8,actual_prefix_update_seconds=seconds,not_a_speed_or_convergence_win=True,
        residual_parameters=tuple(m.ks3_interaction.shape),residual_raw_max_abs=m.ks3_interaction.abs().max().item(),
        per_site_residual_learned_absmax=learned.cpu().tolist(),
        routed_backward_sites=sorted(core.EXECUTED_MOE_BACKWARDS),five_original_dense_MLPs_retained=True,
        fixed_probe_oracle_max_abs=oracle_error,state_restoration=restoration,pending_state_restoration=pending_restoration,
        continuation_parity={k:v for k,v in parity.items() if k!='parameters'},odd_continuation_loss=loss.item(),
        cold_probe_seconds=cold['probe_seconds'],warm_probe_seconds=warm['probe_seconds'],
        pending_probe_seconds=pending['probe_seconds'],projected_seven_probe_and_sparse_diagnostic_seconds=projected,
        projected_fraction_of900=projected/900,projection_limitation='Screen/ordinary instrumentation costs also remain on clock; actual run overhead must be inspected.',
        gradient_measurements=rows,activation_effect_diagnostics=cold['ks3_activation_diagnostics'],
        signature_hash=cold['signature_hash'],not_a_screening_reference=True,full_convergence_verified=False)
    atomic(r.research_run/'calibration.json',result);r.log(event='calibration_complete',**result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--gate',action='store_true');a=parser.parse_args()
    cfg=read(a.run/'run.json')
    if any(Path(f).exists() for f in cfg['stop_files']):raise RuntimeError('Explicit study STOP')
    assert_fence(cfg['recipe'])
    from wr_reproduction.probe_controller import sources
    if sources()!=cfg['source_hashes']:raise RuntimeError('Source hashes changed before launch')
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(1)
    from wr_reproduction.local_collectives import LocalCollectives
    with LocalCollectives(),diagnostics():
        r=SparseInteractionsRunner('pr393',a.run/'events.jsonl',True,'bf16',a.run)
        from wr_reproduction.matched_architecture import audit_model
        r.log(event='feedforward_inventory',**audit_model(r.model,'ks3_experts'))
        r.warmup()
        if a.gate:gate(r)
        else:probe_train.train(r)

if __name__=='__main__':main()
