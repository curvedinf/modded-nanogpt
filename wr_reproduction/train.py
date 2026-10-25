"""Pinned WR training on one MI300 via eight original logical microbatches.

All training configuration is imported from the selected immutable record. This
launcher adapts execution/collectives for gradient accumulation, not the recipe.
Use --preflight to run the original warmup/capture plan without starting a control.
"""
import argparse,copy,json,time,gc,os,threading
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from wr_reproduction.backend import install
from wr_reproduction.local_collectives import LocalCollectives
from wr_reproduction.rank_state import RankStates
from wr_reproduction.virtual_tables import VirtualTables
from wr_reproduction.accumulation import OriginalGradientAccumulator
from wr_reproduction.bench_updates import Sinks
from wr_reproduction.contracts import audit

class Runner:
    def __init__(self,version,out,ks3=False,ks3_precision='bf16',research_run=None):
        self.version=version;self.out=out;self.manifest=audit(version)
        self.research_run=research_run;self.observations=None;self.log_lock=threading.Lock()
        self.research_config=None;self.metadata=None;self.accepted_evaluation=None
        if research_run:
            from wr_reproduction.research import read,Metadata
            self.research_config=read(research_run/'run.json')
            if self.research_config.get('study')=='matched-study':
                from wr_reproduction.matched_study import Metadata,assert_fence
                assert_fence(self.research_config['recipe'])
            self.metadata=Metadata(data=self.research_config['validation_data'])
        self.ks3=ks3
        self.ks3_precision=ks3_precision
        self.architecture=self.research_config['recipe'].get('architecture','ks3_experts' if ks3 else 'mlp') if research_run else ('ks3_experts' if ks3 else 'mlp')
        install(version)
        from wr_reproduction import fused_fp8
        fused_fp8.TILE_MODE=self.research_config['recipe'].get('shared_fp8_gemm_tuning','legacy') if self.research_config else 'legacy'
        fused_fp8.AMAX_MODE=self.research_config['recipe'].get('shared_fp8_amax','atomic') if self.research_config else 'atomic'
        fused_fp8.DECODE_MODE=self.research_config['recipe'].get('shared_fp8_decode','scalar') if self.research_config else 'scalar'
        fused_fp8.HEAD_MODE=self.research_config['recipe'].get('shared_fp8_head_tuning','legacy') if self.research_config else 'legacy'
        fused_fp8.SHARED_PIPELINE_MODE=self.research_config['recipe'].get('shared_fp8_pipeline_tuning','legacy') if self.research_config else 'legacy'
        from wr_reproduction import ks3_bf16_gemm
        ks3_bf16_gemm.TILE_MODE=self.research_config['recipe'].get('bf16_gemm_tuning','mi300-v1') if self.research_config else 'mi300-v1'
        from wr_reproduction import ks3_fp8_gemm
        ks3_fp8_gemm.TILE_MODE=self.research_config['recipe'].get('fp8_native_gemm_tuning','legacy') if self.research_config else 'legacy'
        if self.research_config:
            self.log(event='research_ks3_native_fp8_gemm',tile_mode=ks3_fp8_gemm.TILE_MODE,
                     scope='Fully prepared native KS3 products only; original formats/scales and BF16 evaluation preserved')
            self.log(event='research_shared_fp8_gemm',tile_mode=fused_fp8.TILE_MODE,
                     amax_mode=fused_fp8.AMAX_MODE,decode_mode=fused_fp8.DECODE_MODE,head_mode=fused_fp8.HEAD_MODE,shared_pipeline_mode=fused_fp8.SHARED_PIPELINE_MODE,
                     scope='Shared nongrouped training FP8 products; original formats, scales, dual layouts and epilogues')
            self.log(event='research_bf16_gemm',tile_mode=ks3_bf16_gemm.TILE_MODE,
                     kernel_mode=self.research_config['recipe'].get('bf16_kernel_mode','legacy'),
                     operand_precision='bf16',accumulation_precision='fp32',validation_uses_original_kernels=True)
        torch._dynamo.config.recompile_limit=64  # both original train_gpt.py launchers
        from track_1_short.model.gpt import GPT,FP8_EXACT_SCALE_CALLS
        if ks3:
            if version!='pr393':raise ValueError('KS3 block trial requires PR393')
            from wr_reproduction.ks3_moe import install as install_ks3
            GPT=install_ks3(ks3_precision, self.research_config['recipe'].get('geometry') if research_run else None,
                            self.research_config['recipe'].get('training_gs32',True) if research_run else True,
                            self.research_config['recipe'].get('charts',1024) if research_run else 1024,
                            self.research_config['recipe'].get('weight_packing','loop')=='batched' if research_run else False,
                            self.research_config['recipe'].get('geometry_backward_rows',64) if research_run else 64,
                            self.research_config['recipe'].get('bf16_kernel_mode','legacy') if research_run else 'legacy',
                            self.research_config['recipe'].get('projection_lr_multiplier',1.) if research_run else 1.,
                            self.research_config['recipe'].get('rank',1) if research_run else 1,
                            self.research_config['recipe'].get('fp8_prepared_native_weights',False) if research_run else False,
                            self.research_config['recipe'].get('fp8_prepared_native_activations',False) if research_run else False,
                            self.research_config['recipe'].get('fp8_prepared_native_dual',False) if research_run else False,
                            self.research_config['recipe'].get('fp8_prepared_native_gather',False) if research_run else False)
            if self.architecture!='ks3_experts':
                from wr_reproduction.ks3_feedforward import install as install_feedforward
                GPT=install_feedforward(GPT,self.research_config['recipe'])
        from track_1_short.config import Hyperparameters,TRAINING_STAGES,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT
        from track_1_short.schedule import TrainingSchedule
        from track_1_short.training import TrainingManager
        from track_1_short.sampled_softmax import SampledSoftmax
        from track_1_short.perf.cuda_graphs.capture_plan import step_graph_key,plan_warmup
        from track_1_short.perf.cuda_graphs.step_graphs import StepGraphs
        from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs
        from track_1_short.perf.cuda_graphs.fp8_refresh_graphs import Fp8RefreshGraphs
        from track_1_short.tail_average import TailAverages
        from track_1_short.ngram_table import is_update_step
        self.hp=hp=Hyperparameters();self.device=dev=torch.device('cuda')
        if hp.train_seed is not None:torch.manual_seed(hp.train_seed)
        self.schedule=schedule=TrainingSchedule(TRAINING_STAGES,hp.num_scheduled_iterations,hp.num_extension_iterations,
            dev,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
        if self.metadata:
            # Copy schedule tensors once before timing, and assert the separate
            # CPU metadata exactly describes the live objective and global batch.
            for s in range(schedule.total_steps):
                sig=self.metadata.training(s);stage,_=schedule.lookup(s)
                assert sig['mtp']==schedule.mtp_weights[s].tolist()
                assert sig['prefix']==schedule.prefix_weights[s].item()
                assert sig['global_batch_tokens']==stage.batch_size
        contract=json.loads((Path(__file__).parent/version/'contract.json').read_text())
        if schedule.total_steps!=contract['total_steps'] or hp.val_tokens!=contract['val_tokens']:
            raise ValueError('Record schedule/evaluation overrides are forbidden')
        self.max_tokens=max(s.batch_size for s in TRAINING_STAGES)//8
        kwargs=dict(ngram_dim=768,world_size=8,device=dev)
        if version=='pr393':kwargs['rank']=0
        self.model=m=GPT(50257,11,6,128,768,hp.val_batch_size//8,**kwargs).cuda()
        m.cast_matrix_weights_bf16()
        if self.research_config and self.ks3:
            charts=self.research_config['recipe']['charts'];rank=self.research_config['recipe']['rank']
            features=charts*rank
            assert tuple(m.moe_bank.shape)==(80,features,768)
            assert tuple(m.ks3_geometry.shape)==(5,8,charts,6*rank+2)
            assert tuple(m.moe_router.shape)==(5,8,768)
            self.log(event='research_architecture',charts=charts,rank=rank,features=features,experts=8,topk=1,
                     expert_projection_parameters=m.moe_bank.numel(),geometry_parameters=m.ks3_geometry.numel(),
                     total_parameters=sum(p.numel() for p in m.parameters()),
                     original_non_expert_initialization_preserved=True)
        if self.research_config and self.research_config.get('study')=='matched-study':
            from wr_reproduction.matched_architecture import audit_model
            self.log(event='feedforward_inventory',**audit_model(m,self.architecture))
        if version=='pr393':
            # All eight optimizer shards reside on the one physical owner. The
            # model/router and logical-rank data still retain world_size=8.
            m._moe_lo,m._moe_hi=0,m._moe_scales.numel()
        m.quantize_attn_fp8();m.quantize_mlp_fp8(refresh_lm=True)
        if version=='pr393':m.quantize_moe_full()
        self.refresh=Fp8RefreshGraphs(m)
        self.optimizer_graphs=AnvilBankGraphs(dev)
        self.tm=TrainingManager(m,schedule,bank_update=self.optimizer_graphs.update,
            **({'moe_publish':self.refresh.publish_moe} if version=='pr393' else {}))
        if self.research_config and self.ks3:
            multiplier=self.research_config['recipe'].get('projection_lr_multiplier',1.)
            cfg=self.tm.optimizer.param_cfgs[m.moe_bank]
            shape_scale=max(1.,features/768)**.5
            assert cfg.lr_mul==shape_scale*multiplier
            self.log(event='research_optimizer',projection_lr_multiplier=multiplier,
                     projection_effective_lr_multiplier=cfg.lr_mul,shape_scale=shape_scale,
                     projection_base_lr=cfg.lr,output_projection_multiplier=2.,
                     projection_weight_decay=cfg.weight_decay,weight_decay_multiplier=cfg.wd_mul)
        self.tail=TailAverages(self.tm.optimizer,0,value_embed_updates=is_update_step,total_steps=schedule.total_steps)
        self.samplers=[]
        for rank in range(8):
            sampler=SampledSoftmax(schedule,m.vocab_size,768,self.max_tokens,rank,8,dev)
            if self.samplers:
                sampler.loss_inputs=self.samplers[0].loss_inputs;sampler.candidates=self.samplers[0].candidates
                sampler.staging=self.samplers[0].staging  # serialize shared-buffer uploads behind all previous readers
            self.samplers.append(sampler)
        if version=='pr393':
            from track_1_short.perf.kernels import sampled_head
            sampled_head.init(dev,rank=0);self.sampled_head=sampled_head
        self.counter_states=[torch.tensor(r,device=dev,dtype=torch.int64) for r in range(8)]
        self.shared_value_grad=torch.zeros_like(m.value_embeds,dtype=torch.float16)
        compiled=torch.compile(m,dynamic=False,fullgraph=True)
        keys=[step_graph_key(schedule,self.samplers[0].counts,s,8) for s in range(schedule.total_steps)]
        self.graphs=StepGraphs(compiled,self.tm.optimizer,keys,Sinks(dev),SimpleNamespace(grad_accum=self.shared_value_grad),self.max_tokens,dev)
        transitions=sorted(set(self.tm.get_transition_steps())|set(self.samplers[0].count_change_steps()))
        split=schedule.split_step
        base=sorted({0,1}|{s+o for s in transitions for o in [-2,-1,0,1] if s+o>=2}|
                    {split+o for o in [1,2,3] if split+o<schedule.total_steps})
        self.plan=plan_warmup(base,keys,self.tm.is_adam_step,FP8_EXACT_SCALE_CALLS)
        self.tables=VirtualTables(m,self.tm.optimizer,schedule.total_steps,version)
        from track_1_short.perf.cuda_graphs.optimizer_graphs import assert_banks_alias_live_state
        assert_banks_alias_live_state(self.tm.optimizer)
        self.rank_states=RankStates(m)
        self.gradients=OriginalGradientAccumulator(p for p,c in self.tm.optimizer.param_cfgs.items() if c.optim=='adam')
        self.has_previous=False
        self.previous_adam=False

    def loaders(self,steps):
        from track_1_short.data import distributed_data_generator,cu_seqlens_rows,ScheduledBatches
        from track_1_short.perf.pinned_batches import PinnedBatchStaging
        stage,_=self.schedule.lookup(0);result=[]
        for rank in range(8):
            staging=PinnedBatchStaging(max(self.max_tokens,self.hp.val_batch_size//8),
                cu_seqlens_rows(max(self.max_tokens,self.hp.val_batch_size//8)),self.device)
            gen=distributed_data_generator(self.hp.train_files,stage.batch_size,stage.train_max_seq_len,staging)
            batches=ScheduledBatches(gen,self.schedule,steps)
            with patch('torch.distributed.get_rank',return_value=rank),patch('torch.distributed.get_world_size',return_value=8):
                batches.peek(steps[0])
            result.append(batches)
        return result

    def step(self,step,batches,next_step=None,prefetch=False):
        from track_1_short.perf.kernels.transpose import transpose_add
        m=self.model;tm=self.tm
        tm.advance_schedule(step);self.gradients.begin(tm.is_adam_step(step));losses=[]
        for rank in range(8):
            if self.observations:self.observations.check()
            self.rank_states.enter(rank)
            if self.has_previous:
                if rank==0:
                    self.refresh.refresh_attn()
                    if self.version=='pr393':self.refresh.refresh_moe_cols()
                    self.refresh.refresh_mlp(self.previous_adam)
                else:self.rank_states.refresh_activations()
            if self.version=='pr393':self.sampled_head.COUNTER.copy_(self.counter_states[rank])
            self.shared_value_grad.copy_(self.tables.value_grads[rank])
            batch=batches[rank].peek(step);slots=self.tables.load(batch.ngram_ids)
            sampler=self.samplers[rank];sampler.upload(step,batch.targets_cpu)
            sampled=sampler.gather(step,m._lm_head_f8_col)
            m.zero_grad(set_to_none=True)
            self.graphs.forward(step,batch,slots,tm.get_forward_args(sampled))
            ng=self.graphs.backward();self.tables.hold(rank,batch.ngram_ids,ng.clone())
            self.tables.value_grads[rank].copy_(self.shared_value_grad)
            self.gradients.add(rank,{p:p.grad for p in tm.optimizer.param_cfgs if p.grad is not None},
                tied_pair=None if tm.optimizer.split_embed else (m.lm_head.weight,m.embed.weight),transpose_add=transpose_add)
            losses.append(self.graphs.current.loss.detach().clone()/batch.inputs.numel())
            self.rank_states.leave(rank)
            if self.version=='pr393':self.counter_states[rank].copy_(self.sampled_head.COUNTER)
            sampler.mark_readers_done();batches[rank].take(step)
            if prefetch:
                if rank<7:self.samplers[rank+1].prefetch(step,batches[rank+1].peek(step).targets_cpu)
                elif next_step is not None:self.samplers[0].prefetch(next_step,batches[0].peek(next_step).targets_cpu)
        m.zero_grad(set_to_none=True)
        for p,g in self.gradients.finish().items():p.grad=g
        tm.step_optimizers(step,self.tables,frozenset())
        self.has_previous=True;self.previous_adam=tm.is_adam_step(step)
        return torch.stack(losses).mean()

    def warmup(self):
        m=self.model
        if self.research_run and len(self.plan.steps)!=24:raise ValueError('Research requires the full original 24-update warmup')
        initial_model=copy.deepcopy(m.state_dict());initial_optimizer=self.tm.get_state()
        self.tables.event_policy=self.tm.is_adam_step  # original warmup's two-step sparse cadence
        warm=self.loaders(self.plan.steps)
        self.log(event='warmup_begin',steps=self.plan.steps)
        for pos,step in enumerate(self.plan.steps):
            # Tail capture happens after flushing the previous step's refresh.
            # step() must not refresh it twice.
            if pos in (self.plan.optimizer_capture_at,self.plan.fp8_capture_at):
                self.rank_states.enter(0)
                self.refresh.refresh_attn()
                if self.version=='pr393':self.refresh.refresh_moe_cols()
                self.refresh.refresh_mlp(self.previous_adam)
                self.rank_states.leave(0)
                # The other logical ranks refresh their activation states now.
                for rank in range(1,8):
                    self.rank_states.enter(rank);self.rank_states.refresh_activations();self.rank_states.leave(rank)
                self.rank_states.enter(0);self.has_previous=False
            if pos==self.plan.optimizer_capture_at:
                self.optimizer_graphs.capture([self.tm.optimizer.banks[k] for k in self.tm.work_order if k in self.tm.optimizer.banks])
            if pos==self.plan.fp8_capture_at:self.refresh.capture()
            loss=self.step(step,warm);torch.cuda.synchronize()
            self.log(event='warmup_step',step=step,objective=loss.item())
        self.graphs.seal();self.optimizer_graphs.seal();self.refresh.seal()
        m.zero_grad(set_to_none=True);m.load_state_dict(initial_model);self.tm.reset(initial_optimizer)
        del initial_model,initial_optimizer,warm
        self.tables.reset();self.tables.event_policy=self.tables.source.is_update_step
        self.shared_value_grad.zero_()
        for rank,sampler in enumerate(self.samplers):
            sampler.reset();self.counter_states[rank].fill_(rank)
        m.rearm_fp8_bootstrap();m.quantize_attn_fp8();m.quantize_mlp_fp8(refresh_lm=True)
        if self.version=='pr393':m.quantize_moe_full()
        from track_1_short.perf.cuda_graphs.optimizer_graphs import assert_banks_alias_live_state
        assert_banks_alias_live_state(self.tm.optimizer)
        self.rank_states=RankStates(m)
        self.gradients=OriginalGradientAccumulator(p for p,c in self.tm.optimizer.param_cfgs.items() if c.optim=='adam')
        self.has_previous=False;self.previous_adam=False
        if self.research_run and self.ks3 and self.ks3_precision=='fp8':
            # Compile the accepted BF16/GS32 evaluation weight packer before the
            # clock, just like the training graph kernels. evaluate() still
            # repeats this packing on the live weights inside timed setup.
            m.prepare_ks3_eval()
            self.log(event='evaluation_packing_warmup',weights='restored-initial',
                     evaluation_packing_remains_in_timed_setup=True)
        if hasattr(m,'feedforward_execution_report'):
            self.log(event='feedforward_execution_coverage',**m.feedforward_execution_report())
        torch.cuda.synchronize();self.log(event='preflight_passed',step_graphs=len(self.graphs.captured),
            optimizer_graphs=len(self.optimizer_graphs.graphs),fp8_refresh_graphs=len(self.refresh.captured))

    def evaluate(self,final,clock_start,elapsed):
        from track_1_short.data import distributed_data_generator,cu_seqlens_rows
        from track_1_short.perf.pinned_batches import PinnedBatchStaging
        m=self.model
        if final:
            self.tm.optimizer.land_bf16_replicas() if self.version=='pr393' else None
            self.tm.apply_final_ws_ext()
            self.canonical.wait();self.canonical.collect(m.canon_mask)
            self.tail.ship()
            if self.ks3:m.quantize_moe_full()
        elif self.has_previous:
            self.rank_states.enter(0);self.refresh.refresh_attn()
            if self.version=='pr393':self.refresh.refresh_moe_cols()
            self.refresh.refresh_mlp(self.previous_adam);self.rank_states.leave(0)
            for rank in range(1,8):
                self.rank_states.enter(rank);self.rank_states.refresh_activations();self.rank_states.leave(rank)
            self.has_previous=False
        if self.ks3 and self.ks3_precision=='fp8':m.prepare_ks3_eval()
        evaluation_args=self.tm.get_forward_args()
        evaluation_protocol=None
        if self.accepted_evaluation is not None:
            if final:raise RuntimeError('Accepted raw research evaluation cannot be canonical')
            evaluation_args,evaluation_protocol=self.accepted_evaluation.prepare(self.tm)
        m.complete_yarn_tables();batches=[]
        val_steps=self.hp.val_tokens//self.hp.val_batch_size
        for rank in range(8):
            staging=PinnedBatchStaging(self.hp.val_batch_size//8,cu_seqlens_rows(self.hp.val_batch_size//8),self.device)
            loader=distributed_data_generator(self.hp.val_files,self.hp.val_batch_size,-1,staging,align_to_bos=False)
            with patch('torch.distributed.get_rank',return_value=rank),patch('torch.distributed.get_world_size',return_value=8):first=next(loader)
            rows=[first]+[next(loader) for _ in range(val_steps-1)]
            for batch in rows:self.tables.load(batch.ngram_ids)
            batches.append(rows)
        torch.cuda.synchronize();elapsed+=time.perf_counter()-clock_start
        m.eval();values=[]
        if self.version=='pr393':m.cplm_diag.zero_()
        with torch.no_grad():
            for rows in batches:
                rank_loss=0
                for batch in rows:
                    slots=self.tables.load(batch.ngram_ids)
                    rank_loss=rank_loss+m(batch.inputs,batch.targets,batch.cum_seqlens,slots,evaluation_args).mean()
                values.append(rank_loss/val_steps)
        loss=torch.stack(values).mean().item();m.train();torch.cuda.synchronize();gc.collect()
        extra={}
        if self.research_run:
            budget=self.research_config['recipe']['max_train_seconds']
            signature=self.metadata.validation(self.eval_step)
            signature['attention_windows']=[evaluation_args.ws_short,evaluation_args.ws_long]
            extra=dict(evaluation_signature=signature,step=self.eval_step,
                       budget_seconds=budget,budget_overrun_seconds=max(0.,elapsed-budget),
                       within_control_budget=elapsed<=budget)
            if evaluation_protocol is not None:extra['evaluation_protocol']=evaluation_protocol
        self.log(event='validation',final=final,loss=loss,train_seconds=elapsed,val_tokens=self.hp.val_tokens,**extra)
        if self.version=='pr393':
            diag=m.cplm_diag.detach().float().cpu();n_tokens=diag[0].clamp_min(1)
            self.log(event='cplm_diagnostics',nll=(diag[1]/n_tokens).item(),lm_nll=(diag[2]/n_tokens).item(),
                alpha=(diag[3]/n_tokens).item(),a_sink=(diag[4]/n_tokens).item(),copy_share=(diag[5]/n_tokens).item())
        return elapsed,time.perf_counter()

    def train(self,max_train_seconds=None,timing_updates=None):
        if self.research_run and timing_updates is None:
            return self.train_research(max_train_seconds)
        from track_1_short.model.prefix_prediction import build_prefix_table_bucket
        m=self.model;batches=self.loaders(list(range(self.schedule.total_steps)))
        m.limit_yarn_rebuild(self.max_tokens)
        from track_1_short.canonical_mask import BackgroundCanonicalMask
        self.canonical=BackgroundCanonicalMask(m.vocab_size,owner=True,print0=lambda msg,**kw:print(msg,flush=True))
        gc.collect();gc.freeze();gc.disable()
        self.log(event='train_begin',max_train_seconds=max_train_seconds,total_steps=self.schedule.total_steps,timing_updates=timing_updates)
        torch.cuda.synchronize();clock_start=time.perf_counter();elapsed=0.
        self.canonical.start()
        m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size,0,1))
        for sampler in self.samplers:sampler.set_prefix_table(m.prefix_table.cpu().numpy())
        for step in range(self.schedule.total_steps+1):
            if timing_updates is not None and step==timing_updates:
                torch.cuda.synchronize()
                self.log(event='timing_complete',steps=step,train_seconds=elapsed+time.perf_counter()-clock_start,
                    full_schedule_complete=False,validation_performed=False)
                self.canonical.wait();gc.enable()
                return
            self.tm.advance_schedule(step)
            final=step==self.schedule.total_steps
            if not final and max_train_seconds is not None and elapsed+time.perf_counter()-clock_start>=max_train_seconds:
                torch.cuda.synchronize()
                self.log(event='time_limit',step=step,train_seconds=elapsed+time.perf_counter()-clock_start)
                elapsed,clock_start=self.evaluate(False,clock_start,elapsed)
                self.canonical.wait()
                gc.enable()
                self.log(event='stopped_at_time_limit',steps=step,train_seconds=elapsed,full_schedule_complete=False)
                return
            if final or (self.hp.val_loss_every>0 and step%self.hp.val_loss_every==0):
                elapsed,clock_start=self.evaluate(final,clock_start,elapsed)
                if final:break
            loss=self.step(step,batches,next_step=step+1 if step+1<self.schedule.total_steps else None,prefetch=True);self.tail.tick(step)
            if self.ks3 and step==24:
                geometry=m.ks3_geometry.detach()
                if not torch.isfinite(geometry).all():raise RuntimeError('Non-finite KS3 geometry')
                rank=self.research_config['recipe']['rank'] if self.research_config else 1
                initial_scale=self.research_config['recipe']['geometry']['raw_scale'] if self.research_config else .75
                self.log(event='ks3_geometry_activity',rank=rank,coefficient_max_abs=geometry[...,:6*rank].abs().max().item(),
                    raw_scale_max_change=(geometry[...,6*rank]-initial_scale).abs().max().item(),raw_shift_max_abs=geometry[...,6*rank+1].abs().max().item())
            if (step+1)%25==0 or step+1>=self.schedule.total_steps-1:
                self.log(event='train',step=step+1,objective=loss.item(),train_seconds=elapsed+time.perf_counter()-clock_start)
        gc.enable()
        self.log(event='complete',steps=self.schedule.total_steps)

    def train_research(self,max_train_seconds):
        """Opt-in study path; default reproduction/control execution is unchanged."""
        from wr_reproduction.research_observer import Observations,ResearchStopped
        from track_1_short.model.prefix_prediction import build_prefix_table_bucket
        from track_1_short.canonical_mask import BackgroundCanonicalMask
        if max_train_seconds!=self.research_config['recipe']['max_train_seconds']:
            raise ValueError('Research uses the exact accepted timed-training budget')
        m=self.model;batches=self.loaders(list(range(self.schedule.total_steps)))
        m.limit_yarn_rebuild(self.max_tokens)
        if self.research_config['recipe'].get('validation_window_policy','training-stage')=='accepted-control':
            from wr_reproduction.research_evaluation import AcceptedEvaluation
            self.accepted_evaluation=AcceptedEvaluation()
        self.canonical=BackgroundCanonicalMask(m.vocab_size,owner=True,print0=lambda msg,**kw:print(msg,flush=True))
        gc.collect();gc.freeze();gc.disable()
        self.log(event='train_begin',max_train_seconds=max_train_seconds,total_steps=self.schedule.total_steps,
                 seed=self.hp.train_seed,continuous_from_initialization=True,
                 evaluation='raw-no-canonical-no-tail-no-final-extension-bf16',
                 cap_setup_reserve_seconds=self.research_config['recipe']['validation_setup_reserve_seconds'])
        torch.cuda.synchronize();clock_start=time.perf_counter();elapsed=0.
        self.observations=Observations(self.research_run,self.log,lambda timestamp:elapsed+timestamp-clock_start,
                                       state=self.research_run.parent.parent)
        self.canonical.start()
        m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size,0,1))
        for sampler in self.samplers:sampler.set_prefix_table(m.prefix_table.cpu().numpy())
        reserve=self.research_config['recipe']['validation_setup_reserve_seconds']
        terminal_condition=None
        try:
            for step in range(self.schedule.total_steps+1):
                self.observations.check()
                self.tm.advance_schedule(step)
                if self.accepted_evaluation is not None:self.accepted_evaluation.observe(self.tm,step)
                wall=elapsed+time.perf_counter()-clock_start
                final=step==self.schedule.total_steps
                stage=self.metadata.stage(step)
                predicted=self.observations.predicted_update_seconds(stage)
                if final or wall+predicted+reserve>=max_train_seconds:
                    self.observations.drain();torch.cuda.synchronize()
                    wall=elapsed+time.perf_counter()-clock_start
                    self.log(event='time_limit' if not final else 'raw_schedule_end',step=step,
                             train_seconds=wall,predicted_next_update_seconds=predicted,
                             validation_setup_reserve_seconds=reserve)
                    self.eval_step=step
                    # Even if the original schedule completes early, never ship
                    # tail weights, collect the mask, or extend final windows.
                    elapsed,clock_start=self.evaluate(False,clock_start,elapsed)
                    self.log(event='stopped_at_time_limit' if not final else 'raw_schedule_complete',steps=step,
                             train_seconds=elapsed,full_schedule_complete=final,
                             budget_overrun_seconds=max(0.,elapsed-max_train_seconds))
                    terminal_condition='raw_schedule_complete' if final else 'time_limit'
                    break
                loss=self.step(step,batches,next_step=step+1 if step+1<self.schedule.total_steps else None,prefetch=True)
                self.tail.tick(step)
                signature=self.metadata.training(step)
                if signature['candidates']!=self.samplers[0].counts[step]:
                    raise ValueError('Live sampled candidate count disagrees with research metadata')
                self.observations.submit(loss,step+1,signature)
        except ResearchStopped as error:
            self.log(event='research_pruned',reason=str(error),train_seconds=elapsed+time.perf_counter()-clock_start)
            terminal_condition=str(error)
        finally:
            self.observations.close();self.observations=None
            gc.enable()
        if self.ks3 and terminal_condition and not (self.research_run.parent.parent/'STOP').exists():
            from wr_reproduction.geometry_diagnostics import collect
            self.log(terminal_condition=terminal_condition,
                     **collect(self.model,self.research_config['recipe'],self.research_run))

    def check_updates(self,count):
        """Exercise the real prefetch/update path without shortening the schedule."""
        from track_1_short.model.prefix_prediction import build_prefix_table_bucket
        self.model.prefix_table.copy_(build_prefix_table_bucket(self.model.vocab_size,0,1))
        prefix=self.model.prefix_table.cpu().numpy()
        for sampler in self.samplers:sampler.set_prefix_table(prefix)
        batches=self.loaders(list(range(count)))
        for step in range(count):
            torch.cuda.synchronize();started=time.perf_counter()
            loss=self.step(step,batches,next_step=step+1 if step+1<count else None,prefetch=True)
            self.tail.tick(step)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            if not torch.isfinite(loss):raise RuntimeError('Non-finite update objective')
            stage,_=self.schedule.lookup(step)
            self.log(event='diagnostic_update',step=step+1,objective=loss.item(),seconds=seconds,
                global_tokens=stage.batch_size,logical_ranks=8)
        self.log(event='diagnostic_complete',updates=count,full_convergence_verified=False)

    def log(self,**record):
        from datetime import datetime,timezone
        charts=self.research_config['recipe'].get('charts',1024) if self.research_config else 1024
        rank=self.research_config['recipe'].get('rank',1) if self.research_config else 1
        record.update(utc=datetime.now(timezone.utc).isoformat(),version=self.version,source_commit=self.manifest['commit'],
            expert_block=f'ks3-gs32-{self.ks3_precision}-c{charts}-r{rank}' if self.ks3 else 'mlp-fp8')
        if self.research_config:
            record['architecture']=self.architecture
            record.update(run_id=self.research_config['run_id'],seed=self.research_config['seed'],code_commit=self.research_config['code_commit'])
            record['training_projection_gs32']=self.ks3 and self.research_config['recipe'].get('training_gs32',True)
            if self.ks3 and not record['training_projection_gs32']:
                record['expert_block']=f'ks3-native-fp8-projections-gs32-coeff-c{charts}-r{rank}'
        line=json.dumps(record)
        with self.log_lock:
            print(line,flush=True)
            with self.out.open('a') as f:f.write(line+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['pr393','record360'],default='pr393')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--check-updates',type=int,default=0,choices=range(0,9),
        help='After warmup, test this many original updates and exit; schedule is unchanged')
    parser.add_argument('--max-train-seconds',type=float,
        help='Stop after this timed-training budget and evaluate the current iterate; do not compress the original schedule')
    parser.add_argument('--ks3',action='store_true',help='Replace only PR393 MoE experts with GS32/BF16 rank-one KS3')
    parser.add_argument('--architecture',choices=['mlp','ks3_all','ks3_first_mlp','ks3_last_mlp','ks3_both_mlp'])
    parser.add_argument('--ks3-precision',choices=['bf16','fp8'],default='bf16')
    parser.add_argument('--timing-updates',type=int,help='Stop after this many actual training updates for timing; skip validation and preserve the original schedule')
    parser.add_argument('--profile-updates',type=int,default=0,choices=range(0,9),help='Isolated GPU trace of1-8 original prefix updates; never convergence or speed evidence')
    parser.add_argument('--research-run',type=Path,help='Isolated WR study run directory; opt-in screening, raw evaluation and seeding')
    args=parser.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.max_train_seconds is not None and args.max_train_seconds<=0:parser.error('--max-train-seconds must be positive')
    if args.timing_updates is not None and not 1<=args.timing_updates<=1053:parser.error('--timing-updates must be between 1 and 1053')
    if args.out.exists():raise FileExistsError('Use a new output file; preserve prior run evidence')
    if args.profile_updates and not args.research_run:parser.error('Profiling requires an independent study registry entry')
    if args.research_run:
        from wr_reproduction.research import read
        cfg=read(args.research_run/'run.json')
        if args.version!='pr393' or args.preflight or args.check_updates:
            parser.error('Research requires continuous PR393 training from initialization')
        if args.architecture:
            if args.architecture!=cfg['recipe'].get('architecture'):parser.error('Architecture does not match registered recipe')
            args.ks3=args.architecture!='mlp'
        elif not args.ks3:parser.error('Explicit --architecture=mlp is required for a control')
        stop_files=cfg.get('stop_files',[str(args.research_run.parent.parent/'STOP')])
        if any(Path(p).exists() for p in stop_files):raise RuntimeError('Explicit study STOP')
        if bool(args.profile_updates)!=(cfg['kind']=='profile'):parser.error('Profiling requires a distinct noncontinuous registry kind')
        if bool(args.timing_updates)!=(cfg['kind']=='timing'):
            parser.error('Timing diagnostics and continuous research trials must have distinct registry kinds')
        os.environ['TRAIN_SEED']=str(cfg['seed'])  # before immutable config import/model initialization
        if args.max_train_seconds!=cfg['recipe']['max_train_seconds']:parser.error('Registered time budget required')
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","1")))
    with LocalCollectives():
        runner=Runner(args.version,args.out,args.ks3,args.ks3_precision,args.research_run);runner.warmup()
        if args.profile_updates:
            from wr_reproduction.profile_updates import run as profile_updates
            profile_updates(runner,args.profile_updates)
        elif args.check_updates:runner.check_updates(args.check_updates)
        elif not args.preflight:runner.train(args.max_train_seconds,args.timing_updates)

if __name__=='__main__':main()
