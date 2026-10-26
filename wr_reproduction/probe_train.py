"""Probe-study entry point. The historical Runner and its shared kernels are untouched."""
import argparse
import gc
import math
import os
import time
from pathlib import Path
import torch
from wr_reproduction.train import Runner
from wr_reproduction.local_collectives import LocalCollectives
from wr_reproduction.research_observer import Observations as TrainingObservations,ResearchStopped
from wr_reproduction.probe_study import (Screen,Metadata,read,assert_fence,stopped,DEFINITION)
from wr_reproduction.probe_evaluation import FixedProbe

class DiagnosticsOnly:
    def __init__(self,config):
        self.config=config;self.stop_files=[Path(p) for p in config["stop_files"]]
    def observe(self,row):
        value=row['objective']
        return dict(abort=not math.isfinite(value),reason='nonfinite_training_objective' if not math.isfinite(value) else 'diagnostic_only_no_comparative_training_pruning')
    def stopped(self):return any(Path(p).exists() for p in self.config['stop_files'])

class Observations(TrainingObservations):
    def __init__(self,run,log,clock,state):
        # Do not construct the historical Screen/reference store even temporarily.
        import queue,threading
        self.screen=DiagnosticsOnly(read(run/'run.json'));self.log,self.clock=log,clock
        self.pending=queue.Queue(maxsize=4);self.failure=None;self.abort=None;self.durations={};self.last_wall=0.
        self.thread=threading.Thread(target=self._work,daemon=True);self.thread.start()
    def reset_timing(self):
        self.pending.join();self.last_wall=0.

def train(r):
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from track_1_short.canonical_mask import BackgroundCanonicalMask
    from wr_reproduction.research_evaluation import AcceptedEvaluation
    cfg=r.research_config;protocol=cfg['probe_protocol'];budget=cfg['recipe']['max_train_seconds']
    r.metadata=Metadata(cfg['validation_data']);r.accepted_evaluation=AcceptedEvaluation()
    m=r.model;batches=r.loaders(list(range(r.schedule.total_steps)));m.limit_yarn_rebuild(r.max_tokens)
    r.canonical=BackgroundCanonicalMask(m.vocab_size,owner=True,print0=lambda msg,**kw:print(msg,flush=True))
    gc.collect();gc.freeze();gc.disable();torch.cuda.synchronize();start=time.perf_counter()
    setup_start=time.perf_counter()
    probe=FixedProbe(r,protocol,cfg['backend_hash']);screen=Screen(r.research_run,r.research_run.parent.parent)
    probe_setup_seconds=time.perf_counter()-setup_start
    r.observations=Observations(r.research_run,r.log,lambda stamp:stamp-start,r.research_run.parent.parent)
    r.log(event='train_begin',max_train_seconds=budget,total_steps=r.schedule.total_steps,continuous_from_initialization=True,
          clock='continuous-all-probe-setup-scoring-restoration-included',probe_protocol=protocol,
          comparative_pruning_exempt=cfg['comparative_pruning_exempt'],probe_initialization_seconds=probe_setup_seconds)
    r.canonical.start();m.prefix_table.copy_(build_prefix_table_bucket(m.vocab_size,0,1))
    for sampler in r.samplers:sampler.set_prefix_table(m.prefix_table.cpu().numpy())
    next_probe=protocol['interval_seconds'];reserve=cfg['recipe']['validation_setup_reserve_seconds'];terminal=None;extra_overhead=probe_setup_seconds
    try:
        for step in range(r.schedule.total_steps+1):
            r.observations.check();r.tm.advance_schedule(step);r.accepted_evaluation.observe(r.tm,step)
            wall=time.perf_counter()-start;final=step==r.schedule.total_steps
            predicted=r.observations.predicted_update_seconds(r.metadata.stage(step))
            if final or wall+predicted+reserve>=budget:
                r.observations.drain();torch.cuda.synchronize();r.eval_step=step
                r.log(event='raw_schedule_end' if final else 'time_limit',step=step,train_seconds=time.perf_counter()-start,
                      cumulative_probe_seconds=probe.cumulative,probe_overhead_fraction=(probe.cumulative+extra_overhead)/(time.perf_counter()-start),cumulative_instrumentation_seconds=probe.cumulative+extra_overhead)
                elapsed,_=r.evaluate(False,start,0.)
                terminal='raw_schedule_complete' if final else 'time_limit'
                r.log(event=terminal,steps=step,train_seconds=elapsed,cumulative_probe_seconds=probe.cumulative,
                      probe_overhead_fraction=(probe.cumulative+extra_overhead)/elapsed,cumulative_instrumentation_seconds=probe.cumulative+extra_overhead,full_schedule_complete=final)
                # Confirmation is terminal and separate from final full validation and training time.
                from wr_reproduction.probe_study import events,RESULTS
                endpoint=[x for x in events(r.out) if x.get('event')=='validation'][-1]
                control=read(RESULTS/'control_set.json')
                finalist=cfg['kind'] in ('control','reference','verification') or bool(control and elapsed<=budget and endpoint['loss']<control['best']['loss'])
                if finalist:
                    confirmation=FixedProbe(r,protocol,cfg['backend_hash'],reserved=True)
                    record=confirmation.observe(step,time.perf_counter())
                    record.update(terminal_training_seconds=elapsed,tuning_use_forbidden=True,timing='separate-terminal-confirmation')
                    r.log(**record)
                break
            update_diagnostics=None
            if wall>=next_probe:
                instrument_start=time.perf_counter()
                r.observations.drain();row=probe.observe(step,start);r.log(**row)
                outcome=screen.observe(row);r.log(event='probe_screen',**outcome)
                r.observations.reset_timing()
                next_probe=(math.floor(row['probe_completion_seconds']/protocol['interval_seconds'])+1)*protocol['interval_seconds']
                if outcome['abort']:
                    r.log(event='screening_abort',**outcome);raise ResearchStopped(outcome['reason'])
                if r.ks3:
                    from wr_reproduction.probe_diagnostics import before_update
                    update_diagnostics=before_update(m)
                extra_overhead+=time.perf_counter()-instrument_start-row['probe_seconds']
            loss=r.step(step,batches,next_step=step+1 if step+1<r.schedule.total_steps else None,prefetch=True)
            r.tail.tick(step);r.observations.submit(loss,step+1,r.metadata.training(step))
            if update_diagnostics is not None:
                from wr_reproduction.probe_diagnostics import after_update
                diagnostic_start=time.perf_counter()
                r.log(event='ks3_update_diagnostics',step=step+1,measurements=after_update(m,update_diagnostics),train_seconds=time.perf_counter()-start)
                extra_overhead+=time.perf_counter()-diagnostic_start
    except ResearchStopped as error:
        terminal=str(error);r.log(event='research_pruned',reason=terminal,train_seconds=time.perf_counter()-start)
    finally:
        r.observations.close();r.observations=None;gc.enable()
    if r.ks3 and terminal and not stopped():
        from wr_reproduction.geometry_diagnostics import collect
        r.log(terminal_condition=terminal,**collect(m,cfg['recipe'],r.research_run))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--calibration',choices=['baseline','probe','precision']);a=p.parse_args()
    cfg=read(a.run/'run.json')
    if any(Path(f).exists() for f in cfg['stop_files']):raise RuntimeError('Explicit study STOP')
    if not a.calibration:assert_fence(cfg['recipe'])
    for key in ('NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE'):
        if key in os.environ:raise RuntimeError('Forbidden inherited override: '+key)
    os.environ['TRAIN_SEED']=str(cfg['seed']);torch.set_num_threads(1)
    with LocalCollectives():
        r=Runner('pr393',a.run/'events.jsonl',cfg['recipe']['architecture']!='mlp',cfg['recipe'].get('precision','bf16'),a.run)
        from wr_reproduction.matched_architecture import audit_model
        r.log(event='feedforward_inventory',**audit_model(r.model,cfg['recipe']['architecture']))
        if a.calibration=='probe':
            r.tm.advance_schedule(0)
            early=FixedProbe(r,cfg['probe_protocol'],cfg['backend_hash'])
            initial=early.observe(0,time.perf_counter())
            initial.update(event='probe_initialization_cost',cold_compilation_included=True,not_a_training_reference=True)
            r.log(**initial)
            repeat=early.observe(0,time.perf_counter())
            repeat.update(event='probe_initialization_repeat_cost',cold_compilation_included=False,not_a_sampling_uncertainty_estimate=True)
            r.log(**repeat)
            del early
        r.warmup()
        if a.calibration=='precision':
            from wr_reproduction.probe_precision_calibration import run
            run(r)
        elif a.calibration:
            from wr_reproduction.probe_calibration import run
            run(r,a.calibration)
        else:train(r)
if __name__=='__main__':main()
