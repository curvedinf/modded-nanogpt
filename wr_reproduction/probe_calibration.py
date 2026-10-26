"""Bounded real-model probe cost, evaluation oracle and continuation test."""
import hashlib
import time
from pathlib import Path
import torch
from wr_reproduction.probe_evaluation import FixedProbe
from wr_reproduction.probe_study import atomic,paired

def fingerprint(t):
    # Stream host hashes; never materialize the130GB sparse table on the host.
    x=t.detach().contiguous().reshape(-1).view(torch.uint8)
    h=hashlib.sha256()
    for offset in range(0,x.numel(),16*1024*1024):
        h.update(x[offset:offset+16*1024*1024].cpu().numpy())
    return h.hexdigest()

def state(r):
    import dataclasses
    import random
    import numpy as np
    result={}
    def walk(value,path):
        if isinstance(value,torch.Tensor):
            result[path]=dict(sha=fingerprint(value),shape=list(value.shape),dtype=str(value.dtype),pointer=value.data_ptr())
        elif isinstance(value,np.ndarray):
            result[path]=dict(sha=hashlib.sha256(value.tobytes()).hexdigest(),shape=list(value.shape))
        elif value is None or isinstance(value,(bool,int,float,str)):
            result[path]=value
        elif isinstance(value,dict):
            for k,v in value.items():walk(v,path+'/'+str(k))
        elif isinstance(value,(list,tuple)):
            for i,v in enumerate(value):walk(v,path+'/'+str(i))
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):walk(getattr(value,field.name),path+'/'+field.name)
    # Wait for an already queued candidate build without taking/advancing it.
    for sampler in r.samplers:
        if sampler.staging.pending:sampler.staging.pending[1].result()
    torch.cuda.synchronize()
    for n,module in r.model.named_modules():walk(module.training,'mode/'+n)
    for key,value in vars(r.model).items():
        if isinstance(value,(int,float,bool)) and ('quant' in key or 'scale' in key or 'call' in key):walk(value,'model_host/'+key)
    for key in ('ws_short','ws_long','train_max_seq_len'):walk(getattr(r.tm,key),'scheduled_context/'+key)
    for n,p in r.model.named_parameters():walk(p,'parameter/'+n)
    for n,b in r.model.named_buffers():walk(b,'buffer/'+n)
    for i,y in enumerate(r.model.yarns):
        walk(y.angular_freq,f'yarn/{i}/frequency');walk(y.attn_scale,f'yarn/{i}/scale');walk(y.valid_rows,f'yarn/{i}/valid')
    walk(r.rank_states.states,'rank_states');walk(r.counter_states,'rank_counters');walk(r.sampled_head.COUNTER,'sampled_head_counter')
    walk(r.has_previous,'has_previous');walk(r.previous_adam,'previous_adam')
    for rank,pending in enumerate(r.gradients.pending):
        for p,gradient in pending.items():walk(gradient,f'pending_dense_gradient/{rank}/'+p.label)
    for p,gradient in r.gradients.sums.items():walk(gradient,'summed_dense_gradient/'+p.label)
    for p,values in r.tm.optimizer.param_states.items():walk(values,'optimizer/'+p.label)
    for label,bank in r.tm.optimizer.banks.items():walk(bank.mutated(),'anvil/'+label)
    walk(r.tm.optimizer.split_embed,'optimizer/split_embed')
    for key in ('moment','last','history','event','pending','value_grads'):walk(getattr(r.tables,key),'sparse/'+key)
    # Probe only index-selects from this tensor. Sampled content plus version/storage
    # protects that readonly contract without a130GB transfer in a bounded test.
    table=r.tables.table
    indices=torch.arange(1024,device=table.device,dtype=torch.int64)*(table.shape[0]-1)//1023
    walk(table[indices],'sparse/table_sample')
    result['sparse/table_contract']=dict(pointer=table.data_ptr(),version=table._version,shape=list(table.shape),readonly_in_probe=True,sampled_rows=1024)
    for i,sampler in enumerate(r.samplers):
        for key,value in vars(sampler.builder).items():walk(value,f'sampler/{i}/builder/{key}')
        walk(sampler.staging.next_slot,f'sampler/{i}/next_slot')
        walk(sampler.staging.pending[0] if sampler.staging.pending else None,f'sampler/{i}/pending_step')
        walk(sampler.candidates,f'sampler/{i}/candidates')
        walk(sampler.loss_inputs,f'sampler/{i}/loss_inputs')
        walk(sampler.staging.slots,f'sampler/{i}/staging_slots')
    walk(torch.get_rng_state(),'rng/cpu');walk(torch.cuda.get_rng_state(),'rng/gpu')
    walk(random.getstate(),'rng/python');walk(np.random.get_state(),'rng/numpy')
    return result

def comparable(s):
    return {k:({a:b for a,b in v.items() if a!='pointer'} if isinstance(v,dict) else v) for k,v in s.items()}

@torch.no_grad()
def oracle(probe):
    """Independent direct original-eval invocation under the same fixed state."""
    from dataclasses import replace
    r=probe.runner;probe.prepare();values=[]
    with probe.temporary_state():
        args=replace(r.tm.get_forward_args(),ws_short=640,ws_long=1408,train_max_seq_len=2560,sampled_loss=None)
        for x,t,cu,ids in probe.batches:
            _,slots=torch.unique(ids,sorted=True,return_inverse=True)
            unique=torch.unique(ids,sorted=True)
            torch.index_select(r.tables.table,0,unique.long(),out=r.model.ngram_cache[:unique.numel()])
            values.append(r.model(x,t,cu,slots.int(),args).float().mean())
    return torch.stack(values).cpu().tolist()

def run(r,mode):
    import subprocess
    from wr_reproduction.probe_study import ROOT,sha,now
    atomic(r.research_run/'execution-source-manifest.json',dict(utc=now(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),calibration_sha256=sha(__file__),reason='Calibration helper is imported after graph warmup; record exact loaded revision independently'))
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    r.model.prefix_table.copy_(build_prefix_table_bucket(r.model.vocab_size,0,1))
    for sampler in r.samplers:sampler.set_prefix_table(r.model.prefix_table.cpu().numpy())
    batches=r.loaders(list(range(4)));r.model.limit_yarn_rebuild(r.max_tokens)
    cfg=r.research_config;probe=FixedProbe(r,cfg['probe_protocol'],cfg['backend_hash'])
    results=[];first=None
    for step in range(4):
        loss=r.step(step,batches,next_step=step+1 if step<3 else None,prefetch=True);r.tail.tick(step)
        torch.cuda.synchronize();r.log(event='calibration_update',step=step+1,loss=loss.item(),logical_ranks=8)
        if step==1:
            if mode=='probe':
                from wr_reproduction.probe_continuation import baseline_and_restore
                continuation=baseline_and_restore(r,batches,2)
            before=state(r);atomic(r.research_run/'preprobe-state.json',comparable(before))
            if mode=='probe':
                row=probe.observe(step+1,time.perf_counter());r.log(**row);first=row['chunk_losses'];results.append(row)
                after=state(r)
                changed=[k for k in before if before[k]!=after.get(k)]
                # RNG reads allocate new snapshot tensors; their addresses are immaterial.
                changed=[k for k in changed if not (k.startswith('rng/') or k=='sparse/table_sample') or comparable({k:before[k]})!=comparable({k:after[k]})]
                atomic(r.research_run/'state-restoration.json',dict(checked_entries=len(before),changed=changed,passed=not changed))
                assert not changed,changed[:20]
                direct=oracle(probe);assert max(abs(a-b) for a,b in zip(first,direct))<1e-6
                repeat=probe.observe(step+1,time.perf_counter());r.log(**repeat);results.append(repeat)
                r.log(event='probe_oracle',passed=True,maximum_chunk_loss_difference=max(abs(a-b) for a,b in zip(first,direct)),
                      repeat_max_difference=max(abs(a-b) for a,b in zip(first,repeat['chunk_losses'])),
                      deterministic_repetition_is_not_sampling_uncertainty=True)
        if step==2 and mode=='probe':
            from wr_reproduction.probe_continuation import compare
            parity=compare(r,loss.item(),continuation)
            atomic(r.research_run/'continuation-parity.json',parity)
            r.log(event='continuation_parity',**{k:v for k,v in parity.items() if k!='parameters'})
            del continuation
            row=probe.observe(step+1,time.perf_counter());r.log(**row);results.append(row)
            uncertainty=paired(row['chunk_losses'],first)
            r.log(event='probe_sampling_calibration',paired_adjacent_update=uncertainty,
                  precision_is_model_pair_dependent=True,not_a_global_precision_guarantee=True)
    atomic(r.research_run/'continuation-state.json',comparable(state(r)))
    # CPU tensors for numerical comparison across independently initialized identical runs.
    torch.save({n:p.detach().cpu() for n,p in r.model.named_parameters()},r.research_run/'continuation-parameters.pt')
    value=dict(passed=True,mode=mode,updates=4,all_original_logical_ranks=8,probe_seconds=[x['probe_seconds'] for x in results],
               paired_adjacent_update=uncertainty if mode=='probe' else None,
               full_convergence_verified=False)
    atomic(r.research_run/'calibration.json',value);r.log(event='calibration_complete',**value)
