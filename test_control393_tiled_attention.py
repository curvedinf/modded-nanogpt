"""Explicit local Triton attention probe using existing attention parity gates."""
import torch
import glob
import statistics
import time
import numpy as np
from control393.tiled_attention import tiled_or_bucketed_eval,TiledEvalSelfAttention
from control393.tiled_attention import compact_tiled_or_bucketed_eval,CompactTiledEvalSelfAttention
from control393.tiled_attention import async_tiled_or_bucketed_eval,AsyncTiledEvalSelfAttention
from control393.attention_metadata import remember_boundaries
from test_ks3_attention import direct_check,wrapper_check,errors
from test_control393_native_flash import benchmark
from control393.bucket_training import bucket_training_attention
from mi300.portable import varlen_attention
from ks3_research.attention import fused64_attention


def measured_errors(actual,expected):
    delta=actual.detach().float()-expected.detach().float()
    report=dict(max_absolute=float(delta.abs().max()),
                relative_rms=float(delta.norm()/expected.detach().float().norm().clamp_min(1e-12)),
                reference_max=float(expected.detach().float().abs().max()))
    try:errors(actual,expected,torch.bfloat16)
    except AssertionError:report['passed']=False
    else:report['passed']=True
    return report


def dense_failure_diagnosis(q,k,v,cu,window,scale,upstream,outputs,gradients):
    """Independent FP64 document oracle at the largest discrepant gradient rows."""
    boundaries=cu.cpu().tolist();documents=set();worst=[]
    for label,a,b in zip(('query','key','value'),gradients['tiled'],gradients['bucketed']):
        row=int((a.float()-b.float()).square().sum((1,2)).argmax())
        document=int(np.searchsorted(boundaries,row,side='right')-1)
        documents.add(document);worst.append(dict(gradient=label,row=row,document=document))
    checks=[]
    for doc in sorted(documents):
        lo,hi=boundaries[doc:doc+2]
        params=[x[lo:hi].detach().double().requires_grad_() for x in (q,k,v)]
        positions=torch.arange(hi-lo,device=q.device)
        mask=(positions[:,None]>=positions[None,:])&(positions[:,None]-positions[None,:]<=window)
        logits=torch.einsum('thd,shd->hts',params[0],params[1])*scale
        exact=torch.einsum('hts,shd->thd',logits.masked_fill(~mask,float('-inf')).softmax(-1),params[2])
        exact_gradients=torch.autograd.grad(exact,params,upstream[lo:hi].double())
        checks.append(dict(document=doc,start=lo,end=hi,
                           forward={name:measured_errors(y[lo:hi],exact) for name,y in outputs.items()},
                           gradients={name:[dict(name=label,**measured_errors(a[lo:hi],b))
                                            for label,a,b in zip(('query','key','value'),gs,exact_gradients)]
                                      for name,gs in gradients.items()}))
        del params,logits,exact,exact_gradients
    return dict(worst_rows=worst,documents=checks,oracle='explicit FP64 document-local softmax and autograd')


def real_document_benchmark(device,n=114688,max_seq_len=3072,short_window=640,long_window=1408,
                            implementation=tiled_or_bucketed_eval,audit_control=False):
    """Paired timing with unchanged loader boundaries and production head shapes."""
    from mi300.config import Hyperparameters
    from mi300.data import HEADER_BYTES,Shard
    path=sorted(glob.glob(Hyperparameters().train_files))[0]
    header=np.fromfile(path,dtype=np.int32,count=256)
    assert tuple(header[:2])==(20240520,1)
    tokens=np.memmap(path,dtype=np.uint16,mode='c',offset=HEADER_BYTES,shape=(int(header[2]),))
    shard=Shard(torch.from_numpy(tokens));shard._loader_thread.join()
    starts,ends=shard.next_batch(n,max_seq_len)
    lengths=np.asarray(ends[0])-np.asarray(starts[0])
    boundaries=np.minimum(np.r_[0,np.cumsum(lengths)],n)
    cases=[]
    # Three paired narrow layers, two half-V layers, two wide long-window layers.
    for heads,qd,vd,window,paired,layers in ((3,64,128,short_window,True,3),
                                           (6,64,64,short_window,False,2),
                                           (6,128,128,long_window,False,2)):
        torch.manual_seed(393149)
        rows=n*(2 if paired else 1)
        cu=torch.tensor(boundaries*(2 if paired else 1),device=device,dtype=torch.int32)
        if implementation is async_tiled_or_bucketed_eval:
            remember_boundaries(cu,boundaries*(2 if paired else 1))
        q=torch.randn(rows,heads,qd,device=device,dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn(rows,heads,vd,device=device,dtype=torch.bfloat16,requires_grad=True)
        upstream=torch.randn_like(v)/v.numel()**.5
        funcs={'bucketed':bucket_training_attention,'tiled':implementation}
        if implementation is not tiled_or_bucketed_eval:funcs['incumbent_tiled']=tiled_or_bucketed_eval
        samples={name:[] for name in funcs}
        parity=None
        for repeat in range(7):
            for name in (list(funcs) if repeat%2==0 else list(reversed(funcs))):
                torch.cuda.synchronize();begun=time.perf_counter()
                output=funcs[name](q,k,v,cu,window,.13 if qd==64 else .085)
                gradients=torch.autograd.grad(output,(q,k,v),upstream)
                torch.cuda.synchronize();duration=time.perf_counter()-begun
                if repeat>=2:samples[name].append(duration)
                if repeat==0 and name=='bucketed':
                    reference=(output.detach(),tuple(g.detach() for g in gradients))
                elif repeat==0 and name=='tiled':
                    parity=dict(forward=measured_errors(output,reference[0]),
                                gradients=[dict(name=label,**measured_errors(a,b)) for label,a,b in
                                           zip(('query','key','value'),gradients,reference[1])])
                    scale=.13 if qd==64 else .085
                    original=varlen_attention(q,k,v,cu,window,scale)
                    original_gradients=torch.autograd.grad(original,(q,k,v),upstream)
                    parity['original_comparison']={
                        'tiled_forward':measured_errors(output,original),
                        'bucketed_forward':measured_errors(reference[0],original),
                        'tiled':[dict(name=label,**measured_errors(a,b)) for label,a,b in
                                 zip(('query','key','value'),gradients,original_gradients)],
                        'bucketed':[dict(name=label,**measured_errors(a,b)) for label,a,b in
                                    zip(('query','key','value'),reference[1],original_gradients)]}
                    control_outputs={};control_grads={}
                    if audit_control:
                        # Optional historical audit. Already recorded153-156;
                        # ordinary new-kernel checks do not repeat it.
                        control=fused64_attention(q,k,v,cu,window,scale)
                        control_gradients=torch.autograd.grad(control,(q,k,v),upstream)
                        parity['control_g64_audit']=dict(
                            forward=measured_errors(control,original),
                            gradients=[dict(name=label,**measured_errors(a,b)) for label,a,b in
                                       zip(('query','key','value'),control_gradients,original_gradients)])
                        control_outputs['control_g64']=control
                        control_grads['control_g64']=control_gradients
                    with torch.no_grad():
                        evaluation=implementation(q,k,v,cu,window,scale)
                        reference_eval=varlen_attention(q,k,v,cu,window,scale)
                        torch.testing.assert_close(evaluation,reference_eval,atol=0,rtol=0)
                        parity['canonical_evaluation_bitwise_equal']=True
                    parity['dense_oracle']=dense_failure_diagnosis(
                        q,k,v,cu,window,scale,upstream,
                        dict(tiled=output,bucketed=reference[0],original=original,**control_outputs),
                        dict(tiled=gradients,bucketed=reference[1],original=original_gradients,
                             **control_grads))
                    del original,original_gradients,evaluation,reference_eval,control_outputs,control_grads
                    if audit_control:del control,control_gradients
                    del reference
                    if implementation is async_tiled_or_bucketed_eval:
                        exact_candidate=(output.detach(),tuple(g.detach() for g in gradients))
                elif repeat==0 and name=='incumbent_tiled' and implementation is async_tiled_or_bucketed_eval:
                    torch.testing.assert_close(output,exact_candidate[0],atol=0,rtol=0)
                    for a,b in zip(gradients,exact_candidate[1]):
                        torch.testing.assert_close(a,b,atol=0,rtol=0)
                    parity['incumbent_forward_and_gradients_bitwise_equal']=True
                    del exact_candidate
                del output,gradients
        medians={name:statistics.median(values) for name,values in samples.items()}
        cases.append(dict(tokens=rows,documents=len(boundaries)-1,heads=heads,qdim=qd,vdim=vd,
                          window=window,paired=paired,layers=layers,samples_seconds=samples,
                          parity=parity,median_seconds=medians,speedup=medians['bucketed']/medians['tiled']))
        del q,k,v,upstream,cu
    totals={name:sum(c['layers']*c['median_seconds'][name] for c in cases) for name in samples}
    passed=all(item['passed'] for c in cases for item in [
        c['parity']['forward'],*c['parity']['gradients'],
        c['parity']['original_comparison']['tiled_forward'],
        c['parity']['original_comparison']['bucketed_forward'],
        *c['parity']['original_comparison']['tiled'],
        *c['parity']['original_comparison']['bucketed']])
    passed=passed and all(item['passed'] for c in cases
                         for doc in c['parity']['dense_oracle']['documents']
                         for item in [*(doc['forward'][name] for name in ('tiled','bucketed','original')),
                                      *(item for name in ('tiled','bucketed','original')
                                        for item in doc['gradients'][name])])
    control_passed=(all(item['passed'] for c in cases for item in [
        c['parity']['control_g64_audit']['forward'],*c['parity']['control_g64_audit']['gradients']])
        if audit_control else None)
    return dict(passed=passed,control_attention_audit_passed=control_passed,
                batch_tokens=n,max_seq_len=max_seq_len,windows=[short_window,long_window],
                cases=cases,weighted_seconds=totals,speedup=totals['bucketed']/totals['tiled'],
                incumbent_speedup=totals.get('incumbent_tiled',totals['tiled'])/totals['tiled'])


def run_device_checks(device,compact=False,async_metadata=False):
    checks=[];case=None
    if compact and async_metadata:raise ValueError('Validate one attention metadata ablation at a time')
    implementation=(async_tiled_or_bucketed_eval if async_metadata else
                    compact_tiled_or_bucketed_eval if compact else tiled_or_bucketed_eval)
    layer=(AsyncTiledEvalSelfAttention if async_metadata else
           CompactTiledEvalSelfAttention if compact else TiledEvalSelfAttention)
    try:
        if async_metadata:
            from test_control393_attention_metadata import run_device_checks as metadata_checks
            case=dict(verification='host_metadata_lifetime')
            checks.append(metadata_checks(device))
        for n,qd,vd,window in ((529,64,64,0),(529,64,128,17),(529,128,64,384),
                               (32545,64,128,1408),(32545,128,64,1664)):
            case=dict(tokens=n,qdim=qd,vdim=vd,window=window)
            checks.append(direct_check(device,torch.bfloat16,qd,vd,window,implementation,n=n))
        for paired,qd,vd in ((False,128,64),(True,64,128)):
            case=dict(wrapper=True,paired=paired,qdim=qd,vdim=vd)
            checks.append(wrapper_check(device,torch.bfloat16,paired,qd,vd,layer))
        case=dict(benchmark='regular_documents')
        timings=benchmark(device,implementation)
        case=dict(benchmark='real_documents_and_independent_oracles')
        real_timings=real_document_benchmark(device,implementation=implementation)
        additional=[]
        for n,length,short,long in ((32768,896,128,384),(65536,2048,384,896),
                                    (40960,3072,640,1408),(16384,3072,768,1664)):
            case=dict(benchmark='additional_stage',tokens=n,window=[short,long])
            additional.append(real_document_benchmark(device,n,length,short,long,implementation))
    except (RuntimeError,AssertionError) as error:
        return dict(passed=False,checks=checks,failed_case=case,error_type=type(error).__name__,error=str(error))
    return dict(passed=real_timings['passed'] and all(x['passed'] for x in additional),
                checks=checks,benchmark=timings,real_document_benchmark=real_timings,
                additional_stage_checks=additional,
                criterion='existing BF16 relativeRMS<=.015 and abs<=max(2e-4,.03*max_reference); exact canonical evaluation')
