"""Exact canonical evaluation layout checks; GPU checks run only via registry."""
from collections import Counter
import glob
import statistics
import time
import unittest
import numpy as np
import torch
from control393.eval_attention import bucketed_attention,metadata_bucketed_attention
from control393.attention_metadata import remember_boundaries
from ks3_research.attention import chunks_for
from mi300.portable import varlen_attention


class EvalMetadataTests(unittest.TestCase):
    def test_independent_document_oracle_and_invalidation(self):
        torch.manual_seed(393171)
        n,heads,qd,vd=801,2,8,16
        q=torch.randn(heads,n,qd,dtype=torch.float64).transpose(0,1)
        k=torch.randn_like(q);v=torch.randn(heads,n,vd,dtype=torch.float64).transpose(0,1)
        cu=remember_boundaries(torch.tensor([0,13,269,525,n,n]),[0,13,269,525,n,n])
        for mutate in (False,True):
            if mutate:cu[1]=17
            for window in (0,37,384):
                positions=torch.arange(n)
                docs=torch.bucketize(positions,cu[1:].contiguous(),right=True)
                mask=(positions[:,None]>=positions[None,:])&(positions[:,None]-positions[None,:]<=window)
                mask=mask&(docs[:,None]==docs[None,:])
                logits=torch.einsum('thd,shd->hts',q,k)*.13
                expected=torch.einsum('hts,shd->thd',logits.masked_fill(~mask,float('-inf')).softmax(-1),v)
                actual=metadata_bucketed_attention(q,k,v,cu,window,.13,group_size=2)
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def validation_boundaries(index,n=262144):
    from mi300.config import Hyperparameters
    from mi300.data import HEADER_BYTES,BOS_ID
    path=sorted(glob.glob(Hyperparameters().val_files))[0]
    header=np.fromfile(path,dtype=np.int32,count=256)
    assert tuple(header[:2])==(20240520,1)
    data=np.memmap(path,dtype=np.uint16,mode='r',offset=HEADER_BYTES,shape=(int(header[2]),))
    inputs=data[index*n:(index+1)*n]
    assert inputs.size==n
    # Exactly the unaligned canonical loader's boundaries, including a
    # possible duplicate initial zero. Terminal duplicates have no chunks.
    return np.r_[0,np.flatnonzero(inputs==BOS_ID),n].tolist()


@torch.no_grad()
def run_device_checks(device):
    checks=[];case=None;benchmarks=[]
    try:
        torch.manual_seed(393171)
        for dtype in (torch.float32,torch.bfloat16):
            for n,qd,vd,window in ((529,64,128,0),(529,128,64,384),
                                   (32545,64,128,768),(32545,128,64,2560)):
                case=dict(tokens=n,qdim=qd,vdim=vd,window=window,dtype=str(dtype))
                q=torch.randn(3,n,qd,device=device,dtype=dtype).transpose(0,1)
                k=torch.randn_like(q);v=torch.randn(3,n,vd,device=device,dtype=dtype).transpose(0,1)
                values=[0,17,289,n,n]
                cu=remember_boundaries(torch.tensor(values,device=device,dtype=torch.int32),values)
                expected=bucketed_attention(q,k,v,cu,window,.13)
                original=varlen_attention(q,k,v,cu,window,.13)
                actual=metadata_bucketed_attention(q,k,v,cu,window,.13)
                for reference in (expected,original):
                    if dtype==torch.bfloat16:torch.testing.assert_close(actual,reference,atol=0,rtol=0)
                    else:torch.testing.assert_close(actual,reference,atol=4e-6,rtol=4e-5)
                checks.append(dict(**case,bitwise_equal=bool(torch.equal(actual,expected)),
                                   max_absolute=float((actual-expected).abs().max())))
                del q,k,v,cu,expected,original,actual
        for index in (0,1,39):
            boundaries=validation_boundaries(index)
            for heads,qd,vd,window,paired,layers in ((3,64,128,768,True,3),
                                                    (6,64,64,768,False,2),
                                                    (6,128,128,2560,False,2)):
                case=dict(validation_batch=index,heads=heads,qdim=qd,vdim=vd,window=window,paired=paired)
                rows=262144*(2 if paired else 1)
                values=[x*(2 if paired else 1) for x in boundaries]
                cu=remember_boundaries(torch.tensor(values,device=device,dtype=torch.int32),values)
                q=torch.randn(rows,heads,qd,device=device,dtype=torch.bfloat16)
                k=torch.randn_like(q);v=torch.randn(rows,heads,vd,device=device,dtype=torch.bfloat16)
                scale=.13 if qd==64 else .085
                expected=bucketed_attention(q,k,v,cu,window,scale)
                original=varlen_attention(q,k,v,cu,window,scale)
                actual=metadata_bucketed_attention(q,k,v,cu,window,scale)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                torch.testing.assert_close(actual,original,atol=0,rtol=0)
                checks.append(dict(**case,bitwise_equal=True))
                del expected,original,actual
                if index==1:
                    funcs=dict(original=bucketed_attention,metadata=metadata_bucketed_attention)
                    samples={name:[] for name in funcs}
                    for repeat in range(7):
                        for name in (list(funcs) if repeat%2==0 else list(reversed(funcs))):
                            torch.cuda.synchronize();begun=time.perf_counter()
                            output=funcs[name](q,k,v,cu,window,scale)
                            torch.cuda.synchronize();duration=time.perf_counter()-begun
                            if repeat>=2:samples[name].append(duration)
                            del output
                    medians={name:statistics.median(times) for name,times in samples.items()}
                    shapes=Counter((hi-lo,hi-kl) for lo,hi,kl in chunks_for(values,window))
                    singletons=sum(count%128==1 for count in shapes.values())
                    benchmarks.append(dict(**case,layers=layers,samples_seconds=samples,
                                           median_seconds=medians,singleton_groups=singletons,
                                           speedup=medians['original']/medians['metadata']))
                del q,k,v,cu
    except (RuntimeError,AssertionError) as error:
        return dict(passed=False,checks=checks,failed_case=case,error_type=type(error).__name__,error=str(error))
    weighted={name:sum(c['layers']*c['median_seconds'][name] for c in benchmarks)
              for name in ('original','metadata')}
    return dict(passed=True,checks=checks,benchmarks=benchmarks,weighted_seconds=weighted,
                speedup=weighted['original']/weighted['metadata'],minimum_attention_speedup=1.05,
                criterion='Bitwise BF16 canonical attention against incumbent and original; FP32 original4e-6/4e-5 tolerance; full-model CE must match exactly')


if __name__=='__main__':unittest.main()
