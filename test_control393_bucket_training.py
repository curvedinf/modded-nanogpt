"""Shape-grouped attention parity and paired timings, GPU only via registry."""
from itertools import accumulate
import statistics
import time
import unittest
import torch
from control393.bucket_training import (bucket_training_attention,BucketTrainingSelfAttention,
    reverse_context_reduce,shape_groups)
from control393.attention128 import fused128_attention
from ks3_research.attention import chunks_for
from test_ks3_attention import direct_check,wrapper_check


def reduction_check(device,dtype,width):
    torch.manual_seed(398)
    n,heads=7969,3
    chunks=chunks_for([0,17,1051,3741,n],1408)
    offsets=[0,*accumulate(hi-lo for _,hi,lo in chunks)]
    packed=torch.randn(offsets[-1],heads,width,device=device,dtype=dtype)
    template=torch.empty(heads,n,width,device=device,dtype=dtype).transpose(0,1)
    expected=torch.zeros_like(template)
    for i in reversed(range(len(chunks))):
        _,hi,lo=chunks[i]
        expected[lo:hi].add_(packed[offsets[i]:offsets[i+1]])
    actual=reverse_context_reduce(packed,chunks,offsets,template)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    return dict(case='ordered_packed_reduction',dtype=str(dtype),width=width,bitwise_equal=True)


def benchmark(device):
    torch.manual_seed(399)
    n,heads,width=49152,6,128
    boundaries=[0];lengths=[517,2049,1153,3072,769,1537]
    while boundaries[-1]<n:
        boundaries.append(min(n,boundaries[-1]+lengths[(len(boundaries)-1)%len(lengths)]))
    seqlens=torch.tensor(boundaries,device=device,dtype=torch.int32)
    params=[torch.randn(n,heads,width,device=device,dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    upstream=torch.randn_like(params[0])/n**.5
    implementations={'g128':fused128_attention,'shape_groups':bucket_training_attention}
    times={name:[] for name in implementations}
    for repeat in range(6):
        order=list(implementations) if repeat%2==0 else list(reversed(implementations))
        for name in order:
            torch.cuda.synchronize(device);begun=time.perf_counter()
            y=implementations[name](*params,seqlens,1408,.125)
            gradients=torch.autograd.grad(y,params,upstream)
            torch.cuda.synchronize(device);duration=time.perf_counter()-begun
            del y,gradients
            if repeat>=2:times[name].append(duration)
    medians={name:statistics.median(values) for name,values in times.items()}
    return dict(tokens=n,heads=heads,width=width,window=1408,samples=times,median_seconds=medians,
                speedup=medians['g128']/medians['shape_groups'],minimum_trial_speedup=1.05)


def run_device_checks(device):
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        for width in (64,128):checks.append(reduction_check(device,dtype,width))
        for qdim,vdim,window in ((64,128,17),(128,64,384),(64,64,0)):
            checks.append(direct_check(device,dtype,qdim,vdim,window,bucket_training_attention))
        for paired,qdim,vdim in ((False,128,64),(True,64,128)):
            checks.append(wrapper_check(device,dtype,paired,qdim,vdim,BucketTrainingSelfAttention))
        checks.append(direct_check(device,dtype,64,128,1408,bucket_training_attention,n=65313))
    return dict(passed=True,checks=checks,benchmark=benchmark(device))


class BucketTrainingTests(unittest.TestCase):
    def test_independent_document_attention_and_gradients(self):
        torch.manual_seed(397)
        n=273
        q=torch.randn(n,2,8,dtype=torch.float64,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True)
        v=torch.randn(n,2,16,dtype=torch.float64,requires_grad=True)
        boundaries=torch.tensor([0,13,n,n],dtype=torch.int32)
        actual=bucket_training_attention(q,k,v,boundaries,37,.13)
        pos=torch.arange(n);docs=pos>=13
        mask=(pos[:,None]>=pos[None,:])&(pos[:,None]-pos[None,:]<=37)&(docs[:,None]==docs[None,:])
        logits=torch.einsum('thd,shd->hts',q,k)*.13
        expected=torch.einsum('hts,shd->thd',logits.masked_fill(~mask,float('-inf')).softmax(-1),v)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        upstream=torch.randn_like(actual)
        a=torch.autograd.grad(actual,(q,k,v),upstream)
        b=torch.autograd.grad(expected,(q,k,v),upstream)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)

    def test_groups_cover_every_chunk_once(self):
        chunks=chunks_for([0,17,1051,3741,7969],1408)
        ids=[i for group in shape_groups(chunks) for i in group]
        self.assertEqual(sorted(ids),list(range(len(chunks))))
        for window in (0,17,384,1408,2560):
            current=chunks_for([0,17,1051,3741,7969],window)
            self.assertEqual([c[2] for c in current],sorted(c[2] for c in current))

    def test_ordered_reduction_is_bitwise(self):
        for dtype in (torch.float32,torch.bfloat16):
            reduction_check('cpu',dtype,64)


if __name__=='__main__':unittest.main()
