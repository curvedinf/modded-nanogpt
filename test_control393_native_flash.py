"""Registered native flash provider probe; fixed existing BF16 parity tolerances."""
import statistics
import time
import unittest
import torch
from control393.native_attention import clean_boundaries,native_or_bucketed_eval,NativeFlashEvalSelfAttention
from control393.native_attention import efficient_or_bucketed_eval,NativeEfficientEvalSelfAttention
from control393.bucket_training import bucket_training_attention
from test_ks3_attention import direct_check,wrapper_check


class BoundaryTests(unittest.TestCase):
    def test_only_empty_documents_are_removed(self):
        self.assertEqual(clean_boundaries([0,0,17,17,289,529,529],529),([0,17,289,529],272))
    def test_coverage_and_order_are_required(self):
        for values,tokens in (([1,17],17),([0,16],17),([0,8,7,17],17),([0,0],0)):
            with self.assertRaises(ValueError):clean_boundaries(values,tokens)


def benchmark(device,implementation=native_or_bucketed_eval):
    torch.manual_seed(393142);n=49152
    q=torch.randn(n,6,128,device=device,dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
    boundaries=[0,17,529]
    while boundaries[-1]<n:boundaries.append(min(n,boundaries[-1]+2048))
    cu=torch.tensor(boundaries+[n],device=device,dtype=torch.int32)
    upstream=torch.randn_like(v)/v.numel()**.5
    samples={'bucketed':[],'native':[]}
    funcs={'bucketed':bucket_training_attention,'native':implementation}
    for repetition in range(7):
        for name in (('bucketed','native') if repetition%2==0 else ('native','bucketed')):
            torch.cuda.synchronize();begun=time.perf_counter()
            out=funcs[name](q,k,v,cu,1408,.13)
            gradients=torch.autograd.grad(out,(q,k,v),upstream)
            torch.cuda.synchronize();duration=time.perf_counter()-begun
            if repetition>=2:samples[name].append(duration)
            del out,gradients
    medians={name:statistics.median(values) for name,values in samples.items()}
    return dict(tokens=n,samples_seconds=samples,median_seconds=medians,
                speedup=medians['bucketed']/medians['native'])


def run_device_checks(device,requested=('aotriton','ck')):
    original=torch.backends.cuda.preferred_rocm_fa_library();providers={}
    try:
        for provider in requested:
            try:
                if provider=='efficient':
                    implementation,layer=efficient_or_bucketed_eval,NativeEfficientEvalSelfAttention
                else:
                    torch.backends.cuda.preferred_rocm_fa_library(provider)
                    implementation,layer=native_or_bucketed_eval,NativeFlashEvalSelfAttention
                checks=[direct_check(device,torch.bfloat16,qd,vd,window,
                                     implementation,n=n)
                        for n,qd,vd,window in ((529,64,64,0),(529,64,128,17),
                            (529,128,64,384),(32545,64,128,1408))]
                checks.extend(wrapper_check(device,torch.bfloat16,paired,qd,vd,layer)
                              for paired,qd,vd in ((False,128,64),(True,64,128)))
                providers[provider]=dict(passed=True,checks=checks,benchmark=benchmark(device,implementation))
            except (RuntimeError,AssertionError,NotImplementedError) as error:
                if 'illegal memory' in str(error).lower() or 'device-side assert' in str(error).lower():raise
                providers[provider]=dict(passed=False,error_type=type(error).__name__,error=str(error))
    finally:
        torch.backends.cuda.preferred_rocm_fa_library(original)
    return dict(passed=any(result['passed'] for result in providers.values()),providers=providers,
                criterion='existing BF16 tests: relativeRMS<=.015 and absolute<=max(2e-4,.03*max_reference); exact canonical evaluation')


if __name__=='__main__':unittest.main()
