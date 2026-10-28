"""Exact scalar-max oracle and paired timing for the opt-in fused update."""
import datetime
import json
import os
from pathlib import Path

import torch
import triton as tr
import triton.testing

from experiments.h100_ks3.fp8_kernels import merge_amax


def fused(partial, out):
    merge_amax[(tr.cdiv(partial.numel(), 1024),)](partial, out, partial.numel(), num_warps=4)


def original(partial, out):
    torch.maximum(out, partial.amax(), out=out)


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(419)
    checks = 0
    for n in (1, 127, 1024, 18432, 32769):
        for mode in ('finite', 'old_max', 'infinity', 'nan', 'old_nan'):
            values = torch.rand(n, device='cuda')
            previous = 10. if mode == 'old_max' else float('nan') if mode == 'old_nan' else 0.
            if mode == 'infinity':
                values[n // 2] = float('inf')
            if mode == 'nan':
                values[n // 2] = float('nan')
            expected = torch.tensor([previous], device='cuda')
            actual = expected.clone()
            original(values, expected)
            fused(values, actual)
            assert torch.allclose(actual, expected, atol=0., rtol=0., equal_nan=True), (n, mode)
            if mode not in ('nan', 'old_nan'):
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
            checks += 1
    timings = []
    for n in (1536, 12288, 36864):
        values = torch.rand(n, device='cuda')
        a = torch.zeros(1, device='cuda')
        b = torch.zeros_like(a)
        row = dict(elements=n)
        for name, fn, out in [('original', original, a), ('fused', fused, b)]:
            row[name + '_ms'] = tr.testing.do_bench_cudagraph(lambda: fn(values, out), rep=80)
        timings.append(row)
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
                  exact_finite_and_nan_propagation_cases=checks, timings=timings)
    path = Path(os.environ['KS3_AMAX_CHECK_OUTPUT'])
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
