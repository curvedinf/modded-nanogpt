"""Locate FP8 block overhead after the standalone numerical gate passes."""
import csv
import os
from pathlib import Path
import torch
import triton.testing

os.environ.setdefault('KS3_FP8_CHECK_TOKENS', '16384')
os.environ.setdefault('KS3_FP8_CHECK_OUTPUT', 'research_results/wr_reproduction/h100-research/fp8-profile-check.json')
from experiments.h100_ks3 import check_fp8

original_bench = triton.testing.do_bench_cudagraph
calls = 0


def profiled_bench(function, **kwargs):
    global calls
    index = calls
    calls += 1
    if index % 3 == 0:
        kind = 'bf16' if index % 6 == 0 else 'fp8'
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(8):
                function()
            torch.cuda.synchronize()
        events = sorted(profile.events(), key=lambda e: e.self_device_time_total, reverse=True)
        path = Path('research_results/wr_reproduction/h100-research') / f'{kind}-block-kernels.csv'
        totals = {}
        for event in events:
            if event.device_type == torch.autograd.DeviceType.CUDA:
                count, elapsed = totals.get(event.name, (0, 0.))
                totals[event.name] = (count+1, elapsed+event.self_device_time_total)
        with path.open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['kernel', 'calls', 'total_us', 'per_block_us'])
            for name, (count, elapsed) in sorted(totals.items(), key=lambda pair: pair[1][1], reverse=True):
                writer.writerow([name, count, elapsed, elapsed/8])
        print(f'PROFILE {kind}: {path}', flush=True)
    return original_bench(function, **kwargs)


triton.testing.do_bench_cudagraph = profiled_bench
check_fp8.main()
