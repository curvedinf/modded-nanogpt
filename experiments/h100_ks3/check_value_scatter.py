"""Check packed FP16 value-gradient atomics for the 640-wide model."""
import datetime
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import triton.testing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'wr_reproduction/pr393/upstream'))
from track_1_short.perf.kernels.value_embed import _row_scatter_add_kernel


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(717)
    path = sorted((ROOT / 'data/fineweb10B').glob('fineweb_train_*.bin'))[0]
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1
    tokens = np.fromfile(path, dtype=np.uint16, count=49152, offset=1024)
    cases = []
    for count, unique in ((256, True), (16384, False), (49152, False)):
        ids = (torch.arange(count, device='cuda', dtype=torch.int32) if unique else
               torch.tensor(tokens[:count].astype(np.int32), device='cuda'))
        grad = (torch.randn(count, 640, device='cuda') * .01).bfloat16()
        reference = torch.zeros(50304, 640, device='cuda')
        reference.index_add_(0, ids, grad.float())
        results = {}
        for warps in (4, 2, 1):
            storage = torch.full((50306, 640), 17., device='cuda', dtype=torch.float16)
            out = storage[1:-1]
            def launch():
                return _row_scatter_add_kernel[(count,)](
                    grad, ids, out, V=50304, D=640, SEG=128, NSEG=5,
                    num_warps=warps, num_stages=1)
            out.zero_()
            kernel = launch()
            error = ((out.float() - reference).square().sum() / reference.square().sum()).sqrt().item()
            assert torch.isfinite(out).all() and error < .025, (count, warps, error)
            assert torch.all(storage[0] == 17.) and torch.all(storage[-1] == 17.)
            if unique:
                assert torch.equal(out, reference.half()), (count, warps)
            # Include the same buffer clear in both timings; accumulation must
            # not overflow during repeated CUDA-graph replay.
            def timed():
                out.zero_()
                launch()
            samples = [triton.testing.do_bench_cudagraph(timed, rep=80) for _ in range(3)]
            atomic_lines = sorted(set(line.strip() for line in kernel.asm['ptx'].splitlines()
                                      if ('atom.' in line or 'red.' in line) and 'add' in line))
            results[str(warps)] = dict(relative_l2_to_fp32=error, maximum_absolute_error=(out.float()-reference).abs().max().item(),
                                      median_ms=statistics.median(samples), samples_ms=samples,
                                      ptx_atomics=atomic_lines, unique_exact=unique)
        case = dict(tokens=count, unique=unique, variants=results)
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
                  source='Unmodified upstream value_embed._row_scatter_add_kernel; only launch warp count varies.',
                  dataset=str(path), dimension=640, segment=128, cases=cases,
                  timing_includes_identical_buffer_clear=True)
    Path(os.environ['KS3_VALUE_SCATTER_CHECK_OUTPUT']).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
