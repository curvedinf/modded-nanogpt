"""Paired KS3 ANVIL implementation screen, independent of full training."""
import datetime
import json
import statistics
import sys
from pathlib import Path
import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3 import moe_optimizer as candidate


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(493)
    rows = []
    for width in (1024, 1280):
        grad = torch.randn(10, width, 768, device='cuda', dtype=torch.bfloat16) * .01
        velocity = torch.randn(2, *grad.shape, device='cuda') * .001
        args = [torch.tensor(v, device='cuda') for v in (.9, .85, .4385)]
        split = width > 1024
        ref_velocity = velocity.clone()
        new_velocity = velocity.clone()
        ref = candidate.ORIGINAL(grad, ref_velocity, args[0], split, args[1], args[2], 1.1)
        out = candidate.cascade(grad, new_velocity, args[0], split, args[1], args[2], 1.1)
        error = ((out.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt().item()
        row = dict(width=width, relative_rms=error, output_bitwise_equal=torch.equal(out, ref),
                   velocity_bitwise_equal=torch.equal(ref_velocity, new_velocity))
        assert torch.isfinite(out).all() and error < .002, row
        assert row['velocity_bitwise_equal'], row
        for name, function in [('original', candidate.ORIGINAL), ('fused', candidate.cascade)]:
            v = velocity.clone()
            samples = [triton.testing.do_bench_cudagraph(
                lambda: function(grad, v, args[0], split, args[1], args[2], 1.1), rep=80) for _ in range(3)]
            row[name] = dict(samples_ms=samples, median_ms=statistics.median(samples))
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=rows)
    Path('research_results/wr_reproduction/h100-research/moe-optimizer-micro.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
