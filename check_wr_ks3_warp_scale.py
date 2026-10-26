"""CPU audit of a KS3-only warp-basis initialization change."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from ks3_research.vendor.ks3_reference import KS3
from wr_reproduction.matched_study import assert_fence, atomic, now, sha


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest()


def measure(raw_scale, inputs, cotangent):
    module = KS3(8, num_subspaces=8, subspace_rank=1, dtype=torch.float64)
    with torch.no_grad():
        module.down.weight.copy_(torch.eye(8, dtype=torch.float64))
        module.up.weight.copy_(torch.eye(8, dtype=torch.float64))
        module.coefficients.zero_()
        module.raw_scale.fill_(raw_scale)
        module.raw_shift.zero_()
    x = inputs.clone().requires_grad_()
    y = module(x)
    (y * cotangent).sum().backward()
    # Independent analytical coefficient derivatives at zero geometry.
    t = 4 * torch.tanh(inputs / 4)
    z = torch.tanh((torch.nn.functional.softplus(torch.tensor(raw_scale, dtype=torch.float64)) + 1e-4) * t)
    derivative = 2 * inputs.relu() * cotangent
    expected = torch.stack(((derivative * (2 * z.square() - 1)).sum(0),
                            (derivative * (4 * z.pow(3) - 3 * z)).sum(0)), -1)
    torch.testing.assert_close(module.coefficients.grad[:, 4:], expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(module.raw_scale.grad, torch.zeros_like(module.raw_scale.grad))
    assert torch.equal(module.raw_shift.grad, torch.zeros_like(module.raw_shift.grad))
    return y.detach(), x.grad, module.coefficients.grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline_path = Path('configs/wr_reproduction/matched_ks3_all.json')
    baseline = json.loads(baseline_path.read_text())
    recipe = json.loads(args.recipe.read_text())
    candidate_scale = recipe['geometry']['raw_scale']
    comparable = json.loads(json.dumps(recipe))
    comparable['geometry']['raw_scale'] = baseline['geometry']['raw_scale']
    comparable['name'] = baseline['name']
    comparable['hypothesis'] = baseline['hypothesis']
    assert comparable == baseline, 'Only KS3 raw warp scale may vary'
    assert candidate_scale == 1.25 and baseline['geometry']['raw_scale'] == .75
    assert recipe['geometry']['coefficients'] == [0] * 6
    generator = torch.Generator().manual_seed(168)
    inputs = torch.randn((257, 8), generator=generator, dtype=torch.float64) * 1.3
    cotangent = torch.randn(inputs.shape, generator=generator, dtype=torch.float64)
    before = measure(.75, inputs, cotangent)
    after = measure(candidate_scale, inputs, cotangent)
    assert torch.equal(before[0], after[0]), 'Initial feedforward function changed'
    assert torch.equal(before[1], after[1]), 'Initial input gradient changed'
    assert torch.equal(before[2][:, :4], after[2][:, :4]), 'Hermite basis changed'
    relative_change = ((before[2][:, 4:] - after[2][:, 4:]).norm()
                       / before[2][:, 4:].norm()).item()
    assert relative_change > .01, 'Warp-basis ablation is ineffective'
    assert not torch.cuda.is_initialized()
    fence = assert_fence(recipe)
    result = dict(utc=now(), passed=True, gpu_initialized=False,
                  recipe_path=str(args.recipe), recipe_sha256=sha(args.recipe),
                  baseline_recipe_sha256=sha(baseline_path),
                  auditor_sha256=sha(Path(__file__)),
                  reference_sha256=sha(Path('ks3_research/vendor/ks3_reference.py')),
                  backend_hash=fence['backend_hash'],
                  only_recipe_change='geometry.raw_scale: 0.75 -> 1.25',
                  initial_function_identical=True, initial_input_gradient_identical=True,
                  initial_raw_scale_and_shift_gradients_zero=True,
                  initial_hermite_coefficient_gradients_identical=True,
                  warp_coefficient_gradients_independently_verified=True,
                  warp_coefficient_gradient_relative_change=relative_change,
                  input_sha256=fingerprint(inputs), cotangent_sha256=fingerprint(cotangent),
                  validation=False,
                  scope='CPU mathematical initialization audit; actual full-model graph warmup remains required.')
    atomic(args.out, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
