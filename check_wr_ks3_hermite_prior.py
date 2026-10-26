"""CPU mathematical and full-model audit for a positive KS3 Hermite slope prior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from ks3_research.vendor.ks3_reference import KS3
from wr_reproduction import ks3_feedforward
from wr_reproduction.matched_study import assert_fence, atomic, sha


def mathematical_audit():
    """Compare the vendored reference with independent analytical derivatives."""
    generator = torch.Generator().manual_seed(168)
    inputs = torch.randn((257, 8), generator=generator, dtype=torch.float64) * 1.3
    cotangent = torch.randn(inputs.shape, generator=generator, dtype=torch.float64)
    cell = KS3(8, num_subspaces=8, subspace_rank=1, dtype=torch.float64)
    with torch.no_grad():
        cell.down.weight.copy_(torch.eye(8, dtype=torch.float64))
        cell.up.weight.copy_(torch.eye(8, dtype=torch.float64))
        cell.coefficients.zero_()
        cell.coefficients[:, 2] = .025
        cell.raw_scale.fill_(.75)
        cell.raw_shift.zero_()
    x = inputs.clone().requires_grad_()
    y = cell(x)
    (y * cotangent).sum().backward()
    t = 4 * torch.tanh(inputs / 4)
    preactivation = inputs + .025 * t * (t - 1).square()
    expected = preactivation.relu().square()
    local = 2 * preactivation.relu() * cotangent
    derivative = 1 + .025 * (3*t.square() - 4*t + 1) * (1 - t.square()/16)
    torch.testing.assert_close(y, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(x.grad, local * derivative, rtol=1e-11, atol=1e-11)
    z = torch.tanh((torch.nn.functional.softplus(torch.tensor(.75, dtype=torch.float64)) + 1e-4) * t)
    bases = (2*t.pow(3)-3*t.square()+1, -2*t.pow(3)+3*t.square(),
             t.pow(3)-2*t.square()+t, t.pow(3)-t.square(),
             2*z.square()-1, 4*z.pow(3)-3*z)
    coefficient_derivatives = torch.stack([(local * basis).sum(0) for basis in bases], -1)
    torch.testing.assert_close(cell.coefficients.grad, coefficient_derivatives, rtol=1e-11, atol=1e-11)
    assert torch.equal(cell.raw_scale.grad, torch.zeros_like(cell.raw_scale.grad))
    assert torch.equal(cell.raw_shift.grad, torch.zeros_like(cell.raw_shift.grad))
    initial_change = ((expected - inputs.relu().square()).norm()
                      / inputs.relu().square().norm()).item()
    assert initial_change > .001
    return dict(passed=True, analytical_output_and_input_derivative=True,
                all_six_coefficient_derivatives_verified=True,
                initial_warp_derivatives_zero=True,
                initial_output_relative_change=initial_change,
                formula='ReLU(x + 0.025 * t * (t - 1)^2)^2; t = 4*tanh(x/4)',
                scope='Unquantized FP64 mathematical reference; existing GPU quantized oracles remain separate.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline_path = Path('configs/wr_reproduction/matched_ks3_all.json')
    baseline = json.loads(baseline_path.read_text())
    recipe = json.loads(args.recipe.read_text())
    comparable = json.loads(json.dumps(recipe))
    comparable['geometry']['coefficients'] = baseline['geometry']['coefficients']
    for name in ('name', 'hypothesis'):
        comparable[name] = baseline[name]
    assert comparable == baseline, 'Only initial KS3 coefficients may differ'
    assert recipe['geometry']['coefficients'] == [0, 0, .025, 0, 0, 0]
    fence = assert_fence(recipe)
    mathematical = mathematical_audit()
    original = ks3_feedforward.initialize_spines
    observed = []

    def initialize(routed, dense, rank, std):
        assert rank == 1 and std == 0
        rng = torch.get_rng_state().clone()
        summaries = []
        for label, value in (('routed', routed), ('dense', dense)):
            expected = torch.zeros_like(value)
            expected[..., 2] = .025
            expected[..., 6] = .75
            assert torch.equal(value, expected), 'Initial KS3 geometry differs: ' + label
            summaries.append(dict(branch=label, shape=list(value.shape),
                                  sha256=hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()))
        policy = original(routed, dense, rank, std)
        assert torch.equal(rng, torch.get_rng_state()), 'Global RNG advanced'
        observed.append(dict(policy=policy, geometry=summaries, global_rng_preserved=True))
        return policy

    ks3_feedforward.initialize_spines = initialize
    from wr_reproduction.check_ks3_feedforward_cpu import main as architecture_audit
    sys.argv = ['architecture_audit', '--recipe', str(args.recipe), '--out', str(args.out)]
    architecture_audit()
    assert len(observed) == 1 and not torch.cuda.is_initialized()
    result = json.loads(args.out.read_text())
    result.update(mathematical=mathematical, model_initialization=observed[0],
                  recipe_sha256=sha(args.recipe), baseline_recipe_sha256=sha(baseline_path),
                  cpu_auditor_sha256=sha(Path(__file__)),
                  reference_sha256=sha(Path('ks3_research/vendor/ks3_reference.py')),
                  backend_hash=fence['backend_hash'],
                  only_recipe_change='geometry.coefficients[2]: 0 -> 0.025',
                  validation=False,
                  decision='Independent CPU initialization audit passed; actual graph warmup remains mandatory.')
    atomic(args.out, result)
    print(json.dumps(dict(passed=True, gpu_initialized=False,
                         coefficients=recipe['geometry']['coefficients'],
                         mathematical=mathematical)))


if __name__ == '__main__':
    main()
