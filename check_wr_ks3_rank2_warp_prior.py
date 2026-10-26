"""CPU oracle and full-model initialization audit for immediate rank-two coupling."""
import argparse
import copy
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
    generator = torch.Generator().manual_seed(168)
    inputs = torch.randn((257, 8), generator=generator, dtype=torch.float64) * 1.3
    cotangent = torch.randn(inputs.shape, generator=generator, dtype=torch.float64)
    cell = KS3(8, num_subspaces=4, subspace_rank=2, dtype=torch.float64)
    with torch.no_grad():
        cell.down.weight.copy_(torch.eye(8, dtype=torch.float64))
        cell.up.weight.copy_(torch.eye(8, dtype=torch.float64))
        cell.coefficients.zero_()
        cell.coefficients[..., 4] = .025
        cell.raw_scale.fill_(.75)
        cell.raw_shift.zero_()
    x = inputs.clone().requires_grad_()
    y = cell(x)
    (y * cotangent).sum().backward()
    grouped = inputs.reshape(-1, 4, 2)
    t = 4 * torch.tanh(grouped.mean(-1, keepdim=True) / 4)
    raw_scale = torch.tensor(.75, dtype=torch.float64)
    scale = torch.nn.functional.softplus(raw_scale) + 1e-4
    z = torch.tanh(scale * t)
    pre = grouped + .025 * (2*z.square() - 1)
    expected = pre.relu().square().reshape_as(y)
    local = 2 * pre.relu() * cotangent.reshape_as(grouped)
    dz_factor = .1 * z * (1-z.square())
    shared_dx = local.sum(-1, keepdim=True) * dz_factor * scale * (1-t.square()/16) / 2
    torch.testing.assert_close(y, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(x.grad, (local + shared_dx).reshape_as(x), rtol=1e-11, atol=1e-11)
    bases = (2*t.pow(3)-3*t.square()+1, -2*t.pow(3)+3*t.square(),
             t.pow(3)-2*t.square()+t, t.pow(3)-t.square(),
             2*z.square()-1, 4*z.pow(3)-3*z)
    expected_coefficients = torch.stack([(local * basis).sum(0) for basis in bases], -1)
    expected_scale = (local * dz_factor * t * raw_scale.sigmoid()).sum((0, 2))
    expected_shift = (local * dz_factor * scale * .5).sum((0, 2))
    torch.testing.assert_close(cell.coefficients.grad, expected_coefficients, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(cell.raw_scale.grad, expected_scale, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(cell.raw_shift.grad, expected_shift, rtol=1e-11, atol=1e-11)
    point = torch.tensor([.3, 1.1, .8, .4, 1.2, .2, .5, .9], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(lambda v: cell(v[None])[0], point)
    with torch.no_grad():
        cell.coefficients.zero_()
    baseline = torch.autograd.functional.jacobian(lambda v: cell(v[None])[0], point)
    index = torch.arange(8)
    paired = (index[:, None] // 2 == index[None, :] // 2) & (index[:, None] != index[None, :])
    unrelated = index[:, None] // 2 != index[None, :] // 2
    assert torch.count_nonzero(baseline[paired]) == 0
    assert torch.count_nonzero(jacobian[unrelated]) == 0
    assert jacobian[paired].abs().min() > .001
    return dict(passed=True, analytical_output_and_input_derivative=True,
                all_coefficient_and_shared_warp_derivatives_verified=True,
                initial_paired_coordinate_jacobian_nonzero=True,
                zero_prior_paired_coordinate_jacobian_zero=True,
                unrelated_chart_jacobian_zero=True,
                paired_jacobian_min_abs=jacobian[paired].abs().min().item(),
                paired_jacobian_max_abs=jacobian[paired].abs().max().item(),
                formula='ReLU(x_i + 0.025*(2*z^2-1))^2; z=tanh((softplus(0.75)+1e-4)*4*tanh(mean(x_chart)/4))',
                scope='Unquantized FP64 mathematical reference; quantized GPU correctness is separately established.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base_path = Path('configs/wr_reproduction/matched_ks3_all_rank2.json')
    baseline = json.loads(base_path.read_text())
    recipe = json.loads(args.recipe.read_text())
    normalized = copy.deepcopy(recipe)
    normalized['geometry']['coefficients'] = baseline['geometry']['coefficients']
    for key in ('name', 'hypothesis'):
        normalized[key] = baseline[key]
    assert normalized == baseline
    assert recipe['geometry']['coefficients'] == [0, 0, 0, 0, .025, 0]
    fence = assert_fence(recipe)
    mathematical = mathematical_audit()
    original = ks3_feedforward.initialize_spines
    observed = []

    def initialize(routed, dense, rank, std):
        assert rank == 2 and std == 0
        rng = torch.get_rng_state().clone()
        summaries = []
        for label, value in (('routed', routed), ('dense', dense)):
            assert value.shape[-2:] == (512, 14)
            expected = torch.zeros_like(value)
            expected[..., 4] = expected[..., 10] = .025
            expected[..., 12] = .75
            assert torch.equal(value, expected), label
            summaries.append(dict(branch=label, shape=list(value.shape),
                                  sha256=hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()))
        policy = original(routed, dense, rank, std)
        assert torch.equal(rng, torch.get_rng_state())
        observed.append(dict(policy=policy, geometry=summaries, global_rng_preserved=True))
        return policy

    ks3_feedforward.initialize_spines = initialize
    from wr_reproduction.check_ks3_feedforward_cpu import main as architecture_audit
    sys.argv = ['architecture_audit', '--recipe', str(args.recipe), '--out', str(args.out)]
    architecture_audit()
    assert len(observed) == 1 and not torch.cuda.is_initialized()
    result = json.loads(args.out.read_text())
    result.update(mathematical=mathematical, model_initialization=observed[0],
                  recipe_sha256=sha(args.recipe), baseline_recipe_sha256=sha(base_path),
                  cpu_auditor_sha256=sha(Path(__file__)),
                  reference_sha256=sha(Path('ks3_research/vendor/ks3_reference.py')),
                  backend_hash=fence['backend_hash'],
                  only_recipe_change='geometry.coefficients[4]: 0 -> 0.025 at each coordinate',
                  validation=False,
                  decision='Immediate within-chart coupling and full-model initialization verified on CPU; actual graph warmup remains mandatory.')
    atomic(args.out, result)
    print(json.dumps(dict(passed=True, gpu_initialized=False, mathematical=mathematical)))


if __name__ == '__main__':
    main()
