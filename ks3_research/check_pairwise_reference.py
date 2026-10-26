"""CPU-only mathematical gates; does not claim GPU/kernel/convergence parity."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ks3_research.pairwise_reference import block, coupled_rank_one, independent_rank_one, pairwise_term
from ks3_research.vendor.ks3_reference import KS3


def close(a, b, tolerance=1e-10):
    error = (a - b).abs().max().item()
    assert error <= tolerance, error
    return error


def oracle_pair(x, raw, bound):
    # Separate per-feature formulation avoids reusing the custom pair/VJP code.
    return torch.stack([
        bound * torch.tanh(raw[i]) * torch.tanh(x[..., i] / 4.) * torch.tanh(x[..., i ^ 1] / 4.)
        for i in range(x.shape[-1])
    ], -1)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--out', type=Path, required=True);args = parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(601)
    start = time.perf_counter();checks = {}
    assert not torch.cuda.is_initialized(), 'CPU gate must not acquire the training GPU'
    dtype = torch.float64;features = 8
    x = (torch.randn(3, features, dtype=dtype) * .7 + .2).requires_grad_()
    raw = (torch.randn(features, dtype=dtype) * .2).requires_grad_()
    weight = torch.randn_like(x)
    actual = pairwise_term(x, raw, .75);expected = oracle_pair(x, raw, .75)
    checks['pair_output_max_error'] = close(actual, expected)
    ga = torch.autograd.grad(actual, (x, raw), weight, create_graph=True)
    ge = torch.autograd.grad(expected, (x, raw), weight, create_graph=True)
    checks['pair_coordinate_vjp_max_error'] = close(ga[0], ge[0])
    checks['pair_coefficient_vjp_max_error'] = close(ga[1], ge[1])
    checks['finite_difference_gradcheck'] = torch.autograd.gradcheck(lambda a, b: pairwise_term(a, b, .75), (x, raw))
    checks['finite_difference_gradgradcheck'] = torch.autograd.gradgradcheck(lambda a, b: pairwise_term(a, b, .75), (x, raw))
    hessian = torch.autograd.functional.hessian(lambda a: pairwise_term(a, raw, .75)[0], x[0])
    b = torch.tanh(x[0] / 4.)
    cross = .75 * torch.tanh(raw[0]) * (1 - b[0].square()) * (1 - b[1].square()) / 16.
    checks['mixed_derivative_max_error'] = close(hessian[0, 1], cross)
    assert abs(hessian[0, 1]) > 1e-6
    checks['nonzero_cross_derivative'] = hessian[0, 1].item()
    assert torch.count_nonzero(hessian[0, 2:]) == 0
    huge = torch.linspace(-1e5, 1e5, 64, dtype=dtype).reshape(8, 8)
    bounded = pairwise_term(huge, torch.full((8,), 100., dtype=dtype), .75)
    assert bounded.abs().max() <= .75
    checks['maximum_bounded_added_feature'] = bounded.abs().max().item()

    # The original vendor module is the independent rank-one baseline oracle.
    base = KS3(5, 4, num_subspaces=features, subspace_rank=1, dtype=dtype)
    inputs = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    geo = torch.cat((base.coefficients.detach(), base.raw_scale.detach()[:, None], base.raw_shift.detach()[:, None]), -1).requires_grad_()
    up = base.down.weight.detach().clone().requires_grad_()
    down = base.up.weight.detach().clone().requires_grad_()
    zero = torch.zeros(features, dtype=dtype, requires_grad=True)
    y0 = base(inputs);y = block(inputs, up, down, geo, zero)
    assert torch.equal(y0, y)
    checks['zero_pair_initial_output_bitwise_equal'] = True
    dy = torch.randn_like(y)
    got = torch.autograd.grad(y, (inputs, up, down, geo, zero), dy)
    ref = torch.autograd.grad(y0, (inputs, base.down.weight, base.up.weight, base.coefficients, base.raw_scale, base.raw_shift), dy)
    geo_ref = torch.cat((ref[3], ref[4][:, None], ref[5][:, None]), -1)
    checks['zero_pair_existing_gradient_max_errors'] = [close(a, b) for a, b in zip(got[:4], (*ref[:3], geo_ref))]
    assert got[4].abs().max() > 1e-6
    checks['zero_pair_new_coefficient_gradient_max_abs'] = got[4].abs().max().item()
    independent_hessian = torch.autograd.functional.hessian(lambda a: coupled_rank_one(a, geo, zero)[0], x[0])
    assert independent_hessian[0, 1] == 0
    checks['zero_pair_cross_derivative_zero'] = True

    # Nonzero coefficients: compare all input, projection and geometry VJPs.
    nz = raw.detach().clone().requires_grad_();test_inputs = inputs.detach().clone().requires_grad_()
    candidate = block(test_inputs, up, down, geo, nz, .75)
    coords = F.linear(test_inputs, up)
    oracle = F.linear(independent_rank_one(coords, geo) + oracle_pair(coords.clamp(-65504., 65504.), nz, .75), down)
    checks['nonzero_pair_block_output_max_error'] = close(candidate, oracle)
    params = (test_inputs, up, down, geo, nz)
    a = torch.autograd.grad(candidate, params, dy);b = torch.autograd.grad(oracle, params, dy)
    checks['nonzero_pair_input_weight_geometry_gradient_max_errors'] = [close(v, w) for v, w in zip(a, b)]
    shape_checks = []
    for width in (1024, 2816):
        c = torch.randn(3, width, dtype=dtype);g = torch.zeros(width, 8, dtype=dtype);g[:, 6] = .75
        h = coupled_rank_one(c, g, torch.zeros(width, dtype=dtype))
        assert torch.equal(h, independent_rank_one(c, g))
        shape_checks.append(dict(width=width,shape_preserved=list(h.shape),zero_initial_function_exact=True))
    checks['original_width_checks'] = shape_checks
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), Path(__file__).with_name('pairwise_reference.py'), Path(__file__).parent/'vendor/ks3_reference.py']
    result = dict(event='pairwise_cpu_mathematical_gate',passed=True,seed=601,device='cpu',seconds=time.perf_counter()-start,
                  checks=checks,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  initial_pair_coefficients=0,independent_rank_one_pathways_retained=True,coordinate_averaging=False,
                  gpu_kernel_tested=False,quantized_ste_tested=False,convergence_claim=False,
                  remaining_gates=['GS32/FP16 GPU forward and all-gradient checks','actual-model graph execution after five isolated placement trials'])
    args.out.parent.mkdir(parents=True, exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
