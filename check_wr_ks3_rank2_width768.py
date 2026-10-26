"""Independent dense and routed GS32/BF16 rank-two768 numerical gates."""
import argparse
import json
import torch

from wr_reproduction import ks3_moe as core
from wr_reproduction.ks3_feedforward import DenseKS3
from wr_reproduction.check_ks3_feedforward_rank import reference, relative_rms
from check_wr_ks3_routed768 import apply_gate_post, cache_checks


FEATURES = 768
CHARTS = 384


def dense_checks():
    checks = []
    for tokens in (1, 257, 4096):
        x = torch.randn(tokens, 768, device='cuda', dtype=torch.bfloat16).requires_grad_()
        up = (torch.randn(FEATURES, 768, device='cuda')*.025).bfloat16().requires_grad_()
        down = (torch.randn(FEATURES, 768, device='cuda')*.02).bfloat16().requires_grad_()
        geometry = torch.randn(CHARTS, 14, device='cuda')*.01
        geometry[:, 12] = .75
        geometry.requires_grad_()
        post = torch.tensor(.7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        output = DenseKS3.apply(x, up, down, core.pack(up.detach()),
                                core.pack(down.detach().T), geometry, 0)*post
        gradient = torch.randn_like(output)*.02
        actual = torch.autograd.grad(output, (x, up, down, geometry, post), gradient)
        xr, ur, dr, gr, pr = [v.detach().clone().requires_grad_()
                             for v in (x, up, down, geometry, post)]
        expected_output = reference(xr, ur, dr, gr, 2)*pr
        expected = torch.autograd.grad(expected_output, (xr, ur, dr, gr, pr), gradient)
        errors = dict(output=relative_rms(output, expected_output))
        for name, a, b in zip(('input_gradient', 'up_gradient', 'down_gradient',
                               'geometry_gradient', 'post_gradient'), actual, expected):
            errors[name] = relative_rms(a, b)
        assert max(errors.values()) < .015, ('dense', tokens, errors)
        checks.append(dict(branch='dense', tokens=tokens, relative_rms=errors))
    return checks


def routed_reference(rows, router, post, up, down, geometry):
    probabilities = torch.softmax(torch.nn.functional.linear(rows, router).float(), -1)
    gate, assignment = probabilities.max(-1)
    values = torch.zeros_like(rows)
    for expert in range(8):
        ids = (assignment == expert).nonzero().flatten()
        if ids.numel():
            result = reference(rows[ids], up[expert], down[expert], geometry[expert], 2)
            values = values.index_copy(0, ids, result)
    return apply_gate_post(values, gate, post), probabilities, assignment


def routed_checks(value):
    checks = []
    for tokens in (1, 257, 4096):
        rows = torch.randn(tokens, 768, device='cuda', dtype=torch.bfloat16)
        router = (torch.randn(8, 768, device='cuda')*.02).bfloat16()
        if tokens == 257:
            rows[:, 0] = rows[:, 0].abs() + .5
            router.zero_(); router[0, 0] = 1.
        rows.requires_grad_(); router.requires_grad_()
        post = torch.tensor(.7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        bank = value.moe_bank.detach().view(5, 2, 8, FEATURES, 768)
        up, down = [bank[0, i].clone().requires_grad_() for i in (0, 1)]
        geometry = torch.randn(8, CHARTS, 14, device='cuda')*.01
        geometry[..., 12] = .75
        geometry.requires_grad_()
        bank_gradient = torch.zeros_like(value.moe_bank)
        output, probabilities, counts = core.RoutedKS3.apply(
            rows, router, post, up, down, value.ks3_up[0], value.ks3_down[0],
            geometry, bank_gradient, 0, True)
        gradient = torch.randn_like(output)*.02
        auxiliary = torch.randn_like(probabilities)*.003
        actual = torch.autograd.grad((output, probabilities), (rows, router, post, geometry),
                                     (gradient, auxiliary))
        xr, rr, pr, ur, dr, gr = [v.detach().clone().requires_grad_()
                                 for v in (rows, router, post, up, down, geometry)]
        expected_output, expected_probabilities, assignment = routed_reference(xr, rr, pr, ur, dr, gr)
        expected = torch.autograd.grad((expected_output, expected_probabilities), (xr, rr, pr, gr, ur, dr),
                                       (gradient, auxiliary))
        assert torch.equal(counts, torch.bincount(assignment, minlength=8).to(counts.dtype))
        assert int((counts > 0).sum()) == (8 if tokens == 4096 else 1)
        errors = dict(output=relative_rms(output, expected_output),
                      probabilities=relative_rms(probabilities, expected_probabilities))
        for name, a, b in zip(('input_gradient', 'router_gradient', 'post_gradient', 'geometry_gradient'), actual, expected):
            errors[name] = relative_rms(a, b)
        errors['up_gradient'] = relative_rms(bank_gradient[:8], expected[4])
        errors['down_gradient'] = relative_rms(bank_gradient[8:16], expected[5])
        assert torch.count_nonzero(bank_gradient[16:]) == 0
        assert max(errors.values()) < .015, ('routed', tokens, errors)
        checks.append(dict(branch='routed', tokens=tokens, active_experts=int((counts > 0).sum()),
                           relative_rms=errors, unrelated_gradient_bank_unchanged=True))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(172)
    from wr_reproduction.backend import install
    install('pr393')
    cls = core.install('bf16', charts=CHARTS, subspace_rank=2, bf16_kernel_mode='tuned')
    value = torch.nn.Module()
    value.moe_bank = torch.nn.Parameter((torch.randn(80, FEATURES, 768, device='cuda')*.025).bfloat16())
    value.register_buffer('ks3_up', torch.empty(5, 8, FEATURES, 768, device='cuda', dtype=torch.bfloat16))
    value.register_buffer('ks3_down', torch.empty(5, 8, 768, FEATURES, device='cuda', dtype=torch.bfloat16))
    caches = cache_checks(value, cls.prepare_ks3_eval)
    assert core.KS3_RANK == 2
    checks = dense_checks() + routed_checks(value)
    result = dict(event='kernel_correctness', passed=True, features=FEATURES, rank=2, charts=CHARTS,
                  checks=checks, cache_checks=caches, relative_rms_tolerance=.015,
                  shared_backend_modified=False, validation=False,
                  oracle='Independent Torch GS32/products/top1 routing, pinned nonlinear forward, independent rank-two smooth STE and original FP32 routed scalar derivatives; dense and routed all-gradient checks plus exact cache/graph replay')
    with open(args.out, 'w') as out:
        out.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
