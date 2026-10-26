"""Diagnostic of routed768 scalar-gradient rounding; never a passing gate.

The supported768-wide routed GS32 mode is checked against
Torch routing/products and pinned nonlinear forward with independent STEs.
These are numerical checks, never validation or convergence measurements.
"""
import argparse
import json
import torch

from wr_reproduction import ks3_moe as core
from wr_reproduction.check_ks3_feedforward_bf16_direct import reference as direct_reference
from wr_reproduction.check_ks3_feedforward_rank import reference as gs32_reference
from wr_reproduction.check_ks3_feedforward_rank import quantized_value, relative_rms


REFERENCE_CAPTURE = {}

def reference(rows, router, post, up, down, geometry, gs32):
    probabilities = torch.softmax(torch.nn.functional.linear(rows, router).float(), -1)
    gate, assignment = probabilities.max(-1)
    values = torch.zeros_like(rows)
    for expert in range(up.shape[0]):
        ids = (assignment == expert).nonzero().flatten()
        if ids.numel():
            if gs32:
                result = gs32_reference(rows[ids], up[expert], down[expert], geometry[expert], 1)
            else:
                result = direct_reference(rows[ids], up[expert], down[expert].T, geometry[expert])
            values = values.index_copy(0, ids, result)
    REFERENCE_CAPTURE.update(values=values.detach(),gate=gate.detach())
    return values * (gate * post.float()).to(values.dtype)[:, None], probabilities, assignment


def cache_model(features):
    from wr_reproduction.backend import install
    install('pr393')
    cls = core.install('bf16', charts=features, bf16_kernel_mode='tuned')
    value = torch.nn.Module()
    value.moe_bank = torch.nn.Parameter((torch.randn(80, features, 768, device='cuda')*.025).bfloat16())
    value.register_buffer('ks3_up', torch.empty(5, 8, features, 768, device='cuda', dtype=torch.bfloat16))
    value.register_buffer('ks3_down', torch.empty(5, 8, 768, features, device='cuda', dtype=torch.bfloat16))
    return value, cls.prepare_ks3_eval


def check_cache(value, gs32):
    bank = value.moe_bank.view(5, 2, 8, value.ks3_up.shape[2], 768)
    for layer in range(5):
        for expert in range(8):
            for weights, cached in ((bank[layer, 0, expert], value.ks3_up[layer, expert]),
                                    (bank[layer, 1, expert].T, value.ks3_down[layer, expert])):
                expected = quantized_value(weights).bfloat16() if gs32 else weights
                assert torch.equal(cached, expected)


def cache_checks(value, refresh):
    pointers = {name: tensor.data_ptr() for name, tensor in value.named_buffers()}
    checks = []
    with torch.no_grad():
        for flag in (True,):
            value.ks3_routed_training_gs32 = flag
            for training in (True, False, True):
                value.train(training)
                core.BF16_TUNED_KERNELS = training
                refresh(value)
                expected_gs32 = not training or flag
                check_cache(value, expected_gs32)
                checks.append(dict(training=training, routed_training_gs32=flag,
                                   cache_gs32=expected_gs32, cache_bytes_exact=True))
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            refresh(value)
        for _ in range(2):
            value.moe_bank.add_(torch.randn_like(value.moe_bank)*.001)
            graph.replay()
            check_cache(value, True)
            checks.append(dict(captured_replay=True, cache_bytes_exact=True))
    assert pointers == {name: tensor.data_ptr() for name, tensor in value.named_buffers()}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--features', type=int, choices=(768,), default=768)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(172)
    core.KS3_RANK = 1
    value, refresh = cache_model(args.features)
    caches = cache_checks(value, refresh)
    checks = []
    features = args.features
    for gs32 in (True,):
        core.BF16_TUNED_KERNELS = True
        value.train()
        value.ks3_routed_training_gs32 = gs32
        refresh(value)
        for tokens in (1, 257, 4096):
            rows = torch.randn(tokens, 768, device='cuda', dtype=torch.bfloat16)
            router = (torch.randn(8, 768, device='cuda')*.02).bfloat16()
            if tokens == 257:
                # Seven empty experts exercise padded gathers and gradient masks.
                rows[:, 0] = rows[:, 0].abs() + .5
                router.zero_()
                router[0, 0] = 1.
            rows.requires_grad_(); router.requires_grad_()
            post = torch.tensor(.7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            bank = value.moe_bank.detach().view(5, 2, 8, features, 768)
            up = bank[0, 0].clone().requires_grad_()
            down = bank[0, 1].clone().requires_grad_()
            geometry = torch.randn(8, features, 8, device='cuda')*.01
            geometry[..., 6] = .75
            geometry.requires_grad_()
            bank_gradient = torch.zeros_like(value.moe_bank)
            output, probabilities, counts = core.RoutedKS3.apply(
                rows, router, post, up, down, value.ks3_up[0], value.ks3_down[0],
                geometry, bank_gradient, 0, gs32)
            saved = output.grad_fn.saved_tensors
            candidate_values = saved[10].detach()[saved[6].long()]
            candidate_gate = saved[4].detach()
            gradient = torch.randn_like(output)*.02
            auxiliary_gradient = torch.randn_like(probabilities)*.003
            actual = torch.autograd.grad((output, probabilities), (rows, router, post, geometry),
                                         (gradient, auxiliary_gradient))
            xr, rr, pr, ur, dr, gr = [tensor.detach().clone().requires_grad_()
                                     for tensor in (rows, router, post, up, down, geometry)]
            expected_output, expected_probabilities, assignment = reference(xr, rr, pr, ur, dr, gr, gs32)
            expected = torch.autograd.grad((expected_output, expected_probabilities), (xr, rr, pr, gr, ur, dr),
                                           (gradient, auxiliary_gradient))
            assert torch.equal(counts, torch.bincount(assignment, minlength=8).to(counts.dtype))
            if tokens == 257:
                assert (counts[1:] == 0).all()
            errors = dict(output=relative_rms(output, expected_output),
                          probabilities=relative_rms(probabilities, expected_probabilities))
            for name, a, b in zip(('input_gradient', 'router_gradient', 'post_gradient', 'geometry_gradient'), actual, expected):
                errors[name] = relative_rms(a, b)
            errors['up_gradient'] = relative_rms(bank_gradient[:8], expected[4])
            errors['down_gradient'] = relative_rms(bank_gradient[8:16], expected[5])
            assert torch.count_nonzero(bank_gradient[16:]) == 0
            ref_values = REFERENCE_CAPTURE['values']
            ref_gate = REFERENCE_CAPTURE['gate']
            ref_dot64 = (gradient.double()*ref_values.double()*ref_gate.double()[:,None]).sum()
            candidate_dot64 = (gradient.double()*candidate_values.double()*candidate_gate.double()[:,None]).sum()
            ref_rows32 = (gradient.float()*ref_values.float()).sum(-1)
            expected_post32 = (ref_rows32*ref_gate).sum()
            candidate_rows32 = (gradient.float()*candidate_values.float()).sum(-1)
            candidate_post32 = (candidate_rows32*candidate_gate).sum()
            bf16_rows = (gradient*ref_values).sum(-1)
            bf16_expected_post = (bf16_rows.float()*ref_gate).sum().to(post.dtype)
            diagnostics = dict(actual_post=float(actual[2]),original_autograd_post=float(expected[2]),
                reference_fp64_dot=float(ref_dot64),candidate_fp64_dot=float(candidate_dot64),
                reference_fp64_dot_bf16=float(ref_dot64.to(post.dtype)),
                candidate_fp64_dot_bf16=float(candidate_dot64.to(post.dtype)),
                reference_fp32_post=float(expected_post32),candidate_fp32_post=float(candidate_post32),
                reference_fp32_post_bf16=float(expected_post32.to(post.dtype)),
                candidate_fp32_post_bf16=float(candidate_post32.to(post.dtype)),
                reconstructed_bf16_autograd_post=float(bf16_expected_post),
                actual_vs_reference_fp32_relative_rms=relative_rms(actual[2],expected_post32.to(post.dtype)),
                actual_vs_candidate_fp32_relative_rms=relative_rms(actual[2],candidate_post32.to(post.dtype)),
                expert_output_relative_rms=relative_rms(candidate_values,ref_values),
                absolute_product_sum=float((gradient.double()*ref_values.double()*ref_gate.double()[:,None]).abs().sum()))
            print(json.dumps(dict(tokens=tokens,relative_rms=errors,post_diagnostic=diagnostics)),flush=True)
            checks.append(dict(tokens=tokens, gs32=gs32, active_experts=int((counts > 0).sum()),
                               relative_rms=errors, post_diagnostic=diagnostics, unrelated_gradient_bank_unchanged=True))
    result = dict(event='kernel_correctness', passed=False, diagnostic_only=True, features=features, checks=checks,
                  cache_checks=caches, routed_training_projection_gs32=True,
                  routed_validation_projection_gs32=True, routed_width_changed_only=True,
                  geometry_coefficient_gs32=True, shared_backend_modified=False, validation=False,
                  oracle='Diagnostic preserves failed reference and tolerance, records FP64 and FP32 analytic post derivatives plus BF16 autograd reconstruction. This does not pass correctness or authorize training.')
    with open(args.out, 'w') as out:
        out.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
