"""Dense direct-BF16 oracle using independent products and pinned nonlinearities.

Projection inputs/weights/hidden values omit GS32 rounding only in training.
Geometry coefficients and FP16 nonlinearities retain the pinned definition.
Validation refresh is checked against independent Torch GS32 decoding.
"""
import argparse
import json
import torch
import triton as tr

from wr_reproduction import ks3_moe as core
from wr_reproduction.ks3_feedforward import DenseKS3, install as install_dense
from wr_reproduction.check_ks3_rank_geometry import surrogate
from wr_reproduction.check_ks3_feedforward_rank import quantized_value, relative_rms
from wr_reproduction.check_ks3_fp8_moe import _raw_upstream_spine
from ks3_research.vendor.ks3_int8 import pack_spine


def reference(rows, up, down, geometry):
    tokens, features = rows.shape[0], up.shape[0]
    coords = rows.float() @ up.float().T
    coords.register_hook(lambda grad: grad.bfloat16().float())
    with torch.no_grad():
        codes, scales = pack_spine(geometry[:, :6].contiguous())
        warp = torch.nn.functional.softplus(geometry[:, 6]) + 1e-4
        shift = .5 * geometry[:, 7].tanh()
        hidden = torch.empty_like(coords)
        _raw_upstream_spine[(tr.cdiv(tokens, 16), tr.cdiv(features, 32))](
            coords.detach(), codes, scales, warp, shift, hidden, tokens, features, num_warps=4)
        hidden = hidden.bfloat16().float()
    smooth = surrogate(coords, geometry, 1)
    value = smooth + (hidden - smooth).detach()
    return (value @ down.float().T).bfloat16()


def model(features):
    from wr_reproduction.backend import install
    install('pr393')
    recipe = dict(precision='bf16', architecture='ks3_all', rank=1, charts=features,
                  dense_training_gs32=False,
                  geometry=dict(lr_mul=.125, coefficients=[0.]*6, raw_scale=.75, raw_shift=0.))
    cls = install_dense(torch.nn.Module, recipe)
    value = torch.nn.Module()
    value.mlp_bank = torch.nn.Parameter((torch.randn(8, 2, features, 768, device='cuda')*.025).bfloat16())
    value.register_buffer('_mlp_ks3_up', torch.empty(5, features, 768, device='cuda', dtype=torch.bfloat16))
    value.register_buffer('_mlp_ks3_down', torch.empty(5, 768, features, device='cuda', dtype=torch.bfloat16))
    return value, cls._refresh_dense_ks3


def check_cache(value, direct):
    for slot in range(5):
        for weights, cached in ((value.mlp_bank[slot, 0], value._mlp_ks3_up[slot]),
                                (value.mlp_bank[slot, 1].T, value._mlp_ks3_down[slot])):
            expected = weights if direct else quantized_value(weights).bfloat16()
            assert torch.equal(cached, expected)


def cache_checks(value, refresh):
    pointers = {name: x.data_ptr() for name, x in value.named_buffers()}
    checks = []
    with torch.no_grad():
        for training in (True, False, True):
            value.train(training)
            refresh(value)
            check_cache(value, training)
            checks.append(dict(training=training, cache_bytes_exact=True))
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            refresh(value)
        for _ in range(2):
            value.mlp_bank.add_(torch.randn_like(value.mlp_bank)*.001)
            graph.replay()
            check_cache(value, True)
            checks.append(dict(captured_replay=True, cache_bytes_exact=True))
    assert pointers == {name: x.data_ptr() for name, x in value.named_buffers()}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--features', type=int, choices=(1024,), default=1024)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(172)
    core.BF16_TUNED_KERNELS = True
    core.KS3_RANK = 1
    value, refresh = model(args.features)
    caches = cache_checks(value, refresh)
    checks = []
    for tokens in (1, 257, 4096):
        features = args.features
        x = torch.randn(tokens, 768, device='cuda', dtype=torch.bfloat16).requires_grad_()
        up = value.mlp_bank[0, 0].detach().clone().requires_grad_()
        down = value.mlp_bank[0, 1].detach().T.contiguous().requires_grad_()
        geometry = torch.randn(features, 8, device='cuda')*.01
        geometry[:, 6] = .75
        geometry.requires_grad_()
        post = torch.tensor(.7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        output = DenseKS3.apply(x, up, down.T, value._mlp_ks3_up[0],
                               value._mlp_ks3_down[0], geometry, 0, False) * post
        gradient = torch.randn_like(output)*.02
        actual = torch.autograd.grad(output, (x, up, down, geometry, post), gradient)
        xr, ur, dr, gr, pr = [v.detach().clone().requires_grad_() for v in (x, up, down, geometry, post)]
        expected_output = reference(xr, ur, dr, gr) * pr
        expected = torch.autograd.grad(expected_output, (xr, ur, dr, gr, pr), gradient)
        errors = dict(output=relative_rms(output, expected_output))
        for name, a, b in zip(('input_gradient', 'up_gradient', 'down_gradient', 'geometry_gradient', 'post_gradient'), actual, expected):
            errors[name] = relative_rms(a, b)
        assert max(errors.values()) < .015, (tokens, errors)
        # Accumulating two invocations must add into the original bank views.
        repeated = DenseKS3.apply(x, up, down.T, value._mlp_ks3_up[0],
                                 value._mlp_ks3_down[0], geometry, 0, False)
        second = DenseKS3.apply(x, up, down.T, value._mlp_ks3_up[0],
                               value._mlp_ks3_down[0], geometry, 0, False)
        total = torch.autograd.grad(repeated + second, (x, up, down, geometry), gradient)
        single = DenseKS3.apply(x, up, down.T, value._mlp_ks3_up[0],
                               value._mlp_ks3_down[0], geometry, 0, False)
        once = torch.autograd.grad(single, (x, up, down, geometry), gradient)
        for a, b in zip(total, once):
            torch.testing.assert_close(a.float(), b.float()*2, atol=1e-5, rtol=.015)
        checks.append(dict(tokens=tokens, relative_rms=errors, additive_backward=True))
    result = dict(event='kernel_correctness', passed=True, features=args.features, checks=checks,
                  cache_checks=caches, dense_training_projection_gs32=False,
                  dense_validation_projection_gs32=True, geometry_coefficient_gs32=True,
                  oracle='Independent FP32 products of BF16 operands; pinned FP16 nonlinear forward and independent smooth STE with BF16 coordinate gradients',
                  shared_backend_modified=False, validation=False)
    with open(args.out, 'w') as out:
        out.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
