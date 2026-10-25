"""Independent dense FP8 KS3 arithmetic, autograd, and cache-replay oracle.

No timing sweep. References use Torch encoding/decoded matrix products, the
pinned upstream nonlinear forward, and independent smooth STE autograd.
"""
import argparse
import json
import torch
import triton as tr
from wr_reproduction.backend import install


def gs32(value):
    shape = value.shape
    grouped = torch.nn.functional.pad(value.detach().float(), (0, (-shape[-1]) % 32)).reshape(shape[0], -1, 32)
    scale = (grouped.abs().amax(-1, keepdim=True) / 127).clamp(2**-24, 65504).half().float()
    decoded = (grouped / scale).round().clamp(-127, 127) * scale
    return decoded.reshape(shape[0], -1)[:, :shape[-1]]


def encode(value, scale, dtype):
    limit = torch.finfo(dtype).max
    return (value / scale).clamp(-limit, limit).to(dtype)


def raw_ste(value, scale, training_gs32):
    raw = value.float() / scale
    decoded = gs32(value) if training_gs32 else value.detach().float()
    codes = encode(decoded, scale, torch.float8_e4m3fn)
    return raw + (codes.float() - raw).detach()


def reference(x, up, down, geometry, us, ws, xs, gs, ps, ds, training_gs32):
    from ks3_research.vendor.ks3_int8 import pack_spine, _geometry_quant
    from wr_reproduction.check_ks3_rank_geometry import surrogate
    tokens, features = x.shape[0], up.shape[0]
    xr, ur, wr = (raw_ste(v, s, training_gs32) for v, s in ((x, xs), (up, us), (down.T, ws)))
    coords = (xr @ ur.T) * (xs * us)
    coords.register_hook(lambda grad: encode(grad, ds, torch.float8_e5m2).float() * ds)
    with torch.no_grad():
        p, pscale = pack_spine(geometry[:, :6].contiguous())
        scale = torch.nn.functional.softplus(geometry[:, 6]) + 1e-4
        shift = .5 * geometry[:, 7].tanh()
        if training_gs32:
            codes = torch.empty_like(coords, dtype=torch.int8)
            scales = torch.empty(tokens, features // 32, device=x.device, dtype=torch.float16)
            _geometry_quant[(tr.cdiv(tokens, 16), tr.cdiv(features, 32))](
                coords.detach(), coords.detach(), p, pscale, scale, shift, None, codes, scales,
                tokens, features, 1, 1, BM=16, BN=32, num_warps=4)
            hidden = codes.float() * scales.float().repeat_interleave(32, -1)
        else:
            from wr_reproduction.check_ks3_fp8_moe import _raw_upstream_spine
            hidden = torch.empty_like(coords, dtype=torch.float32)
            _raw_upstream_spine[(tr.cdiv(tokens, 16), tr.cdiv(features, 32))](
                coords.detach(), p, pscale, scale, shift, hidden, tokens, features, num_warps=4)
        encoded = encode(hidden, ps, torch.float8_e4m3fn)
    smooth = surrogate(coords, geometry, 1) / ps
    hidden = smooth + (encoded.float() - smooth).detach()
    y = (hidden @ wr.T) * (ps * ws)
    y.register_hook(lambda grad: encode(grad, gs, torch.float8_e5m2).float() * gs)
    return y.bfloat16()


def relative_rms(actual, expected):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    return ((actual.float() - expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-12)).item()


def model(features, training_gs32=True):
    from wr_reproduction.ks3_dense_fp8 import init_buffers
    from track_1_short.model.gpt import FP8_MLP_X_SCALE, FP8_GRAD_SCALE
    m = torch.nn.Module()
    m.device = torch.device('cuda')
    m.retained_slots = ()
    m.mlp_bank = torch.nn.Parameter((torch.randn(8, 2, features, 768, device='cuda') * .025).bfloat16())
    m.ks3_dense_geometry = torch.nn.Parameter(torch.randn(5, features, 8, device='cuda') * .01)
    with torch.no_grad():
        m.ks3_dense_geometry[..., 6] = .75
    m.register_buffer('_mlp_x_scale', torch.tensor(FP8_MLP_X_SCALE, device='cuda'))
    m.register_buffer('_mlp_grad_scale', torch.tensor(FP8_GRAD_SCALE, device='cuda'))
    init_buffers(m, features, training_gs32=training_gs32)
    return m


@torch.no_grad()
def compare_cache(m, previous, exact):
    from wr_reproduction.check_ks3_prepared_native import native_copy
    for slot in range(5):
        for projection, suffix in ((0, 'up'), (1, 'down')):
            value = m.mlp_bank[slot, projection]
            if projection:
                value = value.T
            largest = value.float().abs().amax() if exact else previous[slot, projection]
            scale = largest.clamp_min(1e-12) / 448.
            decoded = gs32(value) if m.ks3_dense_training_gs32 else value.detach().float()
            expected = encode(decoded, scale, torch.float8_e4m3fn)
            torch.testing.assert_close(m.ks3_dense_weight_scales[slot, projection], scale.reshape(1), rtol=0, atol=0)
            torch.testing.assert_close(m.ks3_dense_weight_amax[slot, projection], decoded.abs().amax().reshape(1), rtol=0, atol=0)
            for prefix in ('f8', 'native'):
                wanted = native_copy(expected) if prefix == 'native' else expected
                for tail, result in [('', wanted), ('_t', wanted.T.contiguous())]:
                    got = getattr(m, 'ks3_dense_' + prefix + '_' + suffix + tail)[slot, 0]
                    assert torch.equal(got.view(torch.uint8), result.view(torch.uint8))


def cache_checks(m):
    from wr_reproduction.ks3_dense_fp8 import refresh_weights
    pointers = {name: value.data_ptr() for name, value in m.named_buffers()}
    checks = []
    for index, exact in enumerate((True, False, False, True)):
        if index:
            with torch.no_grad():
                m.mlp_bank.add_(torch.randn_like(m.mlp_bank) * .001)
        previous = m.ks3_dense_weight_amax.clone()
        refresh_weights(m, exact)
        compare_cache(m, previous, exact)
        checks.append(dict(exact=exact, original_and_native_bytes_exact=True, delayed_scale_exact=True))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refresh_weights(m, False)
    for _ in range(2):
        with torch.no_grad():
            m.mlp_bank.add_(torch.randn_like(m.mlp_bank) * .001)
        previous = m.ks3_dense_weight_amax.clone()
        graph.replay()
        compare_cache(m, previous, False)
        checks.append(dict(captured_replay=True, original_and_native_bytes_exact=True, delayed_scale_exact=True))
    assert pointers == {name: value.data_ptr() for name, value in m.named_buffers()}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--features', type=int, choices=(1024, 2816), required=True)
    parser.add_argument('--dense-native-pack', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(172)
    install('pr393')
    from wr_reproduction import ks3_fp8_gemm
    from wr_reproduction.ks3_dense_fp8 import DenseFP8, refresh_weights
    ks3_fp8_gemm.TILE_MODE = 'mi300-v2'
    training_gs32 = not args.dense_native_pack
    m = model(args.features, training_gs32)
    caches_checked = cache_checks(m)
    refresh_weights(m, True)
    # These representative post/coordinate-gradient scales exercise nonzero
    # geometry without clipping most products. Bootstrap and rank-local
    # refresh use the immutable model constants and are checked separately.
    m._mlp_post_scales.fill_(.01)
    m._mlp_dpre_scales.fill_(.0001)
    checks = []
    for tokens in (1, 257, 4096):
        x = torch.randn(tokens, 768, device='cuda', dtype=torch.bfloat16).requires_grad_()
        up = m.mlp_bank[0, 0].detach().requires_grad_()
        down = m.mlp_bank[0, 1].detach().requires_grad_()
        geometry = m.ks3_dense_geometry[0].detach().requires_grad_()
        post = torch.tensor(.7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        caches = tuple(getattr(m, 'ks3_dense_native_' + name)[0] for name in ('up', 'up_t', 'down', 'down_t'))
        us, ws = m.ks3_dense_weight_scales[0]
        xs, gs = m._mlp_x_scale, m._mlp_grad_scale
        ps, ds = m._mlp_post_scales[:1], m._mlp_dpre_scales[:1]
        scales = (us, ws, xs, gs, ps, m._mlp_post_amax[0], ds, m._mlp_dpre_amax[0])
        y = DenseFP8.apply(x, up, down, geometry, caches, scales, 0, training_gs32) * post
        gradient = torch.randn_like(y) * .02
        actual = torch.autograd.grad(y, (x, up, down, geometry, post), gradient)
        xr, ur, dr, gr, pr = [value.detach().clone().requires_grad_() for value in (x, up, down, geometry, post)]
        yr = reference(xr, ur, dr, gr, us, ws, xs, gs, ps, ds, training_gs32) * pr
        expected = torch.autograd.grad(yr, (xr, ur, dr, gr, pr), gradient)
        errors = dict(output=relative_rms(y, yr))
        for name, a, b in zip(('input_gradient', 'up_gradient', 'down_gradient', 'geometry_gradient', 'post_gradient'), actual, expected):
            errors[name] = relative_rms(a, b)
        assert max(errors.values()) < .015, (tokens, errors)
        # A fresh autograd invocation must overwrite its temporary projection
        # bank, while normal PyTorch leaf accumulation remains additive.
        leaves = (x, up, down, geometry, post)
        for leaf in leaves:
            leaf.grad = None
        for _ in range(2):
            (DenseFP8.apply(x, up, down, geometry, caches, scales, 0, training_gs32) * post).backward(gradient)
        for leaf, expected_grad in zip(leaves, actual):
            torch.testing.assert_close(leaf.grad, expected_grad * 2, rtol=0, atol=0)
        checks.append(dict(tokens=tokens, relative_rms=errors, repeated_backward_additive=True))
    # Two live sites share rank-local amax base tensors and the original
    # projection bank. This catches illegal autograd saved-buffer versions
    # and verifies that ordinary bank accumulation leaves padding untouched.
    from wr_reproduction.ks3_dense_fp8 import apply
    x = torch.randn(257, 768, device='cuda', dtype=torch.bfloat16).requires_grad_()
    m.mlp_bank.grad = None
    m.ks3_dense_geometry.grad = None
    outputs = [apply(m, slot, slot, x, m.mlp_bank[slot, 0], m.mlp_bank[slot, 1]) for slot in (0, 1)]
    (outputs[0] + outputs[1]).backward(torch.randn_like(outputs[0]) * .02)
    assert torch.isfinite(x.grad).all()
    for value in (m.mlp_bank.grad, m.ks3_dense_geometry.grad):
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value[0]) and torch.count_nonzero(value[1])
        assert not torch.count_nonzero(value[2:])
    result = dict(event='kernel_correctness', passed=True, features=args.features,
                  multiple_live_sites_and_padded_bank_verified=True,
                  dense_precision='fp8', routed_precision='bf16', checks=checks, cache_checks=caches_checked,
                  dense_training_projection_gs32=training_gs32, geometry_coefficient_gs32=True,
                  oracle='Independent Torch GS32/FP8 encoding and decoded products, pinned nonlinear forward, smooth STE with explicit FP8 gradient boundaries',
                  shared_backend_modified=False, validation=False, timing=False)
    with open(args.out, 'w') as out:
        out.write(json.dumps(result) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
