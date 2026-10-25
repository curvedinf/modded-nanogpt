"""Independent full dense KS3 rank oracle: pinned forward and Torch STE.

The reference never calls the candidate projection/geometry/backward kernels.
GS32 encoding uses Torch; the nonlinear forward uses the byte-identical pinned
upstream kernel. Differentiation uses independent smooth equations and the
specified BF16 coordinate-gradient storage boundary.
"""
import argparse
import json
import torch
import torch.nn.functional as F
import triton as tr

from wr_reproduction import ks3_moe as core
from wr_reproduction.ks3_feedforward import DenseKS3
from wr_reproduction.check_ks3_rank_geometry import surrogate
from ks3_research.vendor.ks3_int8 import pack_spine, _geometry_quant


def quantized_value(value):
    grouped = value.detach().float().reshape(value.shape[0], -1, 32)
    scale = (grouped.abs().amax(-1, keepdim=True) / 127).clamp(2**-24, 65504).half().float()
    return ((grouped / scale).round().clamp(-127, 127) * scale).reshape(value.shape).bfloat16().float()


def quantized_ste(value):
    real = value.float()
    return real + (quantized_value(value) - real).detach()


def reference(rows, up, down, geometry, rank):
    tokens, features = rows.shape[0], up.shape[0]
    x = quantized_ste(rows)
    u = quantized_ste(up)
    w = quantized_ste(down.T)
    coords = x @ u.T
    # The candidate stores d(coordinates) in BF16 before both projection VJPs.
    coords.register_hook(lambda grad: grad.bfloat16().float())
    with torch.no_grad():
        detached = coords.detach().contiguous()
        coefficients = geometry[:, :6*rank].reshape(features, 6).contiguous()
        packed, scales = pack_spine(coefficients)
        means = detached.clamp(-65504, 65504).reshape(tokens, features//rank, rank).mean(-1)
        warp = F.softplus(geometry[:, 6*rank]) + 1e-4
        shift = .5 * geometry[:, 6*rank+1].tanh()
        codes = torch.empty_like(detached, dtype=torch.int8)
        hs = torch.empty(tokens, tr.cdiv(features, 32), device=rows.device, dtype=torch.float16)
        _geometry_quant[(tr.cdiv(tokens, 16), tr.cdiv(features, 32))](
            detached, means, packed, scales, warp, shift, None, codes, hs,
            tokens, features, rank, 1, BM=16, BN=32, num_warps=4)
        actual_value = (codes.float() * hs.float().repeat_interleave(32, -1)).bfloat16().float()
    smooth = surrogate(coords, geometry, rank)
    h = smooth + (actual_value - smooth).detach()
    return (h @ w.T).bfloat16()


def relative_rms(actual, expected):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    return ((actual.float()-expected.float()).square().mean().sqrt()
            / expected.float().square().mean().sqrt().clamp_min(1e-12)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--features', type=int, choices=(1024,), default=1024)
    parser.add_argument('--rank', type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(172)
    core.BF16_TUNED_KERNELS = True
    core.KS3_RANK = args.rank
    checks = []
    for tokens in (1, 257, 4096):
        n, d, rank = args.features, 768, args.rank
        x = torch.randn(tokens, d, device='cuda', dtype=torch.bfloat16).requires_grad_()
        up = (torch.randn(n, d, device='cuda') * .025).bfloat16().requires_grad_()
        down = (torch.randn(n, d, device='cuda') * .02).bfloat16().requires_grad_()
        geo = torch.randn(n//rank, 6*rank+2, device='cuda') * .01
        geo[:, 6*rank] = .75
        geo.requires_grad_()
        post = torch.tensor(.7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        packed_up, packed_down = core.pack(up.detach()), core.pack(down.detach().T)
        output = DenseKS3.apply(x, up, down, packed_up, packed_down, geo, 0) * post
        gradient = torch.randn_like(output) * .02
        actual = torch.autograd.grad(output, (x, up, down, geo, post), gradient)
        xr, ur, dr, gr, pr = [v.detach().clone().requires_grad_() for v in (x, up, down, geo, post)]
        expected_output = reference(xr, ur, dr, gr, rank) * pr
        expected = torch.autograd.grad(expected_output, (xr, ur, dr, gr, pr), gradient)
        errors = dict(output=relative_rms(output, expected_output))
        for name, a, b in zip(('input_gradient', 'up_gradient', 'down_gradient', 'geometry_gradient', 'post_gradient'), actual, expected):
            errors[name] = relative_rms(a, b)
        assert max(errors.values()) < .015, (tokens, errors)
        checks.append(dict(tokens=tokens, relative_rms=errors))
    result = dict(event='kernel_correctness', passed=True, rank=args.rank, features=args.features,
                  checks=checks, oracle='Pinned upstream nonlinear forward, independent Torch GS32 and matrix products, independent smooth STE autograd with specified BF16 gradient storage',
                  shared_backend_modified=False, validation=False)
    with open(args.out, 'w') as out:
        out.write(json.dumps(result) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
