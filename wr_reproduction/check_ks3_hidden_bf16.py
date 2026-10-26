"""Independent FP16-stage forward and smooth-STE checks for hidden precision."""
from unittest.mock import patch
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_hidden_bf16 as hidden
from wr_reproduction.check_ks3_pairwise import error, matrix
from wr_reproduction.check_ks3_split_coefficient_scale import matrix_reference


def coefficients(g, ste=False):
    c = g[..., :6]
    scale = (c.abs().amax(-1, keepdim=True)/127).clamp(2**-24, 65504).half().float()
    packed = (c/scale).round().clamp(-127, 127)*scale
    if ste:
        packed = c+(packed-c).detach()
    return torch.cat((packed, g[..., 6:]), -1)


def reference_h(c, g, offsets):
    rounded = lambda x: x.half().float()
    out = torch.empty_like(c, dtype=torch.bfloat16)
    ends = list(offsets); ends[-1] = c.shape[0]; lo = 0
    for e, hi in enumerate(ends):
        if hi == lo:
            continue
        cc = g[e, :, :6]
        ps = rounded((cc.abs().amax(-1)/127).clamp(2**-24, 65504))
        c0, c1, c2, c3, c4, c5 = (cc/ps[:, None]).round().clamp(-127, 127).unbind(-1)
        x = c[lo:hi].clamp(-65504, 65504)
        t = rounded(4*torch.tanh(rounded(x*.25)))
        scale = torch.nn.functional.softplus(g[e, :, 6])+1e-4
        shift = .5*torch.tanh(g[e, :, 7])
        z = rounded(torch.tanh(rounded(scale*(t+shift))))
        a = 2*c0-2*c1+c2+c3; b = -3*c0+3*c1-2*c2-c3
        poly = rounded(rounded(a)*t+rounded(b))
        poly = rounded(poly*t+rounded(c2)); poly = rounded(poly*t+rounded(c0))
        warp = rounded(4*rounded(c5)*z+2*rounded(c4))
        warp = rounded(warp*z-3*rounded(c5)); warp = rounded(warp*z-rounded(c4))
        h = rounded(x+(poly+warp)*ps).clamp_min(0.)
        out[lo:hi] = rounded(h*h); lo = hi
    return out


def reference_vjp(c, g, v, src, tokens, offsets):
    cc = c.detach().clone().requires_grad_(); gg = g.detach().clone().requires_grad_()
    packed = coefficients(gg, ste=True); terms = []; lo = 0
    for e, hi in enumerate(offsets):
        if hi == lo:
            continue
        x = cc[lo:hi].clamp(-65504, 65504)
        c0, c1, c2, c3, c4, c5, rs, rh = packed[e].unbind(-1)
        t = 4*torch.tanh(x*.25)
        z = torch.tanh((torch.nn.functional.softplus(rs)+1e-4)*(t+.5*torch.tanh(rh)))
        a = 2*c0-2*c1+c2+c3; b = -3*c0+3*c1-2*c2-c3
        spine = ((a*t+b)*t+c2)*t+c0
        warp = ((4*c5*z+2*c4)*z-3*c5)*z-c4
        h = (x+spine+warp).clamp_min(0.).square()
        terms.append((h*torch.where((src[lo:hi]<tokens)[:, None], v[lo:hi], 0.)).sum())
        lo = hi
    return torch.autograd.grad(sum(terms), (cc, gg))


def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(1769); torch.set_num_threads(1); core.KS3_RANK = 1
    e, n, d = 8, 1024, 768; checks = []
    up = core.pack((torch.randn(e*n, d, device='cuda')*.025).bfloat16()).view(e, n, d)
    down = torch.stack([core.pack((torch.randn(d, n, device='cuda')*.025).bfloat16()) for _ in range(e)])
    original_backward = core.backward
    for tuned, tokens in ((True, 1), (True, 257), (True, 16384), (True, 49152), (False, 257)):
        core.BF16_TUNED_KERNELS = tuned
        x = torch.randn(tokens, d, device='cuda', dtype=torch.bfloat16)
        grad = torch.randn_like(x)*.02; gate = torch.rand(tokens, device='cuda')
        assignments = torch.zeros(tokens, device='cuda', dtype=torch.int64) if tokens==257 else torch.randint(e, (tokens,), device='cuda')
        dest, src, offs, tiles, counts = moe.moe_dispatch(assignments, e); offsets = offs.cpu().tolist()
        bank = torch.zeros(16, n, d, device='cuda', dtype=torch.bfloat16)
        for zero in (True, False):
            g = torch.randn(e, n, 8, device='cuda')*.015
            g[..., 6] = .75; g[..., 7] = 0.
            if zero:
                g[..., :6].zero_()
            original = hidden.ORIGINAL_FORWARD(x, src, tiles, up, down, g)
            actual = hidden.forward(x, src, tiles, up, down, g)
            assert torch.equal(actual[1], original[1]) and torch.equal(actual[2], original[2])
            got = core.backward(grad, *actual, gate, src, offs, tiles, up, down, g, bank, 0)
            got_bank = bank.clone(); y, xp, c, h = actual
            if not tuned:
                old = core.backward(grad, *original, gate, src, offs, tiles, up, down, g, bank, 0)
                assert all(torch.equal(a, b) for a, b in zip(actual, original))
                assert all(torch.equal(a, b) for a, b in zip(got, old)) and torch.equal(got_bank, bank)
                row = dict(tuned=False, tokens=tokens, zero_coefficients=zero,
                    original_eval_output_input_geometry_gate_weight_gradients_bitwise=True)
            else:
                href = reference_h(c, g, offsets)
                yr = matrix(href, down.transpose(1, 2), tiles, x.dtype)
                valid = (src<tokens)[:, None]; idx = src.clamp(0, tokens-1).long()
                gathered = grad[idx].float()
                gg = torch.where(valid, gathered*gate[idx, None].bfloat16().float(), 0.).bfloat16()
                # Smooth STE is checked independently at the actual downstream cotangent.
                v = core.mm(gg, down, tiles)
                with torch.enable_grad():
                    dc_ref, dg_ref = reference_vjp(c, g, v, src, tokens, offsets)
                dc = torch.empty_like(c); part = torch.empty(tr.cdiv(c.shape[0], 64), n, 8, device='cuda')
                core._vjp[(tr.cdiv(c.shape[0], 64), tr.cdiv(n, 32))](
                    v, c, g, dc, part, tiles, src, *c.shape, tokens, RANK=1, num_warps=4)
                dg = torch.empty_like(g)
                core._reduce[(e, n)](part, dg, offs, n, part.shape[0], tr.next_power_of_2(part.shape[0]), RANK=1, num_warps=4)
                smooth = dict(coordinate=error(dc, dc_ref), geometry=error(dg, dg_ref),
                    components=[error(dg[..., i], dg_ref[..., i]) for i in range(8)])
                assert max(smooth['coordinate'], smooth['geometry'], *smooth['components']) < 3e-5, smooth
                dxr, dwr = matrix_reference(dc_ref, href, xp, gg, up, offsets)
                errs = dict(features=error(h, href), output=error(y, yr), input_gradient=error(got[0], dxr),
                    geometry_gradient=error(got[2], dg_ref), weight_gradient=error(got_bank, dwr))
                assert errs['features'] < .002 and errs['output'] < .003, errs
                assert errs['input_gradient'] < .003 and errs['weight_gradient'] < .003, errs
                assert errs['geometry_gradient'] < 3e-5, errs
                assert torch.equal(got[1], torch.where(valid, gathered*y.float(), 0.).sum(1))
                row = dict(tuned=True, tokens=tokens, zero_coefficients=zero, active_experts=int((counts>0).sum()),
                    padded_rows=src.numel(), input_pack_and_coordinates_bitwise_unchanged=True,
                    independent_relative_rms=errs, independent_smooth_ste_relative_rms=smooth,
                    hidden_changed_fraction=(h!=original[3]).float().mean().item(),
                    original_initial_training_output_equivalence_expected=False)
            # Actual routed autograd must use this forward and the unchanged backward.
            xx = x.clone().requires_grad_(); router = (torch.randn(e, d, device='cuda')*.02).bfloat16().requires_grad_()
            post = torch.tensor(.8, device='cuda', requires_grad=True); geometry = g.clone().requires_grad_()
            with patch.object(core, 'forward', hidden.forward):
                out, probs, _ = core.RoutedKS3.apply(xx, router, post, up, down, up, down, geometry, bank, 0, True)
                gradients = torch.autograd.grad(out, (xx, router, post, geometry), grad)
                assert all(torch.isfinite(v).all() for v in (*gradients, bank, out))
                if not tuned:
                    saved = [v.clone() for v in (out, *gradients, bank)]
            if not tuned:
                with patch.object(core, 'forward', hidden.ORIGINAL_FORWARD):
                    old_out, _, _ = core.RoutedKS3.apply(xx, router, post, up, down, up, down, geometry, bank, 0, True)
                    old_grads = torch.autograd.grad(old_out, (xx, router, post, geometry), grad)
                assert all(torch.equal(a, b) for a, b in zip(saved, (old_out, *old_grads, bank)))
                row['original_eval_full_routed_gradients_bitwise'] = True
            row['routed_training_executed'] = tuned
            checks.append(row); print(row, flush=True)
    hidden.install(); hidden.install()
    assert core.forward is hidden.forward and core.backward is original_backward
    assert hidden.EXECUTED_MODES == {'training_hidden_bf16', 'original_evaluation'}
    return dict(passed=True, numerical_seed=1769, checks=checks, new_parameters=0,
        original_input_weight_coefficient_packing=True, original_FP16_nonlinearity=True,
        original_evaluation_bitwise=True, original_smooth_ste_unchanged=True, install_idempotent=True,
        initial_training_output_equivalence_expected=False,
        oracle='Independent Torch joint coefficient packing, original FP16 rounding stages without hidden GS32, smooth FP32 STE autograd and grouped FP32 matrix products with BF16 storage.',
        tolerance_scope='FP16/BLAS reduction and BF16 storage included in forward/input/weight bounds; smooth gradients below3e-5 relativeRMS. All evaluation values and gradients require bitwise equality.',
        actual_model_training_tested=False, full_convergence_verified=False)
