"""Independent FP32-Horner forward/original-STE and actual-width block checks."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_fp32_horner as split


@contextmanager
def kernels(enabled):
    with patch.object(core, 'forward', split.forward if enabled else split.ORIGINAL_FORWARD):
        yield


def relative(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))


def rounded(x):
    return x.half().float()


def cpu_check():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2769)
        g=torch.randn(5,17,8)*.02;g[...,6]=.75
        raw=g[...,:6];scale=(raw.abs().amax(-1,keepdim=True)/127).clamp(2**-24,65504).half().float()
        expected=(raw/scale).round().clamp(-127,127)*scale
        assert torch.equal(split.reference_coefficients(g)[...,:6],expected)
        differentiable=g.clone().requires_grad_()
        grad=torch.autograd.grad(split.reference_coefficients(differentiable,ste=True).sum(),differentiable)[0]
        assert torch.equal(grad,torch.ones_like(grad))
        assert torch.equal(split.reference_coefficients(g)[...,6:],g[...,6:])
        g[...,:6]=0
        assert torch.equal(split.reference_coefficients(g),g)
        empty_tail_g=torch.zeros(2,32,8);empty_tail_g[...,6]=.75
        assert torch.equal(reference_h(torch.ones(384,32),empty_tail_g,[128,128],False),torch.ones(384,32,dtype=torch.bfloat16))
        x=torch.linspace(-8,8,384)[:,None].expand(384,32)
        assert torch.equal(reference_h(x,empty_tail_g,[128,128],fp32_horner=True),reference_h(x,empty_tail_g,[128,128],fp32_horner=False))
        dx,_=matrix_reference(torch.ones(384,32),torch.ones(384,32),torch.ones(384,4),torch.ones(384,4),torch.ones(2,32,4),[128,128])
        assert torch.count_nonzero(dx[128:])==0
    return dict(passed=True,original_joint_coefficient_GS32=True,original_STE_identity=True,raw_warp_scalars_unchanged=True,zero_initialization_exact=True,unused_dispatch_capacity_oracles_initialized=True)


def reference_h(c, g, offsets, gs32=True, fp32_horner=True):
    """Original joint GS32 coefficients and FP16 boundaries; independent Horner."""
    out = torch.empty_like(c, dtype=torch.bfloat16); lo = 0
    # Original dispatch clamps every extra-capacity tile to the last expert.
    # Forward computes these unused rows; backward masks them using SRC.
    forward_ends = list(offsets); forward_ends[-1] = c.shape[0]
    for e, hi in enumerate(forward_ends):
        if hi == lo: continue
        gg = g[e]; coeff = gg[:, :6]
        ps=rounded((coeff.abs().amax(-1)/127).clamp(2**-24,65504))
        c0,c1,c2,c3,c4,c5=(coeff/ps[:,None]).round().clamp(-127,127).unbind(-1)
        x = c[lo:hi].clamp(-65504, 65504)
        t = rounded(4*torch.tanh(rounded(x*.25)))
        scale = torch.nn.functional.softplus(gg[:, 6])+1e-4
        shift = .5*torch.tanh(gg[:, 7])
        z = rounded(torch.tanh(rounded(scale*(t+shift))))
        a = 2*c0-2*c1+c2+c3; b = -3*c0+3*c1-2*c2-c3
        if fp32_horner:
            poly=((a*t+b)*t+c2)*t+c0
            warp=((4*c5*z+2*c4)*z-3*c5)*z-c4
        else:
            poly=rounded(rounded(a)*t+rounded(b));poly=rounded(poly*t+rounded(c2));poly=rounded(poly*t+rounded(c0))
            warp=rounded(4*rounded(c5)*z+2*rounded(c4));warp=rounded(warp*z-3*rounded(c5));warp=rounded(warp*z-rounded(c4))
        h=rounded(x+(poly+warp)*ps).clamp_min(0.)
        h = rounded(h*h)
        if gs32:
            groups = h.reshape(h.shape[0], -1, 32)
            hs = rounded((groups.abs().amax(-1, keepdim=True)/127).clamp(2**-24, 65504))
            h = ((groups/hs).round().clamp(-127, 127)*hs).reshape_as(h)
        out[lo:hi] = h; lo = hi
    return out


def reference_vjp(c, g, v, src, token_count, offsets):
    """Autograd of original smooth FP32 geometry at original GS32 coefficients."""
    cc = c.detach().clone().requires_grad_(); gg = g.detach().clone().requires_grad_()
    packed = split.reference_coefficients(gg, ste=True)
    terms = []; lo = 0
    for e, hi in enumerate(offsets):
        if hi == lo: continue
        x = cc[lo:hi].clamp(-65504, 65504)
        c0, c1, c2, c3, c4, c5, rs, rh = packed[e].unbind(-1)
        t = 4*torch.tanh(x*.25)
        z = torch.tanh((torch.nn.functional.softplus(rs)+1e-4)*(t+.5*torch.tanh(rh)))
        cubic = 2*c0-2*c1+c2+c3; quadratic = -3*c0+3*c1-2*c2-c3
        spine = ((cubic*t+quadratic)*t+c2)*t+c0
        warp = ((4*c5*z+2*c4)*z-3*c5)*z-c4
        h = (x+spine+warp).clamp_min(0.).square()
        cotangent = torch.where((src[lo:hi]<token_count)[:, None], v[lo:hi], 0.)
        terms.append((h*cotangent).sum()); lo = hi
    return torch.autograd.grad(sum(terms), (cc, gg))


def matrix_reference(dc, h, x, gg, up, offsets):
    e, n, d = up.shape
    dx = torch.zeros((dc.shape[0], d), device=dc.device, dtype=torch.bfloat16)
    bank = torch.zeros(2*e, n, d, device=dc.device, dtype=torch.bfloat16)
    lo = 0
    for index, hi in enumerate(offsets):
        if hi != lo:
            dd = dc[lo:hi].bfloat16().float()
            dx[lo:hi] = dd@up[index].float()
            bank[index] = dd.T@x[lo:hi].float()
            bank[e+index] = h[lo:hi].float().T@gg[lo:hi].float()
        lo = hi
    return dx, bank


def check():
    cpu = cpu_check()
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(2769); torch.set_num_threads(1); core.KS3_RANK = 1
    e, n, d = 8, 1024, 768
    up = core.pack((torch.randn(e*n, d, device='cuda')*.025).bfloat16()).view(e, n, d)
    down = torch.stack([core.pack((torch.randn(d, n, device='cuda')*.025).bfloat16()) for _ in range(e)])
    checks = []
    for tuned, tokens in ((True, 1), (True, 257), (True, 4096), (True, 16384), (False, 257)):
        core.BF16_TUNED_KERNELS = tuned
        x = torch.randn(tokens, d, device='cuda', dtype=torch.bfloat16)
        grad = torch.randn_like(x)*.02; gate = torch.rand(tokens, device='cuda')
        assignments = torch.zeros(tokens, device='cuda', dtype=torch.int64) if tokens==257 else torch.randint(e, (tokens,), device='cuda')
        dest, src, offs, tiles, counts = moe.moe_dispatch(assignments, e); offsets = offs.cpu().tolist()
        bank = torch.zeros(16, n, d, device='cuda', dtype=torch.bfloat16)
        for zero in (True, False):
            g = torch.randn(e, n, 8, device='cuda')*.015
            g[..., 4:6] *= 2; g[..., 6] = .75; g[..., 7] = 0.
            if zero: g[..., :6].zero_()
            with kernels(True):
                actual = core.forward(x, src, tiles, up, down, g)
                got = core.backward(grad, *actual, gate, src, offs, tiles, up, down, g, bank, 0)
                got_bank = bank.clone()
            y, xp, c, h = actual
            href = reference_h(c, g, offsets, fp32_horner=tuned)
            h_error = relative(h, href); assert h_error < .002, ('FP32 Horner packed forward', tuned, tokens, zero, h_error)
            yref = core.mm(href, down.transpose(1, 2), tiles, dtype=x.dtype)
            y_error = relative(y, yref); assert y_error < .003, ('output', tokens, y_error)
            valid = (src<tokens)[:, None]; idx = src.clamp(0, tokens-1).long()
            gathered = grad[idx].float()
            gg = torch.where(valid, gathered*gate[idx, None].bfloat16().float(), 0.).bfloat16()
            v = core.mm(gg, down, tiles)
            with torch.enable_grad(): dc_ref, dg_ref = reference_vjp(c, g, v, src, tokens, offsets)
            dc = torch.empty_like(c); part = torch.empty(tr.cdiv(c.shape[0], 64), n, 8, device='cuda')
            split.ORIGINAL_VJP[(tr.cdiv(c.shape[0], 64), tr.cdiv(n, 32))](
                v, c, g, dc, part, tiles, src, *c.shape, tokens, num_warps=4)
            dg = torch.empty_like(g)
            core._reduce[(e, n)](part, dg, offs, n, part.shape[0], tr.next_power_of_2(part.shape[0]), num_warps=4)
            dc_error = relative(dc, dc_ref); dg_error = relative(dg, dg_ref)
            assert dc_error < 2e-5 and dg_error < 2e-5, ('independent smooth STE', tokens, dc_error, dg_error)
            component_errors = [relative(dg[..., i], dg_ref[..., i]) for i in range(8)]
            assert max(component_errors) < 3e-5, ('individual geometry gradients', tokens, component_errors)
            dx_ref, dw_ref = matrix_reference(dc_ref, href, xp, gg, up, offsets)
            dx_error = relative(got[0], dx_ref); dw_error = relative(got_bank, dw_ref)
            assert dx_error < .003 and dw_error < .003, ('input and weight gradients', tokens, dx_error, dw_error)
            assert relative(got[2], dg_ref) < 2e-5
            initial_exact = None
            if zero or not tuned:
                with kernels(False):
                    original = core.forward(x, src, tiles, up, down, g)
                    back = core.backward(grad, *original, gate, src, offs, tiles, up, down, g, bank, 0)
                initial_exact = all(torch.equal(a, b) for a, b in zip(actual, original)) and all(torch.equal(a, b) for a, b in zip(got, back)) and torch.equal(got_bank, bank)
                assert initial_exact, ('zero initialization or unchanged evaluation block parity', tuned, tokens)
                # Full routed wrapper also includes router and post gradients.
                xx = x.clone().requires_grad_(); router = (torch.randn(e, d, device='cuda')*.02).bfloat16().requires_grad_()
                post = torch.tensor(.8, device='cuda', requires_grad=True); geometry = g.clone().requires_grad_()
                def routed(enabled):
                    with kernels(enabled):
                        out, _, _ = core.RoutedKS3.apply(xx, router, post, up, down, up, down, geometry, bank, 0, True)
                        gradients = torch.autograd.grad(out, (xx, router, post, geometry), grad)
                        return out.detach(), [z.detach().clone() for z in gradients], bank.clone()
                a = routed(False); b = routed(True)
                assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
                assert all(torch.equal(q, z) for q, z in zip(a[1], b[1]))
            row = dict(tuned=tuned, tokens=tokens, zero_coefficients=zero, active_experts=int((counts>0).sum()),
                padded_rows=src.numel(), zero_initialization_or_evaluation_and_all_routed_gradients_bitwise_equal=initial_exact,
                forward_feature_relative_rms=h_error, output_relative_rms=y_error,
                independent_coordinate_gradient_relative_rms=dc_error, independent_geometry_gradient_relative_rms=dg_error,
                individual_geometry_gradient_relative_rms=component_errors,
                independent_input_gradient_relative_rms=dx_error, independent_weight_gradient_relative_rms=dw_error)
            checks.append(row); print(row, flush=True)
    return dict(passed=True, cpu_checks=cpu, numerical_seed=2769, checks=checks, independent_rank_one_components_retained=True,
        new_parameters=0, original_FP16_tanh_and_relu_boundaries=True, polynomial_accumulation_FP32_during_training=True, original_evaluation_bitwise=True, original_GS32_activation_and_projection_packing=True,
        actual_model_training_tested=False, full_convergence_verified=False,
        oracle='Independent Torch original joint GS32 coefficients, FP32 Horner with FP16 tanh/ReLU boundaries during training, original FP16-stage evaluation and original autograd smooth FP32 STE; original BF16 matrix products compared with grouped Torch FP32 reference and BF16 storage.',
        tolerance_scope='Forward/input/weight errors include BF16 storage and reduction rounding; smooth coordinate/geometry gradients use2e-5 relativeRMS. Zero-initialized and untuned evaluation original routed blocks remain bitwise identical.')
