"""Actual-width32-row BF16 geometry checks, including independent smooth VJP."""
from unittest.mock import patch
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_bf16_vjp32 as small
from wr_reproduction.check_ks3_hidden_bf16 import reference_vjp
from wr_reproduction.check_ks3_pairwise import error


def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(1869); torch.set_num_threads(1); core.KS3_RANK = 1
    e, n, d = 8, 1024, 768; checks = []; original_forward = core.forward
    up = core.pack((torch.randn(e*n, d, device='cuda')*.025).bfloat16()).view(e, n, d)
    down = torch.stack([core.pack((torch.randn(d, n, device='cuda')*.025).bfloat16()) for _ in range(e)])
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
            y, xp, c, h = core.forward(x, src, tiles, up, down, g)
            old = small.ORIGINAL_BACKWARD(grad, y, xp, c, h, gate, src, offs, tiles, up, down, g, bank, 0)
            old_bank = bank.clone()
            actual = small.backward(grad, y, xp, c, h, gate, src, offs, tiles, up, down, g, bank, 0)
            exact = dict(input_gradient=torch.equal(actual[0], old[0]), gate_gradient=torch.equal(actual[1], old[1]),
                         weight_gradient=torch.equal(bank, old_bank))
            assert all(exact.values()), (tuned, tokens, zero, exact)
            geometry_error = error(actual[2], old[2])
            component_errors = [error(actual[2][..., j], old[2][..., j]) for j in range(8)]
            assert max(geometry_error, *component_errors) < 3e-6, (tokens, geometry_error, component_errors)
            independent = None
            if tuned:
                valid = (src<tokens)[:, None]; idx = src.clamp(0, tokens-1).long()
                gg = torch.where(valid, grad[idx].float()*gate[idx, None].bfloat16().float(), 0.).bfloat16()
                v = core.mm(gg, down, tiles)
                with torch.enable_grad():
                    dc_ref, dg_ref = reference_vjp(c, g, v, src, tokens, offsets)
                # Coordinate values require the original BF16 output boundary.
                dc32 = torch.empty_like(c, dtype=torch.bfloat16); dc64 = torch.empty_like(dc32)
                for bm, dc in ((32, dc32), (64, dc64)):
                    part = torch.empty(tr.cdiv(c.shape[0], bm), n, 8, device='cuda')
                    core._vjp[(tr.cdiv(c.shape[0], bm), tr.cdiv(n, 32))](
                        v, c, g, dc, part, tiles, src, *c.shape, tokens, BM=bm, RANK=1, num_warps=4)
                assert torch.equal(dc32, dc64), (tokens, zero, 'BF16 coordinate storage differs')
                independent = dict(geometry=error(actual[2], dg_ref),
                    components=[error(actual[2][..., j], dg_ref[..., j]) for j in range(8)])
                assert max(independent['geometry'], *independent['components']) < 3e-5, independent
                # FP32 output checks the smooth derivative separately from BF16 rounding.
                dc_float = torch.empty_like(c); part = torch.empty(tr.cdiv(c.shape[0], 32), n, 8, device='cuda')
                core._vjp[(tr.cdiv(c.shape[0], 32), tr.cdiv(n, 32))](
                    v, c, g, dc_float, part, tiles, src, *c.shape, tokens, BM=32, RANK=1, num_warps=4)
                independent['coordinate'] = error(dc_float, dc_ref)
                assert independent['coordinate'] < 3e-5, independent
            else:
                assert torch.equal(actual[2], old[2])
            # Cover the actual routed wrapper, router/post, and both projection banks.
            xx = x.clone().requires_grad_(); router = (torch.randn(e, d, device='cuda')*.02).bfloat16().requires_grad_()
            post = torch.tensor(.8, device='cuda', requires_grad=True); geometry = g.clone().requires_grad_()
            def routed(backward):
                with patch.object(core, 'backward', backward):
                    out, _, _ = core.RoutedKS3.apply(xx, router, post, up, down, up, down, geometry, bank, 0, True)
                    grads = torch.autograd.grad(out, (xx, router, post, geometry), grad)
                    return out.detach(), [v.detach().clone() for v in grads], bank.clone()
            a = routed(small.ORIGINAL_BACKWARD); b = routed(small.backward)
            assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
            assert all(torch.equal(ga, gb) for ga, gb in zip(a[1][:3], b[1][:3]))
            routed_geometry = error(b[1][3], a[1][3]); assert routed_geometry < 3e-6
            if not tuned:
                assert torch.equal(a[1][3], b[1][3])
            row = dict(tuned=tuned, tokens=tokens, zero_coefficients=zero, active_experts=int((counts>0).sum()),
                padded_rows=src.numel(), original_input_gate_weight_gradients_bitwise=exact,
                original_forward_and_routed_input_router_post_weight_gradients_bitwise=True,
                geometry_reduction_relative_rms=geometry_error, geometry_component_relative_rms=component_errors,
                routed_geometry_relative_rms=routed_geometry, independent_smooth_ste_relative_rms=independent,
                original_untuned_all_values_bitwise=not tuned)
            checks.append(row); print(row, flush=True)
    small.install(); small.install()
    assert core.backward is small.backward and core.forward is original_forward
    assert small.EXECUTED_MODES == {'training_bf16_vjp32', 'original_untuned_backward'}
    return dict(passed=True, numerical_seed=1869, checks=checks, new_parameters=0,
        original_forward_and_evaluation=True, original_coordinate_gradient_BF16_storage=True,
        original_BF16_products_and_GS32_packing=True, original_optimizer_and_function=True,
        install_idempotent=True, full_convergence_verified=False, actual_model_training_tested=False,
        tolerance_scope='Forward/input/router/post/weight values require bitwise original parity. FP32 geometry reduction changes grouping:3e-6 relativeRMS bound, including each component. Independent smooth-autograd bound3e-5. Original untuned path bitwise unchanged.',
        oracle='Original64-row kernel plus independent Torch joint-coefficient packing and smooth FP32 autograd VJP; actual1/257/16384/49152-token shapes and empty experts.')
