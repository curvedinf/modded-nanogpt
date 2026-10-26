"""Evaluation-only component removals at unchanged quantized KS3 coefficients.

These counterfactuals are diagnostics, never training recipes or screening
references. Keep the original six-coefficient quantization scale when removing
a component. This avoids silently requantizing the surviving coefficients.
"""
from contextlib import contextmanager
from unittest.mock import patch
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale, _spine_features
from wr_reproduction import ks3_moe as core

MODES = ('full', 'no_spine', 'no_warp', 'coordinate_only')


@tr.jit
def _effect(C, G, H, TILES, M: tl.constexpr, N: tl.constexpr,
            SPINE: tl.constexpr, WARP: tl.constexpr, GS32: tl.constexpr,
            BM: tl.constexpr = 16):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    e = tl.load(TILES + tl.program_id(0) * BM // 128)
    valid = cols < N
    gp = G + (e * N + cols) * 8
    c0 = tl.load(gp + 0, valid, other=0.); c1 = tl.load(gp + 1, valid, other=0.)
    c2 = tl.load(gp + 2, valid, other=0.); c3 = tl.load(gp + 3, valid, other=0.)
    c4 = tl.load(gp + 4, valid, other=0.); c5 = tl.load(gp + 5, valid, other=0.)
    amax = tl.maximum(tl.maximum(tl.abs(c0), tl.abs(c1)), tl.maximum(
        tl.maximum(tl.abs(c2), tl.abs(c3)), tl.maximum(tl.abs(c4), tl.abs(c5))))
    ps = _half_scale(amax)
    c0 = libdevice.nearbyint(c0 / ps); c1 = libdevice.nearbyint(c1 / ps)
    c2 = libdevice.nearbyint(c2 / ps); c3 = libdevice.nearbyint(c3 / ps)
    c4 = libdevice.nearbyint(c4 / ps); c5 = libdevice.nearbyint(c5 / ps)
    c0 = tl.minimum(tl.maximum(c0, -127.), 127.); c1 = tl.minimum(tl.maximum(c1, -127.), 127.)
    c2 = tl.minimum(tl.maximum(c2, -127.), 127.); c3 = tl.minimum(tl.maximum(c3, -127.), 127.)
    c4 = tl.minimum(tl.maximum(c4, -127.), 127.); c5 = tl.minimum(tl.maximum(c5, -127.), 127.)
    if not SPINE:
        c0 = tl.full((32,), 0., tl.float32); c1 = c0; c2 = c0; c3 = c0
    if not WARP:
        c4 = tl.full((32,), 0., tl.float32); c5 = c4
    raw = tl.load(gp + 6, valid, other=0.); rh = tl.load(gp + 7, valid, other=0.)
    scale = tl.where(raw > 20., raw, libdevice.log1p(libdevice.exp(raw))) + 1e-4
    shift = .5 * libdevice.tanh(rh)
    x = tl.load(C + rows[:, None] * N + cols[None, :],
                (rows[:, None] < M) & valid[None, :], other=0.)
    x = tl.minimum(tl.maximum(x, -65504.), 65504.)
    h = _spine_features(x, x, c0, c1, c2, c3, c4, c5,
                        ps, scale, shift, G, valid, 1)
    if GS32:
        hs = _half_scale(tl.max(tl.abs(h), 1))
        q = tl.minimum(tl.maximum(libdevice.nearbyint(h / hs[:, None]), -127.), 127.)
        h = q * hs[:, None]
    tl.store(H + rows[:, None] * N + cols[None, :], h,
             (rows[:, None] < M) & valid[None, :])


def geometry(c, g, tiles, mode, gs32=True, bm=16):
    if mode not in MODES:
        raise ValueError('Unknown geometry counterfactual: ' + mode)
    assert g.shape[-1] == 8 and g.shape[1] == c.shape[1]
    out = torch.empty_like(c, dtype=torch.bfloat16)
    _effect[(tr.cdiv(c.shape[0], bm), tr.cdiv(c.shape[1], 32))](
        c, g, out, tiles, *c.shape, mode in ('full', 'no_warp'),
        mode in ('full', 'no_spine'), gs32, BM=bm, num_warps=4)
    return out


@torch.no_grad()
def numerical_check():
    """Exact original producer parity; component identity and no input mutation."""
    cases = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1369)
        for m in (128, 384):
            c = torch.randn(m, 1024, device='cuda') * 3
            tiles = (torch.arange(tr.cdiv(m, 128), device='cuda') % 8).int()
            for nonzero in (False, True):
                g = torch.randn(8, 1024, 8, device='cuda') * .03
                if not nonzero: g[..., :6].zero_()
                g[..., 6] = .75; g[..., 7] = 0.
                saved_c, saved_g = c.clone(), g.clone()
                for gs32 in (False, True):
                    for bm in (16, 64):
                        expected = torch.empty_like(c, dtype=torch.bfloat16)
                        core._geometry[(tr.cdiv(m, bm), tr.cdiv(1024, 32))](
                            c, g, expected, tiles, m, 1024, BM=bm, RANK=1, GS32=gs32, num_warps=4)
                        actual = geometry(c, g, tiles, 'full', gs32, bm)
                        assert torch.equal(actual, expected), ('original producer parity', m, nonzero, gs32, bm)
                        if not nonzero:
                            for mode in MODES[1:]:
                                assert torch.equal(geometry(c, g, tiles, mode, gs32, bm), expected)
                        cases.append(dict(rows=m, nonzero_geometry=nonzero, gs32=gs32, block_rows=bm, bitwise_equal=True))
                for mode, removed in (('no_spine', slice(0, 4)), ('no_warp', slice(4, 6)),
                                      ('coordinate_only', slice(0, 6))):
                    zeroed = g.clone(); zeroed[..., removed] = 0.
                    assert torch.equal(geometry(c, zeroed, tiles, mode), geometry(c, zeroed, tiles, 'full'))
                assert torch.equal(c, saved_c) and torch.equal(g, saved_g)
    return dict(passed=True, cases=cases, component_removal_after_original_coefficient_quantization=True,
                initial_function_bitwise_equal=True, original_inputs_unchanged=True,
                evaluation_only=True, backward_unchanged=True)


@torch.no_grad()
def local_effects(args, result, site):
    """Same-coordinate effects; full production kernels, bounded moment samples."""
    from wr_reproduction.ks3_update_observer import evenly
    rows, src, tiles, up, down, g, *rest = args
    y, x, c, h = result
    exact = geometry(c, g, tiles, 'full')
    assert torch.equal(h, exact), 'Counterfactual producer differs from actual model baseline'
    raw = geometry(c, g, tiles, 'full', gs32=False)
    hi = evenly(h.numel(), h.device); yi = evenly(y.numel(), y.device)
    hv = src[hi // h.shape[1]] < rows.shape[0]
    yv = src[yi // y.shape[1]] < rows.shape[0]
    hs = h.flatten()[hi].float(); rs = raw.flatten()[hi].float(); ys = y.flatten()[yi].float()
    def rms(v, valid):
        return ((v.square() * valid).sum() / valid.sum().clamp_min(1)).sqrt()
    output = []
    for mode in MODES[1:]:
        counter = geometry(c, g, tiles, mode)
        unquantized = geometry(c, g, tiles, mode, gs32=False)
        projected = core.mm(counter, down.transpose(1, 2), tiles, dtype=rows.dtype)
        dh = counter.flatten()[hi].float() - hs
        dr = unquantized.flatten()[hi].float() - rs
        dy = projected.flatten()[yi].float() - ys
        vals = torch.stack((rms(hs, hv), rms(dh, hv), rms(dr, hv), rms(dh-dr, hv),
                            ((dh != 0) * hv).sum() / hv.sum().clamp_min(1),
                            ((dr != 0) * hv).sum() / hv.sum().clamp_min(1),
                            rms(ys, yv), rms(dy, yv)))
        output.append(dict(site=site, mode=mode, values=vals))
    return output


def reports(rows):
    keys = ('packed_activation_rms', 'packed_delta_rms', 'unpacked_delta_rms',
            'packing_delta_error_rms', 'packed_changed_fraction', 'unpacked_changed_fraction',
            'projected_output_rms', 'projected_delta_rms')
    result = []
    for row in rows:
        values = dict(zip(keys, row['values'].cpu().tolist(), strict=True))
        values['relative_packed_delta_rms'] = values['packed_delta_rms'] / max(values['packed_activation_rms'], 1e-12)
        values['relative_projected_delta_rms'] = values['projected_delta_rms'] / max(values['projected_output_rms'], 1e-12)
        result.append(dict(site=row['site'], mode=row['mode'], **values))
    return result


@contextmanager
def counterfactual(runner, mode, capture=False):
    if mode not in MODES:
        raise ValueError('Unknown geometry counterfactual: ' + mode)
    from wr_reproduction.probe_study import stopped
    original = core.forward
    collected = []; calls = 0
    def forward(*args, **kwargs):
        nonlocal calls
        if stopped(): raise RuntimeError('Explicit study STOP')
        if runner.model.training or torch.is_grad_enabled():
            raise RuntimeError('Geometry counterfactual is evaluation-only')
        result = original(*args, **kwargs)
        if capture and calls < 5:
            collected.extend(local_effects(args, result, calls))
        calls += 1
        if mode == 'full': return result
        rows, src, tiles, up, down, g, *rest = args
        y, x, c, h = result
        h = geometry(c, g, tiles, mode)
        y = core.mm(h, down.transpose(1, 2), tiles, dtype=rows.dtype)
        return y, x, c, h
    with patch.object(core, 'forward', forward):
        yield collected
