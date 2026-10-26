"""Compact interactions between coordinate differences, preserving rank1 paths.

h_i = BF16(original_GS32_h_i + tanh(raw_i) * b(c_i - c_(i xor1)))
b(d) = max(1 - (d/2)^2, 0)^2, using separately clamped coordinates.

The bounded C1 residual responds to proximity across sign quadrants. It is
neither a separable product of coordinate functions nor a replacement of the
independent features by an average. Zero coefficients preserve the original
function and all original gradients. No convergence benefit is assumed.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from wr_reproduction import ks3_moe as core, ks3_pairwise as routed

KIND = 'bounded-compact-coordinate-difference-v1'
ORIGINAL_FEATURES = routed.features
ORIGINAL_FEATURE_VJP = routed.feature_vjp
ORIGINAL_ROUTED_FORWARD = routed.forward
forward = ORIGINAL_ROUTED_FORWARD
backward = routed.backward
RoutedDifferencePair = routed.RoutedPairwise


def reference(c, h, raw, bound=1.):
    partner = torch.arange(c.shape[-1], device=c.device) ^ 1
    x = c.clamp(-65504., 65504.)
    difference = x - x[..., partner]
    basis = (1. - (difference * .5).square()).clamp_min(0.).square()
    return h + bound * raw.tanh() * basis


def configure(settings):
    expected = dict(kind=KIND, bound=1., difference_width=2., lr_mul=.125,
        initial_raw_coefficient=0., placement='after-independent-GS32')
    if settings != expected:
        raise ValueError('Only the frozen compact coordinate-difference interaction is covered')
    if routed.features not in (ORIGINAL_FEATURES, features) or routed.feature_vjp not in (ORIGINAL_FEATURE_VJP, feature_vjp):
        raise RuntimeError('Different pair feature overlay already installed')
    if routed.forward not in (ORIGINAL_ROUTED_FORWARD, _routed_forward):
        raise RuntimeError('Different routed forward overlay already installed')
    routed.features = features
    routed.feature_vjp = feature_vjp
    routed.forward = _routed_forward
    routed.configure(settings)
    return expected


def _routed_forward(*args, **kwargs):
    return forward(*args, **kwargs)


@tr.jit
def _basis(original, other):
    x = tl.minimum(tl.maximum(original, -65504.), 65504.)
    xj = tl.minimum(tl.maximum(other, -65504.), 65504.)
    difference = x - xj
    t = difference * .5
    z = tl.maximum(1. - t*t, 0.)
    basis = z*z
    derivative = tl.where((original >= -65504.) & (original <= 65504.), -difference*z, 0.)
    return basis, derivative


@tr.jit
def _add_features(C, RAW, BASE, H, TILES, M:tl.constexpr, N:tl.constexpr,
                  BOUND:tl.constexpr, BM:tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*32 + tl.arange(0, 32)
    mask = (row[:, None] < M) & (col[None, :] < N)
    expert = tl.load(TILES + tl.program_id(0)*BM//128)
    x = tl.load(C + row[:, None]*N + col[None, :], mask, other=0.)
    xj = tl.load(C + row[:, None]*N + (col[None, :] ^ 1), mask, other=0.)
    basis, derivative = _basis(x, xj)
    a = libdevice.tanh(tl.load(RAW + expert*N + col, col < N, other=0.))
    h = tl.load(BASE + row[:, None]*N + col[None, :], mask, other=0.).to(tl.float32)
    tl.store(H + row[:, None]*N + col[None, :], h + BOUND*a[None, :]*basis, mask)


@tr.jit
def _add_vjp(V, C, RAW, BASE_DC, DC, PART, TILES, SRC,
             M:tl.constexpr, N:tl.constexpr, ROWS:tl.constexpr,
             BOUND:tl.constexpr, BM:tl.constexpr=64):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*32 + tl.arange(0, 32)
    mask = (row[:, None] < M) & (col[None, :] < N)
    real = tl.load(SRC + row, row < M, other=ROWS) < ROWS
    expert = tl.load(TILES + tl.program_id(0)*BM//128)
    x = tl.load(C + row[:, None]*N + col[None, :], mask, other=0.)
    xj = tl.load(C + row[:, None]*N + (col[None, :] ^ 1), mask, other=0.)
    basis, derivative = _basis(x, xj)
    a = libdevice.tanh(tl.load(RAW + expert*N + col, col < N, other=0.))
    aj = libdevice.tanh(tl.load(RAW + expert*N + (col ^ 1), col < N, other=0.))
    v = tl.load(V + row[:, None]*N + col[None, :], mask, other=0.).to(tl.float32)
    vj = tl.load(V + row[:, None]*N + (col[None, :] ^ 1), mask, other=0.).to(tl.float32)
    # b is even. Both directed residuals have the same derivative wrt c_i.
    added = (v*a[None, :] + vj*aj[None, :])*BOUND*derivative
    added = tl.where(mask & real[:, None], added, 0.)
    base = tl.load(BASE_DC + row[:, None]*N + col[None, :], mask, other=0.).to(tl.float32)
    tl.store(DC + row[:, None]*N + col[None, :], base + added, mask)
    dp = tl.where(mask & real[:, None], v*BOUND*(1.-a[None, :]*a[None, :])*basis, 0.)
    tl.store(PART + tl.program_id(0)*N + col, tl.sum(dp, 0), col < N)


def features(c, g, raw, tiles, bound, gs32=True):
    if not gs32 or core.KS3_RANK != 1:
        raise ValueError('Difference pair requires original GS32 independent rank1 paths')
    m, n = c.shape
    bm = 64 if core.BF16_TUNED_KERNELS else 16
    base = torch.empty_like(c, dtype=torch.bfloat16)
    core._geometry[(tr.cdiv(m, bm), tr.cdiv(n, 32))](
        c, g, base, tiles, m, n, BM=bm, RANK=1, GS32=True, num_warps=4)
    h = torch.empty_like(base)
    _add_features[(tr.cdiv(m, bm), tr.cdiv(n, 32))](
        c, raw, base, h, tiles, m, n, bound, BM=bm, num_warps=4)
    return h


def feature_vjp(v, c, g, raw, tiles, src, offs, rows, bound):
    m, n = c.shape
    e = g.shape[0]
    # Preserve the original BF16 coordinate-gradient storage boundary.
    base_dc = torch.empty_like(c, dtype=torch.bfloat16)
    part = torch.empty((tr.cdiv(m, 64), n, 8), device=c.device, dtype=torch.float32)
    core._vjp[(tr.cdiv(m, 64), tr.cdiv(n, 32))](
        v, c, g, base_dc, part, tiles, src, m, n, rows, RANK=1, num_warps=4)
    dg = torch.empty_like(g)
    core._reduce[(e, n)](part, dg, offs, n, part.shape[0],
        tr.next_power_of_2(part.shape[0]), RANK=1, num_warps=4)
    dc = torch.empty_like(c, dtype=torch.bfloat16)
    pair_part = torch.empty((tr.cdiv(m, 64), n), device=c.device, dtype=torch.float32)
    _add_vjp[(tr.cdiv(m, 64), tr.cdiv(n, 32))](
        v, c, raw, base_dc, dc, pair_part, tiles, src, m, n, rows, bound, num_warps=4)
    dp = torch.empty_like(raw)
    routed._reduce_pair[(e, tr.cdiv(n, 32))](pair_part, dp, offs, n, pair_part.shape[0],
        tr.next_power_of_2(pair_part.shape[0]), num_warps=4)
    return dc, dg, dp


def install(base_install, settings, *args, **kwargs):
    configure(settings)
    return routed.install(base_install, settings, *args, **kwargs)
