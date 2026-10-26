"""Bounded coactivation of negative coordinates, added to original KS3 features.

h_i = BF16(original_GS32_h_i + tanh(raw_i) * u(c_i) * u(c_(i xor1)))
u(c) = max(-tanh(clamp(c, -65504, 65504)), 0)^2.

The new C1 basis vanishes if either coordinate is nonnegative. Its residual
is bounded by one and can activate an otherwise zero independent feature.
The original rank1 nonlinearities, BF16 coordinate VJP, projections, router
and optimizer plumbing remain unchanged. Coefficients start at exactly zero.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from wr_reproduction import ks3_moe as core, ks3_pairwise as routed

KIND = 'bounded-negative-coordinate-coactivation-v1'
ORIGINAL_FEATURES = routed.features
ORIGINAL_FEATURE_VJP = routed.feature_vjp
ORIGINAL_ROUTED_FORWARD = routed.forward
forward = ORIGINAL_ROUTED_FORWARD
backward = routed.backward
RoutedNegativePair = routed.RoutedPairwise


def reference(c, h, raw, bound=1.):
    partner = torch.arange(c.shape[-1], device=c.device) ^ 1
    u = (-c.clamp(-65504., 65504.).tanh()).clamp_min(0).square()
    return h + bound*raw.tanh()*u*u[..., partner]


def configure(settings):
    expected = dict(kind=KIND, bound=1., coordinate_scale=1., lr_mul=.125,
        initial_raw_coefficient=0., placement='after-independent-GS32')
    if settings != expected:
        raise ValueError('Only the frozen negative-coordinate coactivation is covered')
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
def _basis(x):
    t = libdevice.tanh(tl.minimum(tl.maximum(x, -65504.), 65504.))
    u = tl.where(x < 0., t*t, 0.)
    du = tl.where((x < 0.) & (x >= -65504.), 2.*t*(1.-t*t), 0.)
    return u, du


@tr.jit
def _add_features(C, RAW, BASE, H, TILES, M:tl.constexpr, N:tl.constexpr,
                  BOUND:tl.constexpr, BM:tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*32 + tl.arange(0, 32)
    mask = (row[:, None] < M) & (col[None, :] < N)
    expert = tl.load(TILES + tl.program_id(0)*BM//128)
    x = tl.load(C + row[:, None]*N + col[None, :], mask, other=0.)
    xj = tl.load(C + row[:, None]*N + (col[None, :] ^ 1), mask, other=0.)
    u, du = _basis(x)
    uj, duj = _basis(xj)
    a = libdevice.tanh(tl.load(RAW + expert*N + col, col < N, other=0.))
    h = tl.load(BASE + row[:, None]*N + col[None, :], mask, other=0.).to(tl.float32)
    tl.store(H + row[:, None]*N + col[None, :], h + BOUND*a[None, :]*u*uj, mask)


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
    u, du = _basis(x)
    uj, duj = _basis(xj)
    a = libdevice.tanh(tl.load(RAW + expert*N + col, col < N, other=0.))
    aj = libdevice.tanh(tl.load(RAW + expert*N + (col ^ 1), col < N, other=0.))
    v = tl.load(V + row[:, None]*N + col[None, :], mask, other=0.).to(tl.float32)
    vj = tl.load(V + row[:, None]*N + (col[None, :] ^ 1), mask, other=0.).to(tl.float32)
    added = (v*a[None, :] + vj*aj[None, :])*BOUND*du*uj
    added = tl.where(mask & real[:, None], added, 0.)
    base = tl.load(BASE_DC + row[:, None]*N + col[None, :], mask, other=0.).to(tl.float32)
    tl.store(DC + row[:, None]*N + col[None, :], base + added, mask)
    dp = tl.where(mask & real[:, None], v*BOUND*(1.-a[None, :]*a[None, :])*u*uj, 0.)
    tl.store(PART + tl.program_id(0)*N + col, tl.sum(dp, 0), col < N)


def features(c, g, raw, tiles, bound, gs32=True):
    if not gs32 or core.KS3_RANK != 1:
        raise ValueError('Negative pair requires original GS32 independent rank1 paths')
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
    # Original storage boundary is essential for bitwise zero-init gradients.
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
