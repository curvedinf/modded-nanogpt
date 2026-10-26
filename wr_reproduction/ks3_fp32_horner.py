"""FP32 polynomial accumulation during KS3 training, with original boundaries.

Original joint coefficient GS32, FP16 tanh arguments/results, FP16 final
ReLU-square, GS32 hidden packing, BF16 products and smooth FP32 STE remain.
Untuned full/probe evaluation uses the original forward without modification.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale
from wr_reproduction import ks3_moe as core

ORIGINAL_FORWARD = core.forward
ORIGINAL_VJP = core._vjp
KIND = 'fp32-polynomial-horner-with-fp16-boundaries-v1'
EXECUTED_MODES = set()


@tr.jit
def geometry(C, G, H, TILES, M: tl.constexpr, N: tl.constexpr,
             BM: tl.constexpr=64):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*32 + tl.arange(0, 32)
    e = tl.load(TILES + tl.program_id(0)*BM//128); valid = cols < N
    gp = G + (e*N + cols)*8
    c0 = tl.load(gp, valid, other=0.); c1 = tl.load(gp+1, valid, other=0.)
    c2 = tl.load(gp+2, valid, other=0.); c3 = tl.load(gp+3, valid, other=0.)
    c4 = tl.load(gp+4, valid, other=0.); c5 = tl.load(gp+5, valid, other=0.)
    amax = tl.maximum(tl.maximum(tl.abs(c0), tl.abs(c1)),
        tl.maximum(tl.maximum(tl.abs(c2), tl.abs(c3)), tl.maximum(tl.abs(c4), tl.abs(c5))))
    ps = _half_scale(amax)
    c0 = tl.minimum(tl.maximum(libdevice.nearbyint(c0/ps), -127.), 127.)
    c1 = tl.minimum(tl.maximum(libdevice.nearbyint(c1/ps), -127.), 127.)
    c2 = tl.minimum(tl.maximum(libdevice.nearbyint(c2/ps), -127.), 127.)
    c3 = tl.minimum(tl.maximum(libdevice.nearbyint(c3/ps), -127.), 127.)
    c4 = tl.minimum(tl.maximum(libdevice.nearbyint(c4/ps), -127.), 127.)
    c5 = tl.minimum(tl.maximum(libdevice.nearbyint(c5/ps), -127.), 127.)
    raw = tl.load(gp+6, valid, other=0.); rh = tl.load(gp+7, valid, other=0.)
    scale = tl.where(raw > 20., raw, libdevice.log1p(libdevice.exp(raw)))+1e-4
    shift = .5*libdevice.tanh(rh)
    x = tl.load(C+rows[:, None]*N+cols[None, :], (rows[:, None]<M)&valid[None, :], other=0.)
    x = tl.minimum(tl.maximum(x, -65504.), 65504.)
    # Preserve original FP16 tanh boundaries; only Horner stages use FP32.
    th = (x*.25).to(tl.float16)
    t = (4*libdevice.tanh(th.to(tl.float32))).to(tl.float16).to(tl.float32)
    warp = (scale[None, :]*(t+shift[None, :])).to(tl.float16)
    z = libdevice.tanh(warp.to(tl.float32)).to(tl.float16).to(tl.float32)
    a = 2*c0-2*c1+c2+c3; b = -3*c0+3*c1-2*c2-c3
    poly = ((a[None, :]*t+b[None, :])*t+c2[None, :])*t+c0[None, :]
    warped = ((4*c5[None, :]*z+2*c4[None, :])*z-3*c5[None, :])*z-c4[None, :]
    h = tl.maximum((x+(poly+warped)*ps[None, :]).to(tl.float16), 0.)
    h = (h*h).to(tl.float16).to(tl.float32)
    h = tl.where(valid[None, :], h, 0.)
    hs = _half_scale(tl.max(tl.abs(h), 1))
    q = tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:, None]), -127.), 127.)
    tl.store(H+rows[:, None]*N+cols[None, :], q*hs[:, None], (rows[:, None]<M)&valid[None, :])


@torch.library.custom_op('wr_ks3_fp32_horner::forward', mutates_args=())
def forward(rows: torch.Tensor, src: torch.Tensor, tiles: torch.Tensor,
            up: torch.Tensor, down: torch.Tensor, g: torch.Tensor,
            gs32: bool=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not gs32 or core.KS3_RANK != 1:
        raise ValueError('FP32 Horner requires original GS32 and rank1 experts')
    if not core.BF16_TUNED_KERNELS:
        EXECUTED_MODES.add('original_evaluation')
        return ORIGINAL_FORWARD(rows, src, tiles, up, down, g, gs32)
    EXECUTED_MODES.add('training_fp32_horner')
    m, d = src.numel(), rows.shape[1]
    x = torch.empty((m, d), device=rows.device, dtype=torch.bfloat16)
    core._pack[(tr.cdiv(m, 128), tr.cdiv(d, 32))](
        rows, x, m, d, *rows.stride(), SRC=src, INPUT_ROWS=rows.shape[0],
        ROWS_PER_TILE=128, GS32=True, num_warps=4)
    c = core.mm(x, up.transpose(1, 2), tiles)
    h = torch.empty_like(c, dtype=torch.bfloat16)
    geometry[(tr.cdiv(m, 64), tr.cdiv(c.shape[1], 32))](
        c, g, h, tiles, *c.shape, BM=64, num_warps=4)
    y = core.mm(h, down.transpose(1, 2), tiles, dtype=rows.dtype)
    return y, x, c, h


@forward.register_fake
def _(rows, src, tiles, up, down, g, gs32=True):
    m, d, n = src.numel(), rows.shape[1], up.shape[1]
    return rows.new_empty((m, d)), rows.new_empty((m, d)), \
        rows.new_empty((m, n), dtype=torch.float32), rows.new_empty((m, n))


def reference_coefficients(g, ste=False):
    """Original coefficient GS32 and identity STE, independent Torch math."""
    raw = g[..., :6]
    scale = (raw.abs().amax(-1, keepdim=True)/127).clamp(2**-24, 65504).half().float()
    quantized = (raw/scale).round().clamp(-127, 127)*scale
    coefficients = raw+(quantized-raw).detach() if ste else quantized
    return torch.cat((coefficients, g[..., 6:]), -1)


def install():
    if core.forward not in (ORIGINAL_FORWARD, forward) or core._vjp is not ORIGINAL_VJP:
        raise RuntimeError('Unexpected KS3 overlay before FP32 Horner')
    core.forward = forward
