"""Direct FP16 geometry coefficients in otherwise original rank-one KS3.

Each of the six coefficients is rounded to FP16 independently, instead of
partial GS32 quantization. FP16 Horner stages, raw FP32 scale/shift, GS32
activation/projection packing, BF16 products and original FP32 STE remain.
This tests coefficient accuracy and quantizer cost without adding pathways.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale
from wr_reproduction import ks3_moe as core

ORIGINAL_GEOMETRY = core._geometry
ORIGINAL_VJP = core._vjp
KIND = 'direct-fp16-geometry-coefficients-v1'


@tr.jit
def _coefficients(G, expert, cols, N: tl.constexpr):
    gp = G + (expert*N + cols)*8
    valid = cols < N
    c0 = tl.load(gp, valid, other=0.); c1 = tl.load(gp+1, valid, other=0.)
    c2 = tl.load(gp+2, valid, other=0.); c3 = tl.load(gp+3, valid, other=0.)
    c4 = tl.load(gp+4, valid, other=0.); c5 = tl.load(gp+5, valid, other=0.)
    c0=c0.to(tl.float16).to(tl.float32); c1=c1.to(tl.float16).to(tl.float32)
    c2=c2.to(tl.float16).to(tl.float32); c3=c3.to(tl.float16).to(tl.float32)
    c4=c4.to(tl.float16).to(tl.float32); c5=c5.to(tl.float16).to(tl.float32)
    unit=tl.full((32,),1.,tl.float32)
    return c0, c1, c2, c3, c4, c5, unit, unit


@tr.jit
def geometry(C, G, H, TILES, M: tl.constexpr, N: tl.constexpr, HT=None,
             SCALE=None, AMAX=None, FP8: tl.constexpr=False,
             PARTIAL_AMAX: tl.constexpr=False, BM: tl.constexpr=16,
             GS32: tl.constexpr=True, RANK: tl.constexpr=1, NATIVE_OUT=None, NATIVE_T=None):
    tl.static_assert(RANK == 1 and not FP8, 'This hypothesis supports rank1 BF16 products only')
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*32 + tl.arange(0, 32)
    e = tl.load(TILES + tl.program_id(0)*BM//128); valid = cols < N
    c0, c1, c2, c3, c4, c5, ps, pw = _coefficients(G, e, cols, N)
    gp = G + (e*N + cols)*8
    rs = tl.load(gp+6, valid, other=0.); rh = tl.load(gp+7, valid, other=0.)
    scale = tl.where(rs > 20., rs, libdevice.log1p(libdevice.exp(rs)))+1e-4
    shift = .5*libdevice.tanh(rh)
    x = tl.load(C + rows[:, None]*N + cols[None, :], (rows[:, None]<M)&valid[None, :], other=0.)
    x = tl.minimum(tl.maximum(x, -65504.), 65504.)
    th = (x*.25).to(tl.float16)
    t = (4*libdevice.tanh(th.to(tl.float32))).to(tl.float16)
    warp = (scale[None, :]*(t.to(tl.float32)+shift[None, :])).to(tl.float16)
    z = libdevice.tanh(warp.to(tl.float32)).to(tl.float16)
    a = 2*c0-2*c1+c2+c3; b = -3*c0+3*c1-2*c2-c3
    poly = (a[None, :].to(tl.float16)*t+b[None, :].to(tl.float16)).to(tl.float16)
    poly = (poly*t+c2[None, :].to(tl.float16)).to(tl.float16)
    poly = (poly*t+c0[None, :].to(tl.float16)).to(tl.float16)
    warped = (4*c5[None, :].to(tl.float16)*z+2*c4[None, :].to(tl.float16)).to(tl.float16)
    warped = (warped*z-3*c5[None, :].to(tl.float16)).to(tl.float16)
    warped = (warped*z-c4[None, :].to(tl.float16)).to(tl.float16)
    # Unit scales: coefficients already have their physical FP16 values.
    pre = x + (poly.to(tl.float32)*ps[None, :] + warped.to(tl.float32)*pw[None, :])
    h = tl.maximum(pre.to(tl.float16), 0.)
    h = (h*h).to(tl.float16).to(tl.float32)
    h = tl.where(valid[None, :], h, 0.)
    if GS32:
        hs = _half_scale(tl.max(tl.abs(h), 1))
        q = tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:, None]), -127.), 127.)
        h = q*hs[:, None]
    tl.store(H + rows[:, None]*N + cols[None, :], h, (rows[:, None]<M)&valid[None, :])


@tr.jit
def vjp(V, C, G, DC, PART, TILES, SRC, M: tl.constexpr, N: tl.constexpr,
        ROWS: tl.constexpr, DCT=None, SCALE=None, AMAX=None,
        FP8: tl.constexpr=False, PARTIAL_AMAX: tl.constexpr=False,
        BM: tl.constexpr=64, RANK: tl.constexpr=1, NATIVE_OUT=None, NATIVE_T=None):
    tl.static_assert(RANK == 1 and not FP8, 'This hypothesis supports rank1 BF16 products only')
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*32 + tl.arange(0, 32)
    e = tl.load(TILES + tl.program_id(0)*BM//128)
    mask = (rows[:, None]<M)&(cols[None, :]<N)
    real = tl.load(SRC+rows, rows<M, other=ROWS)<ROWS
    value = tl.load(V+rows[:, None]*N+cols[None, :], mask, other=0.)
    original = tl.load(C+rows[:, None]*N+cols[None, :], mask, other=0.)
    x = tl.minimum(tl.maximum(original, -65504.), 65504.)
    c0, c1, c2, c3, c4, c5, ps, pw = _coefficients(G, e, cols, N)
    c0 = c0*ps; c1 = c1*ps; c2 = c2*ps; c3 = c3*ps; c4 = c4*pw; c5 = c5*pw
    gp = G+(e*N+cols)*8
    rs = tl.load(gp+6, cols<N, other=0.); rh = tl.load(gp+7, cols<N, other=0.)
    scale = tl.where(rs>20., rs, libdevice.log1p(libdevice.exp(rs)))+1e-4
    shift = .5*libdevice.tanh(rh)
    t = 4.*libdevice.tanh(x*.25); z = libdevice.tanh(scale[None, :]*(t+shift[None, :]))
    cubic = 2*c0-2*c1+c2+c3; quadratic = -3*c0+3*c1-2*c2-c3
    spine = ((cubic*t+quadratic)*t+c2)*t+c0
    warped = ((4*c5*z+2*c4)*z-3*c5)*z-c4
    dp = tl.where(mask&real[:, None], value*2.*tl.maximum(x+spine+warped, 0.), 0.)
    dz = ((12*c5*z+4*c4)*z-3*c5)*(1.-z*z)
    dm = dp*((3*cubic*t+2*quadratic)*t+c2+dz*scale)*(1.-t*t*.0625)
    dc = tl.where((original>=-65504.)&(original<=65504.), dp+dm, 0.)
    tl.store(DC+rows[:, None]*N+cols[None, :], dc, mask)
    off = (tl.program_id(0)*N+cols)*8; t2 = t*t; t3 = t2*t
    tl.store(PART+off, tl.sum(dp*(2*t3-3*t2+1), 0), cols<N)
    tl.store(PART+off+1, tl.sum(dp*(-2*t3+3*t2), 0), cols<N)
    tl.store(PART+off+2, tl.sum(dp*(t3-2*t2+t), 0), cols<N)
    tl.store(PART+off+3, tl.sum(dp*(t3-t2), 0), cols<N)
    tl.store(PART+off+4, tl.sum(dp*(2*z*z-1), 0), cols<N)
    tl.store(PART+off+5, tl.sum(dp*(4*z*z*z-3*z), 0), cols<N)
    tl.store(PART+off+6, tl.sum(dp*dz*(t+shift)*tl.sigmoid(rs), 0), cols<N)
    tl.store(PART+off+7, tl.sum(dp*dz*scale*.5*(1.-libdevice.tanh(rh)*libdevice.tanh(rh)), 0), cols<N)


def reference_coefficients(g, ste=False):
    """Independent Torch FP16 rounding with the original straight-through rule."""
    raw=g[..., :6]; rounded=raw.half().float()
    coeff=raw+(rounded-raw).detach() if ste else rounded
    return torch.cat((coeff,g[...,6:]),-1)


def install():
    if core._geometry is geometry and core._vjp is vjp: return
    if core._geometry is not ORIGINAL_GEOMETRY or core._vjp is not ORIGINAL_VJP:
        raise RuntimeError('Unexpected prior KS3 geometry kernel overlay')
    core._geometry = geometry; core._vjp = vjp
