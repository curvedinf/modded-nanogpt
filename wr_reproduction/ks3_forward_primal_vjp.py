"""Original forward; forward-rounded ReLU multiplier in the KS3 expert VJP.

The smooth FP32 polynomial Jacobian and coefficient identity STE remain. Only
its squared-ReLU multiplier becomes twice the FP16 primal used by forward.
This is a defined hybrid STE, not the derivative of the discrete FP16 map.
All forward/evaluation functions, BF16 products, packing and optimizers remain.
"""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale
from wr_reproduction import ks3_moe as core

ORIGINAL_VJP=core._vjp
ORIGINAL_GEOMETRY=core._geometry
ORIGINAL_FORWARD=core.forward
KIND='forward-fp16-relu-multiplier-smooth-jacobian-v1'

@tr.jit
def relu_primal(x,c0,c1,c2,c3,c4,c5,ps,scale,shift):
    # Pinned _spine_features NL=1 through the pre-square FP16 ReLU.
    # c0..c5 are integer-valued jointly GS32-quantized coefficients.
    a=2*c0-2*c1+c2+c3; b=-3*c0+3*c1-2*c2-c3
    th=(x*.25).to(tl.float16)
    t=(4*libdevice.tanh(th.to(tl.float32))).to(tl.float16)
    warp=(scale[None,:]*(t.to(tl.float32)+shift[None,:])).to(tl.float16)
    z=libdevice.tanh(warp.to(tl.float32)).to(tl.float16)
    # Use explicitFP32 FMA followed by the declaredFP16 stage boundary.
    # Intended to prevent arithmetic dependence on the DC output storage type;
    # operands still enter each stage with their originalFP16 values.
    tf=t.to(tl.float32);zf=z.to(tl.float32)
    poly=tl.fma(a[None,:].to(tl.float16).to(tl.float32),tf,b[None,:].to(tl.float16).to(tl.float32)).to(tl.float16)
    poly=tl.fma(poly.to(tl.float32),tf,c2[None,:].to(tl.float16).to(tl.float32)).to(tl.float16)
    poly=tl.fma(poly.to(tl.float32),tf,c0[None,:].to(tl.float16).to(tl.float32)).to(tl.float16)
    warped=tl.fma(4*c5[None,:].to(tl.float16).to(tl.float32),zf,2*c4[None,:].to(tl.float16).to(tl.float32)).to(tl.float16)
    warped=tl.fma(warped.to(tl.float32),zf,-3*c5[None,:].to(tl.float16).to(tl.float32)).to(tl.float16)
    warped=tl.fma(warped.to(tl.float32),zf,-c4[None,:].to(tl.float16).to(tl.float32)).to(tl.float16)
    return tl.maximum((x+(poly.to(tl.float32)+warped.to(tl.float32))*ps[None,:]).to(tl.float16),0.)

@tr.jit
def _coefficients(G, expert, cols, N: tl.constexpr):
    gp = G + (expert*N + cols)*8
    valid = cols < N
    c0 = tl.load(gp, valid, other=0.); c1 = tl.load(gp+1, valid, other=0.)
    c2 = tl.load(gp+2, valid, other=0.); c3 = tl.load(gp+3, valid, other=0.)
    c4 = tl.load(gp+4, valid, other=0.); c5 = tl.load(gp+5, valid, other=0.)
    ps = _half_scale(tl.maximum(tl.maximum(tl.abs(c0), tl.abs(c1)),
        tl.maximum(tl.maximum(tl.abs(c2), tl.abs(c3)), tl.maximum(tl.abs(c4), tl.abs(c5)))))
    pw = ps
    c0 = tl.minimum(tl.maximum(libdevice.nearbyint(c0/ps), -127.), 127.)
    c1 = tl.minimum(tl.maximum(libdevice.nearbyint(c1/ps), -127.), 127.)
    c2 = tl.minimum(tl.maximum(libdevice.nearbyint(c2/ps), -127.), 127.)
    c3 = tl.minimum(tl.maximum(libdevice.nearbyint(c3/ps), -127.), 127.)
    c4 = tl.minimum(tl.maximum(libdevice.nearbyint(c4/pw), -127.), 127.)
    c5 = tl.minimum(tl.maximum(libdevice.nearbyint(c5/pw), -127.), 127.)
    return c0, c1, c2, c3, c4, c5, ps, pw


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
    q0=c0; q1=c1; q2=c2; q3=c3; q4=c4; q5=c5
    c0 = c0*ps; c1 = c1*ps; c2 = c2*ps; c3 = c3*ps; c4 = c4*pw; c5 = c5*pw
    gp = G+(e*N+cols)*8
    rs = tl.load(gp+6, cols<N, other=0.); rh = tl.load(gp+7, cols<N, other=0.)
    scale = tl.where(rs>20., rs, libdevice.log1p(libdevice.exp(rs)))+1e-4
    shift = .5*libdevice.tanh(rh)
    t = 4.*libdevice.tanh(x*.25); z = libdevice.tanh(scale[None, :]*(t+shift[None, :]))
    cubic = 2*c0-2*c1+c2+c3; quadratic = -3*c0+3*c1-2*c2-c3
    spine = ((cubic*t+quadratic)*t+c2)*t+c0
    warped = ((4*c5*z+2*c4)*z-3*c5)*z-c4
    forward_relu = relu_primal(x,q0,q1,q2,q3,q4,q5,ps,scale,shift)
    # Explicit surrogate: original smooth Jacobian, multiplier from the
    # original rounded pre-square ReLU. No derivative of quantization.
    dp = tl.where(mask&real[:, None], value*2.*forward_relu.to(tl.float32), 0.)
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


def install():
    if core._vjp not in (ORIGINAL_VJP,vjp) or core._geometry is not ORIGINAL_GEOMETRY or core.forward is not ORIGINAL_FORWARD:
        raise RuntimeError('Unexpected KS3 overlay before forward-primal VJP')
    core._vjp=vjp
