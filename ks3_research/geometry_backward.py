"""Standalone KS3 geometry VJP after an FP32 BLAS feature-gradient product.

Adapted from curvedinf/k-splanifolds ks3_int8._backward_mm, pinned
ea7a9c1121efe711345bd9e6cf22e69401c0c7b4, CC-BY-SA-4.0. See vendor/PROVENANCE.json
and vendor/LICENSE.k-splanifolds. The original source remains unchanged.
The surrogate and clipping rules are identical; a larger row tile reduces
partial-gradient storage, while the feature gradient is overwritten in place.
"""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice


@tr.jit
def geometry_vjp(VALUE, COORD, P, PS, SCALE, SHIFT, RAW_SCALE, RAW_SHIFT,
                 PARTIAL, MEAN, DMEAN, RANK: tl.constexpr,
                 M: tl.constexpr, N: tl.constexpr,
                 BM: tl.constexpr = 64, BN: tl.constexpr = 32):
    rows = tl.program_id(0)*BM + tl.arange(0,BM)
    cols = tl.program_id(1)*BN + tl.arange(0,BN)
    mask = (rows[:,None]<M) & (cols[None,:]<N)
    value = tl.load(VALUE+rows[:,None]*N+cols[None,:],mask,other=0.)
    original = tl.load(COORD+rows[:,None]*N+cols[None,:],mask,other=0.)
    x = tl.minimum(tl.maximum(original,-65504.),65504.)
    scale = tl.load(SCALE+cols//RANK,cols<N,other=1).to(tl.float32)
    shift = tl.load(SHIFT+cols//RANK,cols<N,other=0).to(tl.float32)
    ps = tl.load(PS+cols,cols<N,other=0).to(tl.float32)
    c0 = tl.load(P+cols*6,cols<N,other=0).to(tl.float32)*ps
    c1 = tl.load(P+cols*6+1,cols<N,other=0).to(tl.float32)*ps
    c2 = tl.load(P+cols*6+2,cols<N,other=0).to(tl.float32)*ps
    c3 = tl.load(P+cols*6+3,cols<N,other=0).to(tl.float32)*ps
    c4 = tl.load(P+cols*6+4,cols<N,other=0).to(tl.float32)*ps
    c5 = tl.load(P+cols*6+5,cols<N,other=0).to(tl.float32)*ps
    if RANK == 1:
        mean = x
    else:
        mean = tl.load(MEAN+rows[:,None]*(N//RANK)+cols[None,:]//RANK,mask,other=0.)
    t = 4.*libdevice.tanh(mean*.25)
    z = libdevice.tanh(scale[None,:]*(t+shift[None,:]))
    cubic = 2*c0-2*c1+c2+c3
    quadratic = -3*c0+3*c1-2*c2-c3
    spine = ((cubic*t+quadratic)*t+c2)*t+c0
    warped = ((4*c5*z+2*c4)*z-3*c5)*z-c4
    dpre = value*2.*tl.maximum(x+spine+warped,0.)
    dpre = tl.where(mask,dpre,0.)
    dpoly = (3*cubic*t+2*quadratic)*t+c2
    dwarp = (12*c5*z+4*c4)*z-3*c5
    dz = dwarp*(1.-z*z)
    dmean = dpre*(dpoly+dz*scale)*(1.-t*t*.0625)
    if RANK == 1:
        value = tl.where((original>=-65504.) & (original<=65504.),dpre+dmean,0.)
    else:
        value = dpre
        tl.store(DMEAN+rows[:,None]*N+cols[None,:],dmean,mask)
    offset = (tl.program_id(0)*N+cols)*8
    t2 = t*t
    t3 = t2*t
    tl.store(PARTIAL+offset,tl.sum(dpre*(2*t3-3*t2+1),0),cols<N)
    tl.store(PARTIAL+offset+1,tl.sum(dpre*(-2*t3+3*t2),0),cols<N)
    tl.store(PARTIAL+offset+2,tl.sum(dpre*(t3-2*t2+t),0),cols<N)
    tl.store(PARTIAL+offset+3,tl.sum(dpre*(t3-t2),0),cols<N)
    tl.store(PARTIAL+offset+4,tl.sum(dpre*(2*z*z-1),0),cols<N)
    tl.store(PARTIAL+offset+5,tl.sum(dpre*(4*z*z*z-3*z),0),cols<N)
    raw_scale = tl.load(RAW_SCALE+cols//RANK,cols<N,other=0)
    raw_shift = tl.load(RAW_SHIFT+cols//RANK,cols<N,other=0)
    dscale = dpre*dz*(t+shift)*tl.sigmoid(raw_scale)
    shift_tanh = libdevice.tanh(raw_shift)
    dshift = dpre*dz*scale*.5*(1.-shift_tanh*shift_tanh)
    tl.store(PARTIAL+offset+6,tl.sum(dscale,0),cols<N)
    tl.store(PARTIAL+offset+7,tl.sum(dshift,0),cols<N)
    tl.store(VALUE+rows[:,None]*N+cols[None,:],value,mask)
