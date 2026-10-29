import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from track_1_short.ks3.quantization import _half_scale

@tr.jit
def _prepare(G,P,CHARTS:tl.constexpr,B:tl.constexpr=256):
    row=tl.program_id(0)*B+tl.arange(0,B);valid=row<CHARTS;gp=G+row*8
    c0=tl.load(gp+0,valid,other=0.);c1=tl.load(gp+1,valid,other=0.)
    c2=tl.load(gp+2,valid,other=0.);c3=tl.load(gp+3,valid,other=0.)
    c4=tl.load(gp+4,valid,other=0.);c5=tl.load(gp+5,valid,other=0.)
    amax=tl.maximum(tl.maximum(tl.abs(c0),tl.abs(c1)),tl.maximum(tl.maximum(tl.abs(c2),tl.abs(c3)),tl.maximum(tl.abs(c4),tl.abs(c5))))
    ps=_half_scale(amax)
    q0=tl.minimum(tl.maximum(libdevice.nearbyint(c0/ps),-127.),127.)
    q1=tl.minimum(tl.maximum(libdevice.nearbyint(c1/ps),-127.),127.)
    q2=tl.minimum(tl.maximum(libdevice.nearbyint(c2/ps),-127.),127.)
    q3=tl.minimum(tl.maximum(libdevice.nearbyint(c3/ps),-127.),127.)
    q4=tl.minimum(tl.maximum(libdevice.nearbyint(c4/ps),-127.),127.)
    q5=tl.minimum(tl.maximum(libdevice.nearbyint(c5/ps),-127.),127.)
    raw=tl.load(gp+6,valid,other=0.);rh=tl.load(gp+7,valid,other=0.)
    scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4
    shift=.5*libdevice.tanh(rh)
    tl.store(P+row*9+0,q0,valid);tl.store(P+row*9+1,q1,valid)
    tl.store(P+row*9+2,q2,valid);tl.store(P+row*9+3,q3,valid)
    tl.store(P+row*9+4,q4,valid);tl.store(P+row*9+5,q5,valid)
    tl.store(P+row*9+6,ps,valid);tl.store(P+row*9+7,scale,valid)
    tl.store(P+row*9+8,shift,valid)

