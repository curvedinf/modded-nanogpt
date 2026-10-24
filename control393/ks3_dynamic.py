"""KS3 kernels with runtime expert partition sizes, avoiding per-route compilation.

Packing, geometry and reduction adapted from curvedinf/k-splanifolds,
ea7a9c1121efe711345bd9e6cf22e69401c0c7b4, CC-BY-SA-4.0, and
ks3_research/geometry_backward.py. Only the row counts become runtime inputs;
the GS32 equations, FP16 nonlinearities and FP32 surrogate are unchanged.
See ks3_research/vendor/PROVENANCE.json and LICENSE.k-splanifolds.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale, _spine_features, SCALE_DTYPE

@tr.jit(do_not_specialize=["M"])
def _quant_tile(X,Q,S,tile,M,K:tl.constexpr,G:tl.constexpr,
                B:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr):
    # Eight groups per program amortize launch/workgroup overhead. Each row
    # still has its own GS32 scale, including partial final groups.
    groups=tl.cdiv(K,G)
    index=tile*8+tl.arange(0,8)
    row=index//groups;group=index%groups;j=tl.arange(0,B)
    offsets=row[:,None]*K+group[:,None]*G+j[None,:]
    valid=(row[:,None]<M)&(group[:,None]*G+j[None,:]<K)&(j[None,:]<G)
    source=row[:,None]*S0+(group[:,None]*G+j[None,:])*S1
    x=tl.load(X+source,valid,other=0.).to(tl.float32)
    scale=_half_scale(tl.max(tl.abs(x),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(x/scale[:,None]),-127.),127.).to(tl.int8)
    tl.store(Q+offsets,q,valid)
    tl.store(S+index,scale,row<M)


@tr.jit(do_not_specialize=["M"])
def _quant(X,Q,S,M,K:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
    _quant_tile(X,Q,S,tl.program_id(0),M,K,G,B,K,1)


@tr.jit(do_not_specialize=["M"])
def _pack_training(X,D,U,P,RS,RH,QX,SX,QD,SD,QU,SU,QP,SP,SC,SH,
                   M,I:tl.constexpr,O:tl.constexpr,N:tl.constexpr,
                   R:tl.constexpr,D0:tl.constexpr,D1:tl.constexpr,
                   U0:tl.constexpr,U1:tl.constexpr):
    """Pack current activations, both projections and geometry in one launch."""
    pid=tl.program_id(0)
    tx=tl.cdiv(M*tl.cdiv(I,32),8)
    td=tl.cdiv(N*tl.cdiv(I,32),8)
    tu=tl.cdiv(O*tl.cdiv(N,32),8)
    if pid<tx:
        _quant_tile(X,QX,SX,pid,M,I,32,32,I,1)
    elif pid<tx+td:
        _quant_tile(D,QD,SD,pid-tx,N,I,32,32,D0,D1)
    elif pid<tx+td+tu:
        _quant_tile(U,QU,SU,pid-tx-td,O,N,32,32,U0,U1)
    else:
        tile=pid-tx-td-tu
        _quant_tile(P,QP,SP,tile,N,6,32,8,6,1)
        coordinate=tile*8+tl.arange(0,8)
        chart=coordinate//R
        valid=(coordinate<N)&(coordinate%R==0)
        raw=tl.load(RS+chart,valid,other=0.)
        shift=tl.load(RH+chart,valid,other=0.)
        scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4
        tl.store(SC+chart,scale,valid)
        tl.store(SH+chart,.5*libdevice.tanh(shift),valid)


def _pack_training_tensors(x,down,up,coefficients,raw_scale,raw_shift,rank):
    m,i=x.shape;n=down.shape[0];o=up.shape[0]
    def buffers(rows,cols):
        return (torch.empty((rows,cols),device=x.device,dtype=torch.int8),
                torch.empty((rows,tr.cdiv(cols,32)),device=x.device,dtype=SCALE_DTYPE))
    qx,sx=buffers(m,i);qd,sd=buffers(n,i);qu,su=buffers(o,n);p,ps=buffers(n,6)
    scale=torch.empty_like(raw_scale);shift=torch.empty_like(raw_shift)
    tiles=sum(tr.cdiv(rows*tr.cdiv(cols,32),8) for rows,cols in ((m,i),(n,i),(o,n),(n,6)))
    _pack_training[(tiles,)](x,down,up,coefficients.reshape(-1,6).contiguous(),raw_scale,raw_shift,
        qx,sx,qd,sd,qu,su,p,ps,scale,shift,m,i,o,n,rank,*down.stride(),*up.stride(),num_warps=4)
    return qx,sx,qd,sd,qu,su,p,ps.flatten(),scale,shift


@tr.jit(do_not_specialize=["M"])
def _geometry_quant(C, MEAN, P, PS, SC, SH, LUT, H, HS,
                    M, N:tl.constexpr, RANK:tl.constexpr,
                    NL:tl.constexpr, BM:tl.constexpr=16, BN:tl.constexpr=32):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    valid=cols<N
    original=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    mean=tl.load(MEAN+rows[:,None]*(N//RANK)+cols[None,:]//RANK,
                 (rows[:,None]<M)&valid[None,:],other=0.)
    c0=tl.load(P+cols*6,valid,other=0).to(tl.float32)
    c1=tl.load(P+cols*6+1,valid,other=0).to(tl.float32)
    c2=tl.load(P+cols*6+2,valid,other=0).to(tl.float32)
    c3=tl.load(P+cols*6+3,valid,other=0).to(tl.float32)
    c4=tl.load(P+cols*6+4,valid,other=0).to(tl.float32)
    c5=tl.load(P+cols*6+5,valid,other=0).to(tl.float32)
    ps=tl.load(PS+cols,valid,other=0).to(tl.float32)
    scale=tl.load(SC+cols//RANK,valid,other=1).to(tl.float32)
    shift=tl.load(SH+cols//RANK,valid,other=0).to(tl.float32)
    h=_spine_features(x,mean,c0,c1,c2,c3,c4,c5,ps,scale,shift,LUT,valid,NL)
    hs=_half_scale(tl.max(tl.abs(h),1))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.).to(tl.int8)
    tl.store(H+rows[:,None]*N+cols[None,:],q,(rows[:,None]<M)&valid[None,:])
    tl.store(HS+rows*tl.cdiv(N,32)+tl.program_id(1),hs,rows<M)


@tr.jit(do_not_specialize=["M"])
def geometry_vjp(VALUE, COORD, P, PS, SCALE, SHIFT, RAW_SCALE, RAW_SHIFT,
                 PARTIAL, MEAN, DMEAN, RANK: tl.constexpr,
                 M, N: tl.constexpr,
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


@tr.jit(do_not_specialize=["TILES"])
def _reduce_geometry(PARTIAL, DC, DS, DH, N: tl.constexpr, TILES,
                     BLOCK: tl.constexpr):
    chart = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    component = tl.arange(0, 8)
    values = tl.load(PARTIAL + (rows[:, None]*N+chart)*8+component[None, :],
                     rows[:, None] < TILES, other=0)
    total = tl.sum(values, 0)
    tl.store(DC + chart*6+component, total, component < 6)
    tl.store(DS + chart, tl.sum(tl.where(component == 6, total, 0.), 0))
    tl.store(DH + chart, tl.sum(tl.where(component == 7, total, 0.), 0))


