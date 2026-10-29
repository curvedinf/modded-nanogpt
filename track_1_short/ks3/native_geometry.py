"""H100 native-tanh geometry experiment.

Prepared kernels are derived from track_1_short.ks3.prepared_geometry;
FP16 forward expressions are from the preserved vendor _spine_features.
Only tanh evaluation changes: PTX tanh.approx.f32, sm_75+, documented maximum
relative error 2**-11. Original libdevice kernels and evaluation stay available.
https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh
"""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from track_1_short.ks3.quantization import _half_scale
from track_1_short.ks3.core import _chart_mean
from track_1_short.ks3.native import _native

@tr.jit
def native_tanh(x):
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)

@tr.jit
def _spine_features(x, mean, c0, c1, c2, c3, c4, c5, ps, scale, shift, LUT, valid, NL:tl.constexpr,CUBIC_ONLY:tl.constexpr=False):
    tl.static_assert(NL == 1)
    a=2*c0-2*c1+c2+c3;b=-3*c0+3*c1-2*c2-c3
    th=(mean*.25).to(tl.float16)
    t=(4*native_tanh(th.to(tl.float32))).to(tl.float16)
    # Each polynomial stage rounds to FP16; tanh uses FP32 internally.
    poly=(a[None,:].to(tl.float16)*t+b[None,:].to(tl.float16)).to(tl.float16)
    poly=(poly*t+c2[None,:].to(tl.float16)).to(tl.float16)
    poly=(poly*t+c0[None,:].to(tl.float16)).to(tl.float16)
    if CUBIC_ONLY:warped=tl.full((x.shape[0],x.shape[1]),0.,tl.float16)
    else:
        warp=(scale[None,:]*(t.to(tl.float32)+shift[None,:])).to(tl.float16)
        z=native_tanh(warp.to(tl.float32)).to(tl.float16)
        warped=(4*c5[None,:].to(tl.float16)*z+2*c4[None,:].to(tl.float16)).to(tl.float16)
        warped=(warped*z-3*c5[None,:].to(tl.float16)).to(tl.float16)
        warped=(warped*z-c4[None,:].to(tl.float16)).to(tl.float16)
    h=tl.maximum((x+(poly.to(tl.float32)+warped.to(tl.float32))*ps[None,:]).to(tl.float16),0.)
    h=(h*h).to(tl.float16).to(tl.float32)
    return tl.where(valid[None,:],h,0.)


@tr.jit
def _geometry_prepared(C,G,H,TILES,M:tl.constexpr,N:tl.constexpr,HT=None,SCALE=None,AMAX=None,FP8:tl.constexpr=False,PARTIAL_AMAX:tl.constexpr=False,BM:tl.constexpr=16,GS32:tl.constexpr=True,RANK:tl.constexpr=1,NATIVE_OUT=None,NATIVE_T=None,PREPARED=None,H_DESC=None,HT_DESC=None,CUBIC_ONLY:tl.constexpr=False):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    e=tl.load(TILES+tl.program_id(0)*BM//128);valid=cols<N
    base=(e*(N//RANK)+cols//RANK)*(6*RANK+2)
    pp=PREPARED+(e*N+cols)*9
    c0=tl.load(pp+0,valid,other=0.);c1=tl.load(pp+1,valid,other=0.)
    c2=tl.load(pp+2,valid,other=0.);c3=tl.load(pp+3,valid,other=0.)
    c4=tl.load(pp+4,valid,other=0.);c5=tl.load(pp+5,valid,other=0.)
    ps=tl.load(pp+6,valid,other=1.)
    scale=tl.load(pp+7,valid,other=1.);shift=tl.load(pp+8,valid,other=0.)
    x=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.).to(tl.float32)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    h=_spine_features(x,_chart_mean(x,RANK),c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1,CUBIC_ONLY)
    if GS32:
        hs=_half_scale(tl.max(tl.abs(h),1));q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.)
        values=q*hs[:,None]
    else:
        # Opt-in dense BF16/FP8 training may retain unquantized projection values.
        values=h
    if FP8:
        out=tl.minimum(tl.maximum(values/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
        if HT_DESC is not None:
            HT_DESC.store([tl.program_id(1)*32,tl.program_id(0)*BM],tl.trans(out))
        else:
            tl.store(HT+cols[None,:]*M+rows[:,None],out,(rows[:,None]<M)&valid[None,:])
        if NATIVE_OUT is not None:
            native=_native(out.to(tl.uint8,bitcast=True),False,False)
            tl.store(NATIVE_OUT+rows[:,None]*N+cols[None,:],native,(rows[:,None]<M)&valid[None,:])
            if NATIVE_T is not None:tl.store(NATIVE_T+cols[None,:]*M+rows[:,None],native,(rows[:,None]<M)&valid[None,:])
        if AMAX is not None:
            largest=tl.max(tl.max(tl.abs(values),1),0)
            if PARTIAL_AMAX:
                tl.store(AMAX+tl.program_id(0)*tl.cdiv(N,32)+tl.program_id(1),largest)
            else:tl.atomic_max(AMAX,largest)
    else:out=values
    if H_DESC is not None:
        H_DESC.store([tl.program_id(0)*BM,tl.program_id(1)*32],out)
    else:
        tl.store(H+rows[:,None]*N+cols[None,:],out,(rows[:,None]<M)&valid[None,:])

@tr.jit
def _tile_chart_mean(x,RANK:tl.constexpr):
    if RANK==1:
        return x
    else:
        grouped=tl.reshape(x,(x.shape[0],x.shape[1]//RANK,RANK))
        mean=tl.sum(grouped,2)/RANK
        return tl.reshape(tl.broadcast_to(mean[:,:,None],grouped.shape),x.shape)


@tr.jit
def _vjp_prepared(V,C,G,DC,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,ROWS:tl.constexpr,DCT=None,SCALE=None,AMAX=None,FP8:tl.constexpr=False,PARTIAL_AMAX:tl.constexpr=False,BM:tl.constexpr=64,RANK:tl.constexpr=1,NATIVE_OUT=None,NATIVE_T=None,PREPARED=None,DC_DESC=None,DCT_DESC=None,CUBIC_ONLY:tl.constexpr=False,COMPACT_PART:tl.constexpr=False,DPRE_E4M3:tl.constexpr=False,BC:tl.constexpr=32):
    tl.static_assert(not COMPACT_PART or (CUBIC_ONLY and RANK==1))
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*BC+tl.arange(0,BC)
    e=tl.load(TILES+tl.program_id(0)*BM//128);mask=(rows[:,None]<M)&(cols[None,:]<N)
    real=tl.load(SRC+rows,rows<M,other=ROWS)<ROWS
    value=tl.load(V+rows[:,None]*N+cols[None,:],mask,other=0.).to(tl.float32)
    original=tl.load(C+rows[:,None]*N+cols[None,:],mask,other=0.).to(tl.float32)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    base=(e*(N//RANK)+cols//RANK)*(6*RANK+2)
    pp=PREPARED+(e*N+cols)*9
    ps=tl.load(pp+6,cols<N,other=1.)
    c0=tl.load(pp+0,cols<N,other=0.)*ps;c1=tl.load(pp+1,cols<N,other=0.)*ps
    c2=tl.load(pp+2,cols<N,other=0.)*ps;c3=tl.load(pp+3,cols<N,other=0.)*ps
    c4=tl.load(pp+4,cols<N,other=0.)*ps;c5=tl.load(pp+5,cols<N,other=0.)*ps
    rs=tl.load(G+base+6*RANK,cols<N,other=0.);rh=tl.load(G+base+6*RANK+1,cols<N,other=0.)
    scale=tl.load(pp+7,cols<N,other=1.);shift=tl.load(pp+8,cols<N,other=0.)
    t=4.*native_tanh(_tile_chart_mean(x,RANK)*.25)
    cubic=2*c0-2*c1+c2+c3;quadratic=-3*c0+3*c1-2*c2-c3
    spine=((cubic*t+quadratic)*t+c2)*t+c0
    if CUBIC_ONLY:
        warped=tl.full((BM,BC),0.,tl.float32);dz=warped
    else:
        z=native_tanh(scale[None,:]*(t+shift[None,:]))
        warped=((4*c5*z+2*c4)*z-3*c5)*z-c4
        dz=((12*c5*z+4*c4)*z-3*c5)*(1.-z*z)
    dp=tl.where(mask&real[:,None],value*2.*tl.maximum(x+spine+warped,0.),0.)
    dm=dp*((3*cubic*t+2*quadratic)*t+c2+dz*scale)*(1.-t*t*.0625)
    dc=tl.where((original>=-65504.)&(original<=65504.),dp+_tile_chart_mean(dm,RANK),0.)
    if FP8:
        if DPRE_E4M3:
            tl.static_assert(NATIVE_OUT is None)
            out=tl.minimum(tl.maximum(dc/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
        else:
            out=tl.minimum(tl.maximum(dc/tl.load(SCALE),-57344.),57344.).to(tl.float8e5)
        if DCT_DESC is not None:
            DCT_DESC.store([tl.program_id(1)*BC,tl.program_id(0)*BM],tl.trans(out))
        else:
            tl.store(DCT+cols[None,:]*M+rows[:,None],out,mask)
        if NATIVE_OUT is not None:
            native=_native(out.to(tl.uint8,bitcast=True),True,True)
            tl.store(NATIVE_OUT+rows[:,None]*N+cols[None,:],native,mask)
            if NATIVE_T is not None:tl.store(NATIVE_T+cols[None,:]*M+rows[:,None],native,mask)
        largest=tl.max(tl.max(tl.abs(dc),1),0)
        if PARTIAL_AMAX:
            tl.store(AMAX+tl.program_id(0)*tl.cdiv(N,BC)+tl.program_id(1),largest)
        else:tl.atomic_max(AMAX,largest)
    else:out=dc
    if DC_DESC is not None:
        DC_DESC.store([tl.program_id(0)*BM,tl.program_id(1)*BC],out)
    else:
        tl.store(DC+rows[:,None]*N+cols[None,:],out,mask)
    off=(tl.program_id(0)*N+cols)*(4 if COMPACT_PART else 8);t2=t*t;t3=t2*t
    tl.store(PART+off,tl.sum(dp*(2*t3-3*t2+1),0),cols<N)
    tl.store(PART+off+1,tl.sum(dp*(-2*t3+3*t2),0),cols<N)
    tl.store(PART+off+2,tl.sum(dp*(t3-2*t2+t),0),cols<N)
    tl.store(PART+off+3,tl.sum(dp*(t3-t2),0),cols<N)
    if CUBIC_ONLY and not COMPACT_PART:
        tl.store(PART+off+4,0.,cols<N);tl.store(PART+off+5,0.,cols<N)
        tl.store(PART+off+6,0.,cols<N);tl.store(PART+off+7,0.,cols<N)
    elif not CUBIC_ONLY:
        tl.store(PART+off+4,tl.sum(dp*(2*z*z-1),0),cols<N)
        tl.store(PART+off+5,tl.sum(dp*(4*z*z*z-3*z),0),cols<N)
        tl.store(PART+off+6,tl.sum(dp*dz*(t+shift)*tl.sigmoid(rs),0),cols<N)
        tl.store(PART+off+7,tl.sum(dp*dz*scale*.5*(1.-native_tanh(rh)*native_tanh(rh)),0),cols<N)


@tr.jit
def reduce_cubic(P,O,OFFS,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BM:tl.constexpr=128):
    e=tl.program_id(0);n=tl.program_id(1)
    lo=0
    if e>0:lo=tl.load(OFFS+e-1)//BM
    hi=tl.load(OFFS+e)//BM
    r=lo+tl.arange(0,B);j=tl.arange(0,4)
    v=tl.load(P+(r[:,None]*N+n)*4+j[None,:],(r[:,None]<hi)&(r[:,None]<T),other=0.)
    total=tl.sum(v,0)
    tl.store(O+(e*N+n)*8+j,total)
    tl.store(O+(e*N+n)*8+j+4,0.)
