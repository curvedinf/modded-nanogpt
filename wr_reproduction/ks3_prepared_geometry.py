"""KS3-only prepared coefficients and scale/shift, with original tile math.

Prepare the original GS32 integer coefficients, scale and warp transforms once
per shared weight refresh. Both consumers retain the original mathematical
expressions, reduction shape and rounding. Untuned BF16 evaluation delegates
to the original kernels and does not mutate the training cache.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale, _spine_features
from wr_reproduction.fused_fp8 import _native
from wr_reproduction import ks3_moe as core, ks3_weight_pack as packing
from wr_reproduction.ks3_moe import _chart_mean

ORIGINAL_GEOMETRY = core._geometry
ORIGINAL_VJP = core._vjp
ORIGINAL_INSTALL = core.install
ORIGINAL_REFRESH = packing.refresh_bf16
BINDINGS = {}
EXECUTED = set()
CACHE_BYTES = 5*8*1024*9*4


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


@tr.jit
def _geometry_prepared(C,G,H,TILES,M:tl.constexpr,N:tl.constexpr,HT=None,SCALE=None,AMAX=None,FP8:tl.constexpr=False,PARTIAL_AMAX:tl.constexpr=False,BM:tl.constexpr=16,GS32:tl.constexpr=True,RANK:tl.constexpr=1,NATIVE_OUT=None,NATIVE_T=None,PREPARED=None):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    e=tl.load(TILES+tl.program_id(0)*BM//128);valid=cols<N
    base=(e*(N//RANK)+cols//RANK)*(6*RANK+2)
    pp=PREPARED+(e*N+cols)*9
    c0=tl.load(pp+0,valid,other=0.);c1=tl.load(pp+1,valid,other=0.)
    c2=tl.load(pp+2,valid,other=0.);c3=tl.load(pp+3,valid,other=0.)
    c4=tl.load(pp+4,valid,other=0.);c5=tl.load(pp+5,valid,other=0.)
    ps=tl.load(pp+6,valid,other=1.)
    scale=tl.load(pp+7,valid,other=1.);shift=tl.load(pp+8,valid,other=0.)
    x=tl.load(C+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&valid[None,:],other=0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    h=_spine_features(x,_chart_mean(x,RANK),c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    if GS32:
        hs=_half_scale(tl.max(tl.abs(h),1));q=tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:,None]),-127.),127.)
        values=q*hs[:,None]
    else:
        # Opt-in dense BF16/FP8 training may retain unquantized projection values.
        values=h
    if FP8:
        out=tl.minimum(tl.maximum(values/tl.load(SCALE),-448.),448.).to(tl.float8e4nv)
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
    tl.store(H+rows[:,None]*N+cols[None,:],out,(rows[:,None]<M)&valid[None,:])

@tr.jit
def _vjp_prepared(V,C,G,DC,PART,TILES,SRC,M:tl.constexpr,N:tl.constexpr,ROWS:tl.constexpr,DCT=None,SCALE=None,AMAX=None,FP8:tl.constexpr=False,PARTIAL_AMAX:tl.constexpr=False,BM:tl.constexpr=64,RANK:tl.constexpr=1,NATIVE_OUT=None,NATIVE_T=None,PREPARED=None):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.program_id(1)*32+tl.arange(0,32)
    e=tl.load(TILES+tl.program_id(0)*BM//128);mask=(rows[:,None]<M)&(cols[None,:]<N)
    real=tl.load(SRC+rows,rows<M,other=ROWS)<ROWS
    value=tl.load(V+rows[:,None]*N+cols[None,:],mask,other=0.)
    original=tl.load(C+rows[:,None]*N+cols[None,:],mask,other=0.)
    x=tl.minimum(tl.maximum(original,-65504.),65504.)
    base=(e*(N//RANK)+cols//RANK)*(6*RANK+2)
    pp=PREPARED+(e*N+cols)*9
    ps=tl.load(pp+6,cols<N,other=1.)
    c0=tl.load(pp+0,cols<N,other=0.)*ps;c1=tl.load(pp+1,cols<N,other=0.)*ps
    c2=tl.load(pp+2,cols<N,other=0.)*ps;c3=tl.load(pp+3,cols<N,other=0.)*ps
    c4=tl.load(pp+4,cols<N,other=0.)*ps;c5=tl.load(pp+5,cols<N,other=0.)*ps
    rs=tl.load(G+base+6*RANK,cols<N,other=0.);rh=tl.load(G+base+6*RANK+1,cols<N,other=0.)
    scale=tl.load(pp+7,cols<N,other=1.);shift=tl.load(pp+8,cols<N,other=0.)
    t=4.*libdevice.tanh(_chart_mean(x,RANK)*.25);z=libdevice.tanh(scale[None,:]*(t+shift[None,:]))
    cubic=2*c0-2*c1+c2+c3;quadratic=-3*c0+3*c1-2*c2-c3
    spine=((cubic*t+quadratic)*t+c2)*t+c0;warped=((4*c5*z+2*c4)*z-3*c5)*z-c4
    dp=tl.where(mask&real[:,None],value*2.*tl.maximum(x+spine+warped,0.),0.)
    dz=((12*c5*z+4*c4)*z-3*c5)*(1.-z*z)
    dm=dp*((3*cubic*t+2*quadratic)*t+c2+dz*scale)*(1.-t*t*.0625)
    dc=tl.where((original>=-65504.)&(original<=65504.),dp+_chart_mean(dm,RANK),0.)
    if FP8:
        out=tl.minimum(tl.maximum(dc/tl.load(SCALE),-57344.),57344.).to(tl.float8e5)
        tl.store(DCT+cols[None,:]*M+rows[:,None],out,mask)
        if NATIVE_OUT is not None:
            native=_native(out.to(tl.uint8,bitcast=True),True,True)
            tl.store(NATIVE_OUT+rows[:,None]*N+cols[None,:],native,mask)
            if NATIVE_T is not None:tl.store(NATIVE_T+cols[None,:]*M+rows[:,None],native,mask)
        largest=tl.max(tl.max(tl.abs(dc),1),0)
        if PARTIAL_AMAX:
            tl.store(AMAX+tl.program_id(0)*tl.cdiv(N,32)+tl.program_id(1),largest)
        else:tl.atomic_max(AMAX,largest)
    else:out=dc
    tl.store(DC+rows[:,None]*N+cols[None,:],out,mask)
    off=(tl.program_id(0)*N+cols)*8;t2=t*t;t3=t2*t
    tl.store(PART+off,tl.sum(dp*(2*t3-3*t2+1),0),cols<N)
    tl.store(PART+off+1,tl.sum(dp*(-2*t3+3*t2),0),cols<N)
    tl.store(PART+off+2,tl.sum(dp*(t3-2*t2+t),0),cols<N)
    tl.store(PART+off+3,tl.sum(dp*(t3-t2),0),cols<N)
    tl.store(PART+off+4,tl.sum(dp*(2*z*z-1),0),cols<N)
    tl.store(PART+off+5,tl.sum(dp*(4*z*z*z-3*z),0),cols<N)
    tl.store(PART+off+6,tl.sum(dp*dz*(t+shift)*tl.sigmoid(rs),0),cols<N)
    tl.store(PART+off+7,tl.sum(dp*dz*scale*.5*(1.-libdevice.tanh(rh)*libdevice.tanh(rh)),0),cols<N)


def key(g):
    return g.data_ptr(),tuple(g.shape),tuple(g.stride())


def attach(m):
    if hasattr(m,'ks3_prepared_geometry'):raise RuntimeError('Geometry cache already attached')
    assert tuple(m.ks3_geometry.shape)==(5,8,1024,8)
    m.register_buffer('ks3_prepared_geometry',torch.empty((5,8,1024,9),device=m.ks3_geometry.device,dtype=torch.float32),persistent=False)
    bind(m)


def bind(m):
    """Bind after replicated Adam moves parameters into its flat storage.

    Call before the first warmup/graph capture. This updates only Python lookup
    keys; neither parameter storage nor graph-owned cache storage is replaced.
    """
    for site in range(5):BINDINGS[key(m.ks3_geometry[site])]=(m.ks3_prepared_geometry[site],site)


@torch.no_grad()
def prepare(m):
    assert core.KS3_RANK==1 and m.training and core.BF16_TUNED_KERNELS
    _prepare[(tr.cdiv(m.ks3_geometry.numel()//8,256),)](m.ks3_geometry,m.ks3_prepared_geometry,m.ks3_geometry.numel()//8,num_warps=4)
    EXECUTED.add('prepare_all_five_sites')


@torch.no_grad()
def refresh(m):
    ORIGINAL_REFRESH(m)
    if hasattr(m,'ks3_prepared_geometry'):prepare(m)


class PreparedKernel:
    def __init__(self,name,original,prepared,g_index):
        self.name,self.original,self.prepared,self.g_index=name,original,prepared,g_index
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            if not core.BF16_TUNED_KERNELS:
                EXECUTED.add('original_evaluation_'+self.name)
                return self.original[grid](*args,**kwargs)
            if core.KS3_RANK!=1 or kwargs.get('RANK',1)!=1 or kwargs.get('FP8',False):
                raise ValueError('Prepared geometry covers original tuned BF16 rank1 only')
            cache,site=BINDINGS[key(args[self.g_index])]
            EXECUTED.add('cached_'+self.name+':'+str(site))
            return self.prepared[grid](*args,**kwargs,PREPARED=cache)
        return launch


GEOMETRY = PreparedKernel('geometry',ORIGINAL_GEOMETRY,_geometry_prepared,1)
VJP = PreparedKernel('vjp',ORIGINAL_VJP,_vjp_prepared,2)


def model_install(*args,**kwargs):
    original=ORIGINAL_INSTALL(*args,**kwargs)
    class PreparedGeometryGPT(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if self.ks3_precision!='bf16' or self.ks3_subspace_rank!=1 or not core.BF16_TUNED_KERNELS:
                raise ValueError('Prepared geometry requires original tuned BF16 rank1')
            attach(self)
    return PreparedGeometryGPT


def install():
    if core._geometry not in (ORIGINAL_GEOMETRY,GEOMETRY) or core._vjp not in (ORIGINAL_VJP,VJP):
        raise RuntimeError('Another geometry kernel overlay is installed')
    if core.install not in (ORIGINAL_INSTALL,model_install) or packing.refresh_bf16 not in (ORIGINAL_REFRESH,refresh):
        raise RuntimeError('Another model or weight producer overlay is installed')
    core._geometry=GEOMETRY;core._vjp=VJP;core.install=model_install;packing.refresh_bf16=refresh
