"""Batched GS32-to-FP8 weight producers with exact delayed-scale semantics."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale
from wr_reproduction.fused_fp8 import _native


@tr.jit
def _weights(BANK,Q,QT,SCALE,PART,N:tl.constexpr,D:tl.constexpr,PROJ:tl.constexpr,
             GS32:tl.constexpr,BR:tl.constexpr=128,FP8:tl.constexpr=True,NQ=None,NQT=None):
    R:tl.constexpr=N if PROJ==0 else D
    C:tl.constexpr=D if PROJ==0 else N
    T:tl.constexpr=tl.cdiv(R,BR)*tl.cdiv(C,32)
    tile=tl.program_id(0);expert=tl.program_id(1)
    row=(tile//tl.cdiv(C,32))*BR+tl.arange(0,BR)
    col=(tile%tl.cdiv(C,32))*32+tl.arange(0,32)
    mask=(row[:,None]<R)&(col[None,:]<C)
    slot=(expert//8)*16+PROJ*8+expert%8
    if PROJ==0:offset=row[:,None]*D+col[None,:]
    else:offset=col[None,:]*D+row[:,None]
    a=tl.load(BANK+slot*N*D+offset,mask,other=0.).to(tl.float32)
    if GS32:
        s=_half_scale(tl.max(tl.abs(a),1))
        values=tl.minimum(tl.maximum(libdevice.nearbyint(a/s[:,None]),-127.),127.)*s[:,None]
    else:values=a
    if FP8:q=tl.minimum(tl.maximum(values/tl.load(SCALE+slot),-448.),448.).to(tl.float8e4nv)
    else:q=values
    tl.store(Q+expert*R*C+row[:,None]*C+col[None,:],q,mask)
    if FP8:
        tl.store(QT+expert*R*C+col[None,:]*R+row[:,None],q,mask)
        tl.store(PART+slot*T+tile,tl.max(tl.max(tl.abs(values),1),0))
        if NQ is not None:
            native=_native(q.to(tl.uint8,bitcast=True),False,False)
            tl.store(NQ+expert*R*C+row[:,None]*C+col[None,:],native,mask)
            tl.store(NQT+expert*R*C+col[None,:]*R+row[:,None],native,mask)


@torch.no_grad()
def refresh(m,exact):
    n=m.ks3_geometry.shape[2];d=768
    bank=m.moe_bank.view(5,2,8,n,d)
    source=bank.float().abs().amax((-2,-1)) if exact else m.ks3_weight_amax
    m.ks3_weight_scales.copy_(source.clamp_min(1e-12)/448.)
    for projection,q,qt in ((0,m.ks3_f8_up,m.ks3_f8_up_t),
                            (1,m.ks3_f8_down,m.ks3_f8_down_t)):
        r,c=(n,d) if projection==0 else (d,n)
        tiles=tr.cdiv(r,128)*tr.cdiv(c,32)
        assert tiles==m.ks3_pack_partial_amax.shape[-1]
        _weights[(tiles,40)](bank,q,qt,m.ks3_weight_scales,m.ks3_pack_partial_amax,n,d,
                            projection,getattr(m,'ks3_training_gs32',True),
                            NQ=getattr(m,'ks3_native_up' if projection==0 else 'ks3_native_down',None)
                                if getattr(m,'ks3_prepared_native_weights',False) else None,
                            NQT=getattr(m,'ks3_native_up_t' if projection==0 else 'ks3_native_down_t',None)
                                if getattr(m,'ks3_prepared_native_weights',False) else None,num_warps=4)
    m.ks3_weight_amax.copy_(m.ks3_pack_partial_amax.amax(-1))


@torch.no_grad()
def refresh_bf16(m):
    n=m.moe_bank.shape[1];d=768
    for projection,q in ((0,m.ks3_up),(1,m.ks3_down)):
        r,c=(n,d) if projection==0 else (d,n)
        _weights[(tr.cdiv(r,128)*tr.cdiv(c,32),40)](
            m.moe_bank,q,None,None,None,n,d,projection,True,FP8=False,num_warps=4)
