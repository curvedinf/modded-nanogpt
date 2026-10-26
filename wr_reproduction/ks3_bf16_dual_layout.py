"""Optional cached BF16 transposes for original KS3 forward projections.

Both orientations come from one unchanged GS32 quantization per weight.
Matmul tile choices use the original orientation, so only physical layout and
producer writes change. Evaluation and original geometry/backward stay intact.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale
from wr_reproduction import ks3_moe as core, ks3_weight_pack as packing
from wr_reproduction.bf16_tiles import select_projection_tiles

ORIGINAL_MM = core.mm
ORIGINAL_INSTALL = core.install
ORIGINAL_REFRESH = packing.refresh_bf16
BINDINGS = {}
EXECUTED = set()


@tr.jit
def _weights_dual(BANK,Q,QT,N:tl.constexpr,D:tl.constexpr,PROJ:tl.constexpr,
                  GS32:tl.constexpr,BR:tl.constexpr=128):
    R:tl.constexpr=N if PROJ==0 else D
    C:tl.constexpr=D if PROJ==0 else N
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
    q=values.to(tl.bfloat16)
    tl.store(Q+expert*R*C+row[:,None]*C+col[None,:],q,mask)
    tl.store(QT+expert*R*C+col[None,:]*R+row[:,None],q,mask)


def key(w):
    return w.data_ptr(),tuple(w.shape),tuple(w.stride())


def attach(m):
    """Allocate once before any model/refresh graph capture; consume no RNG."""
    if hasattr(m,'ks3_bf16_up_t'):raise RuntimeError('Dual BF16 buffers already attached')
    assert tuple(m.ks3_up.shape)==(5,8,1024,768)
    assert tuple(m.ks3_down.shape)==(5,8,768,1024)
    m.register_buffer('ks3_bf16_up_t',torch.empty((5,8,768,1024),device=m.ks3_up.device,dtype=torch.bfloat16),persistent=False)
    m.register_buffer('ks3_bf16_down_t',torch.empty((5,8,1024,768),device=m.ks3_down.device,dtype=torch.bfloat16),persistent=False)
    for site in range(5):
        for label,original,transposed in [('up',m.ks3_up[site],m.ks3_bf16_up_t[site]),
                                         ('down',m.ks3_down[site],m.ks3_bf16_down_t[site])]:
            view=original.transpose(1,2)
            BINDINGS[key(view)]=(transposed,f'{label}:{site}')


@torch.no_grad()
def refresh(m):
    if not hasattr(m,'ks3_bf16_up_t'):return ORIGINAL_REFRESH(m)
    if not m.training or not core.BF16_TUNED_KERNELS:
        raise RuntimeError('Dual BF16 refresh is training-only')
    n=m.moe_bank.shape[1];d=768
    assert n==1024 and core.KS3_RANK==1
    for projection,q,qt in [(0,m.ks3_up,m.ks3_bf16_up_t),(1,m.ks3_down,m.ks3_bf16_down_t)]:
        r,c=(n,d) if projection==0 else (d,n)
        _weights_dual[(tr.cdiv(r,128)*tr.cdiv(c,32),40)](
            m.moe_bank,q,qt,n,d,projection,getattr(m,'ks3_routed_training_gs32',True),num_warps=4)
    EXECUTED.add('refresh_both_layouts')


def mm(a,w,tiles,dtype=torch.float32):
    if not core.BF16_TUNED_KERNELS:
        EXECUTED.add('original_evaluation_mm')
        return ORIGINAL_MM(a,w,tiles,dtype)
    binding=BINDINGS.get(key(w))
    if binding is None:
        EXECUTED.add('original_training_mm')
        return ORIGINAL_MM(a,w,tiles,dtype)
    cached,label=binding
    m,k=a.shape;e,kw,n=w.shape
    assert k==kw and a.is_contiguous() and a.dtype==w.dtype==cached.dtype==torch.bfloat16
    assert cached.is_contiguous() and cached.shape==w.shape
    from wr_reproduction.ks3_bf16_gemm import TILE_MODE
    # Keep the old shape/orientation tile choice, independent of new storage.
    bm,bn,bk,warps,stages,mfma=select_projection_tiles(m,n,k,w.stride(2)==1,TILE_MODE)
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    core._mm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,cached,out,tiles,m,n,k,*cached.stride(),
        BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
    EXECUTED.add('cached_'+label)
    return out


def model_install(*args,**kwargs):
    original=ORIGINAL_INSTALL(*args,**kwargs)
    class DualLayoutGPT(original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if self.ks3_precision!='bf16' or self.ks3_subspace_rank!=1 or not core.BF16_TUNED_KERNELS:
                raise ValueError('Only original tuned BF16 rank1 experts are covered')
            attach(self)
    return DualLayoutGPT


def install():
    if core.mm not in (ORIGINAL_MM,mm) or core.install not in (ORIGINAL_INSTALL,model_install):
        raise RuntimeError('Another KS3 math/model overlay is already installed')
    if packing.refresh_bf16 not in (ORIGINAL_REFRESH,refresh):
        raise RuntimeError('Another BF16 weight producer is already installed')
    core.mm=mm;core.install=model_install;packing.refresh_bf16=refresh
