"""Fuse the original BF16 up product with original FP16 geometry and GS32 packing.

The original GEMM tile selection, coefficient packing and nonlinear equations
are retained. FP32 coordinates are still stored for the unchanged backward.
Untuned/evaluation execution delegates to the original forward in its entirety.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice
from ks3_research.vendor.ks3_int8 import _half_scale, _spine_features
from wr_reproduction import ks3_moe as core
from wr_reproduction.bf16_tiles import select_projection_tiles

ORIGINAL_FORWARD = core.forward
EXECUTED_MODES = set()


@tr.jit
def _up_geometry(A, W, C, G, H, TILES, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
                 W0:tl.constexpr, W1:tl.constexpr, W2:tl.constexpr,
                 BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    r = tl.program_id(0)*BM+tl.arange(0, BM)
    c = tl.program_id(1)*BN+tl.arange(0, BN)
    kk = tl.arange(0, BK)
    e = tl.load(TILES+tl.program_id(0)*BM//128)
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base*BK+kk
        a = tl.load(A+r[:, None]*K+k[None, :], (r[:, None]<M)&(k[None, :]<K), other=0.)
        w = tl.load(W+e*W0+k[:, None]*W1+c[None, :]*W2, (k[:, None]<K)&(c[None, :]<N), other=0.)
        acc = tl.dot(a, w, acc)
    tl.store(C+r[:, None]*N+c[None, :], acc, (r[:, None]<M)&(c[None, :]<N))
    valid = c<N
    gp = G+(e*N+c)*8
    c0=tl.load(gp+0,valid,other=0.);c1=tl.load(gp+1,valid,other=0.)
    c2=tl.load(gp+2,valid,other=0.);c3=tl.load(gp+3,valid,other=0.)
    c4=tl.load(gp+4,valid,other=0.);c5=tl.load(gp+5,valid,other=0.)
    amax=tl.maximum(tl.maximum(tl.abs(c0),tl.abs(c1)),tl.maximum(tl.maximum(tl.abs(c2),tl.abs(c3)),tl.maximum(tl.abs(c4),tl.abs(c5))))
    ps=_half_scale(amax)
    c0=libdevice.nearbyint(c0/ps);c1=libdevice.nearbyint(c1/ps);c2=libdevice.nearbyint(c2/ps)
    c3=libdevice.nearbyint(c3/ps);c4=libdevice.nearbyint(c4/ps);c5=libdevice.nearbyint(c5/ps)
    c0=tl.minimum(tl.maximum(c0,-127.),127.);c1=tl.minimum(tl.maximum(c1,-127.),127.)
    c2=tl.minimum(tl.maximum(c2,-127.),127.);c3=tl.minimum(tl.maximum(c3,-127.),127.)
    c4=tl.minimum(tl.maximum(c4,-127.),127.);c5=tl.minimum(tl.maximum(c5,-127.),127.)
    raw=tl.load(gp+6,valid,other=0.);rh=tl.load(gp+7,valid,other=0.)
    scale=tl.where(raw>20.,raw,libdevice.log1p(libdevice.exp(raw)))+1e-4
    shift=.5*libdevice.tanh(rh)
    x=tl.where((r[:, None]<M)&valid[None, :],acc,0.)
    x=tl.minimum(tl.maximum(x,-65504.),65504.)
    h=_spine_features(x,x,c0,c1,c2,c3,c4,c5,ps,scale,shift,G,valid,1)
    # A GEMM tile contains several independent 32-feature quantization groups.
    grouped=tl.reshape(h,(BM,BN//32,32))
    hs=_half_scale(tl.max(tl.abs(grouped),2))
    q=tl.minimum(tl.maximum(libdevice.nearbyint(grouped/hs[:,:,None]),-127.),127.)
    values=tl.reshape(q*hs[:,:,None],(BM,BN))
    tl.store(H+r[:,None]*N+c[None,:],values,(r[:,None]<M)&valid[None,:])


@torch.library.custom_op('wr_ks3_fused_geometry_forward::forward', mutates_args=())
def forward(rows:torch.Tensor, src:torch.Tensor, tiles:torch.Tensor,
            up:torch.Tensor, down:torch.Tensor, g:torch.Tensor,
            gs32:bool=True) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if not gs32 or core.KS3_RANK != 1:
        raise ValueError('This overlay requires original GS32 independent rank1 experts')
    if not core.BF16_TUNED_KERNELS:
        EXECUTED_MODES.add('original_evaluation')
        return ORIGINAL_FORWARD(rows,src,tiles,up,down,g,gs32)
    EXECUTED_MODES.add('training_fused_up_geometry')
    from wr_reproduction.ks3_bf16_gemm import TILE_MODE
    m,d,n=src.numel(),rows.shape[1],up.shape[1]
    if (up.shape[0],n,d)!=(8,1024,768):
        raise ValueError('Only the actual incumbent routed expert shape is authorized')
    x=torch.empty((m,d),device=rows.device,dtype=torch.bfloat16)
    core._pack[(tr.cdiv(m,128),tr.cdiv(d,32))](
        rows,x,m,d,*rows.stride(),SRC=src,INPUT_ROWS=rows.shape[0],
        ROWS_PER_TILE=128,GS32=True,num_warps=4)
    w=up.transpose(1,2)
    bm,bn,bk,warps,stages,mfma=select_projection_tiles(m,n,d,w.stride(2)==1,TILE_MODE)
    c=torch.empty((m,n),device=rows.device,dtype=torch.float32)
    h=torch.empty((m,n),device=rows.device,dtype=torch.bfloat16)
    _up_geometry[(tr.cdiv(m,bm),tr.cdiv(n,bn))](
        x,w,c,g,h,tiles,m,n,d,*w.stride(),BM=bm,BN=bn,BK=bk,
        num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
    y=core.mm(h,down.transpose(1,2),tiles,dtype=rows.dtype)
    return y,x,c,h


@forward.register_fake
def _(rows,src,tiles,up,down,g,gs32=True):
    m,d,n=src.numel(),rows.shape[1],up.shape[1]
    return rows.new_empty((m,d)),rows.new_empty((m,d)), \
        rows.new_empty((m,n),dtype=torch.float32),rows.new_empty((m,n))


def install():
    if core.forward not in (ORIGINAL_FORWARD,forward):
        raise RuntimeError('A different KS3 forward overlay is installed')
    core.forward=forward
