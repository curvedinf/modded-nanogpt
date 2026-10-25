"""Opt-in MI300 tile dispatch; same BF16 operands and FP32 accumulation."""
import torch
import triton as tr
from wr_reproduction.bf16_tiles import select_projection_tiles

TILE_MODE='mi300-v1'


def run(a,w,tiles,dtype=torch.float32):
    from wr_reproduction.ks3_moe import _mm
    m,k=a.shape;e,kw,n=w.shape
    assert k==kw and a.is_contiguous()
    assert a.dtype==w.dtype==torch.bfloat16
    bm,bn,bk,warps,stages,mfma=select_projection_tiles(m,n,k,w.stride(2)==1,TILE_MODE)
    out=torch.empty((m,n),device=a.device,dtype=dtype)
    _mm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a,w,out,tiles,m,n,k,*w.stride(),
        BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,matrix_instr_nonkdim=mfma)
    return out


def dw(a,b,out,offs):
    from wr_reproduction.ks3_moe import _dw
    e,h,d=out.shape
    _dw[(tr.cdiv(h,128),tr.cdiv(d,128),e)](a,b,out,offs,h,d,False,
        BM=128,BN=128,BK=64,num_warps=8,num_stages=1,matrix_instr_nonkdim=32)
