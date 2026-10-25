"""KS3-only MI300 FP8 tile dispatch, numerically checked at 16k/49k rows.

The immutable record control retains its original port's launch configuration.
Only launch tiles, warp count and MFMA shape change here; quantization,
accumulation types, scale application and invalid-value handling are unchanged.
"""
import torch
import triton as tr
from wr_reproduction.fused_fp8 import _gemm,_dw
from wr_reproduction.native_fp8_tiles import projection_tiles,weight_gradient_tiles

TILE_MODE='legacy'
NATIVE_DTYPES=(torch.float8_e4m3fnuz,torch.float8_e5m2fnuz)

def _run_tiled(a,b,sa,sb,tiles,n,config,out_dtype=torch.float32):
    bm,bn,bk,warps,stages,mfma=config;m,k=a.shape
    out=torch.empty((m,n),device=a.device,dtype=out_dtype)
    _gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a.view(torch.uint8),b.view(torch.uint8),out,out,out,out,sa,
        sa,sb,sa,sa,sa,tiles,m,n,k,*a.stride(),*b.stride(),a.dtype in (torch.float8_e5m2,torch.float8_e5m2fnuz),
        b.dtype==torch.float8_e5m2,0,True,False,False,False,BM=bm,BN=bn,BK=bk,num_warps=warps,
        PREPARED_B=b.dtype==torch.float8_e4m3fnuz,
        PREPARED_A=a.dtype in (torch.float8_e4m3fnuz,torch.float8_e5m2fnuz),
        num_stages=stages,enable_fp_fusion=False,matrix_instr_nonkdim=mfma)
    return out

def _dw_tiled(a,b,offs,sa,sb,e,out,flag,config):
    bm,bn,bk,warps,stages,mfma=config;h,n=a.shape[0],b.shape[0]
    _dw[(tr.cdiv(h,bm),tr.cdiv(n,bn),e)](a.view(torch.uint8),b.view(torch.uint8),out,offs,sa,sb,flag,
        h,n,*a.stride(),*b.stride(),a.dtype in (torch.float8_e5m2,torch.float8_e5m2fnuz),b.dtype in (torch.float8_e5m2,torch.float8_e5m2fnuz),
        PREPARED_A=a.dtype in (torch.float8_e4m3fnuz,torch.float8_e5m2fnuz),
        PREPARED_B=b.dtype in (torch.float8_e4m3fnuz,torch.float8_e5m2fnuz),
        BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False,matrix_instr_nonkdim=mfma)

def run(a,b,sa,sb,*,out_dtype=torch.float32,tiles,n):
    if a.dtype not in (torch.float8_e4m3fn,torch.float8_e5m2,torch.float8_e4m3fnuz,torch.float8_e5m2fnuz) or b.dtype not in (torch.float8_e4m3fn,torch.float8_e4m3fnuz):
        raise TypeError('KS3 FP8 operands required')
    # Keep the small-/large-batch winner for each projection shape; the
    # alternative large-batch input tile was within one percent.
    config=projection_tiles(a.shape[0],n,a.shape[1],a.dtype in NATIVE_DTYPES and b.dtype in NATIVE_DTYPES,TILE_MODE)
    out=_run_tiled(a,b,sa,sb,tiles,n,config,out_dtype)
    return out,None,None,None

def grouped_dw(a,b,offs,sa,sb,experts,out,acc_flag):
    config=weight_gradient_tiles(a.shape[1],a.shape[0],b.shape[0],a.dtype in NATIVE_DTYPES and b.dtype in NATIVE_DTYPES,TILE_MODE)
    _dw_tiled(a,b,offs,sa,sb,experts,out,acc_flag,config)
