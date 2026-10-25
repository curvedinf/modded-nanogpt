"""Preserve original FP8 cache bytes in native MI300 FP8 matrix products.

The runtime reads source E4M3FN/E5M2 bytes directly. The fused kernel interprets
finite bytes as half-valued FNUZ operands in registers and restores the factor
four in the product scale. No tensor requantization or conversion pass occurs.
encode_native is a standalone oracle utility used by the exhaustive format test.
"""
import torch

FORMATS={torch.float8_e4m3fn:torch.float8_e4m3fnuz,
         torch.float8_e5m2:torch.float8_e5m2fnuz}

def encode_native(quantized):
    if quantized.dtype not in FORMATS:raise TypeError('Expected original E4M3FN or E5M2 FP8 tensor')
    return (quantized.float()*.5).to(FORMATS[quantized.dtype])

def mm(a,b,scale_a,scale_b,*,out_dtype=torch.float32,use_fast_accum=False):
    if not a.is_cuda or not b.is_cuda or not torch.version.hip:
        raise RuntimeError('This adapter requires native ROCm GPU execution')
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('Invalid GEMM shape')
    from wr_reproduction.fused_fp8 import scaled_mm_op,_OUTPUT_DTYPES
    # Source cache bytes are read directly by the hardware kernel. There is no
    # tensor conversion, requantization, or temporary native-format cache.
    sa=torch.as_tensor(scale_a,device=a.device,dtype=torch.float32)
    sb=torch.as_tensor(scale_b,device=b.device,dtype=torch.float32)
    return scaled_mm_op(a,b,sa,sb,_OUTPUT_DTYPES.index(out_dtype),use_fast_accum)
