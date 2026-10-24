"""One-pass GS32 dequantization without repeated full-matrix temporaries."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _unpack(C, S, O, rows, columns, cs0, cs1, ss0, ss1,
            BLOCK: tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=offsets//columns
    col=offsets%columns
    valid=row<rows
    codes=tl.load(C+row*cs0+col*cs1,valid,other=0).to(tl.float32)
    scales=tl.load(S+row*ss0+(col//32)*ss1,valid,other=0).to(tl.float32)
    tl.store(O+offsets,codes*scales,valid)


def unpack_gs32(codes,scales,dtype=torch.float32):
    if dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('GS32 unpack output must be FP32 or BF16')
    if (codes.ndim!=2 or scales.ndim!=2 or codes.dtype!=torch.int8
            or codes.device!=scales.device or scales.shape!=(codes.shape[0],tr.cdiv(codes.shape[1],32))):
        raise ValueError('GS32 unpacking requires aligned two-dimensional codes and scales')
    if not codes.is_cuda:
        return (codes.float()*scales.float().repeat_interleave(32,-1)[...,:codes.shape[-1]]).to(dtype)
    output=torch.empty(codes.shape,device=codes.device,dtype=dtype)
    if output.numel():
        _unpack[(tr.cdiv(output.numel(),1024),)](
            codes,scales,output,*codes.shape,*codes.stride(),*scales.stride(),BLOCK=1024)
    return output
