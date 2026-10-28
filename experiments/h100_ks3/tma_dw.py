"""KS3-only launch tuning for PR393's unchanged FP8 weight-gradient kernel."""
import os
import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from track_1_short.perf.kernels.moe_mlp import grouped_dw_kernel


def run(a, b, out, offs, sa, sb, flag, tile=None, stages=None):
    h, m = a.shape
    n, m2 = b.shape
    e = offs.numel()
    bm, bn = map(int, (tile or os.environ['KS3_FP8_DW_TILE']).split('x'))
    stages = stages or int(os.environ.get('KS3_FP8_DW_STAGES', '3'))
    assert (bm, bn) in ((64, 128), (64, 256), (128, 128), (128, 256), (256, 128))
    assert m == m2 and m % 128 == 0 and h % bm == n % bn == 0
    assert a.is_contiguous() and b.is_contiguous() and out.is_contiguous()
    assert out.shape == (e, h, n) and out.dtype == torch.bfloat16
    ad = TensorDescriptor.from_tensor(a, [bm, 128])
    bd = TensorDescriptor.from_tensor(b, [bn, 128])
    cd = TensorDescriptor.from_tensor(out.view(e * h, n), [bm, bn // 2])
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grouped_dw_kernel[(min(sms, e * (h // bm) * (n // bn)),)](
        ad, bd, cd, offs, sa, sb, flag, h, n, E=e,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
        NUM_SMS=sms, num_stages=stages, num_warps=4 if bm == 64 else 8)

