"""Install the fixed H100 KS3 expert blocks and their optimizer state."""
import torch
import triton as tr
from track_1_short.ks3 import core as ks, bf16_gemm as gemm
from track_1_short.ks3.bf16_tiles import select_projection_tiles


def install():
    from track_1_short import ngram_table
    from track_1_short.ks3.geometry_tail import install as install_geometry_tail
    from track_1_short.ks3.moe_optimizer import install as install_optimizer
    from track_1_short.ks3.fp8 import attach
    ngram_table.NGRAM_LR_MUL = 70. * 1.15
    install_geometry_tail()
    install_optimizer()

    def mm(a, w, tiles, dtype=torch.float32):
        m, k = a.shape
        _, _, n = w.shape
        bm, bn, bk, warps, stages, _ = select_projection_tiles(m, n, k, w.stride(2) == 1)
        out = torch.empty((m, n), device=a.device, dtype=dtype)
        ks._mm[(tr.cdiv(m, bm), tr.cdiv(n, bn))](a, w, out, tiles, m, n, k, *w.stride(),
            BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
        return out

    def dw(a, b, out, offs):
        e, h, d = out.shape
        ks._dw[(tr.cdiv(h, 128), tr.cdiv(d, 128), e)](a, b, out, offs, h, d, False,
            BM=128, BN=128, BK=64, num_warps=8, num_stages=1)

    gemm.run, gemm.dw = mm, dw
    base = ks.install()

    class H100KS3(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_buffer('ks3_packed_bank', torch.empty_like(
                self.moe_bank, dtype=torch.bfloat16, device=self.device), persistent=False)

    return attach(H100KS3)
