"""Training-only BF16 hidden features; retain the original KS3 evaluation path.

Only the groupwise INT8 rounding after the FP16 nonlinearity is bypassed.
Input, weight and coefficient packing, every product, and the smooth STE stay
unchanged. No extra parameters or random draws are introduced.
"""
import torch
import triton as tr
from wr_reproduction import ks3_moe as core

ORIGINAL_FORWARD = core.forward
EXECUTED_MODES = set()


@torch.library.custom_op('wr_ks3_hidden_bf16::forward', mutates_args=())
def forward(rows: torch.Tensor, src: torch.Tensor, tiles: torch.Tensor,
            up: torch.Tensor, down: torch.Tensor, g: torch.Tensor,
            gs32: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not gs32 or core.KS3_RANK != 1:
        raise ValueError('This overlay requires original GS32 inputs and independent rank1 experts')
    if not core.BF16_TUNED_KERNELS:
        EXECUTED_MODES.add('original_evaluation')
        return ORIGINAL_FORWARD(rows, src, tiles, up, down, g, gs32)
    EXECUTED_MODES.add('training_hidden_bf16')
    m, d = src.numel(), rows.shape[1]
    x = torch.empty((m, d), device=rows.device, dtype=torch.bfloat16)
    core._pack[(tr.cdiv(m, 128), tr.cdiv(d, 32))](
        rows, x, m, d, *rows.stride(), SRC=src, INPUT_ROWS=rows.shape[0],
        ROWS_PER_TILE=128, GS32=True, num_warps=4)
    c = core.mm(x, up.transpose(1, 2), tiles)
    h = torch.empty_like(c, dtype=torch.bfloat16)
    core._geometry[(tr.cdiv(m, 64), tr.cdiv(c.shape[1], 32))](
        c, g, h, tiles, *c.shape, BM=64, RANK=1, GS32=False, num_warps=4)
    y = core.mm(h, down.transpose(1, 2), tiles, dtype=rows.dtype)
    return y, x, c, h


@forward.register_fake
def _(rows, src, tiles, up, down, g, gs32=True):
    m, d, n = src.numel(), rows.shape[1], up.shape[1]
    return rows.new_empty((m, d)), rows.new_empty((m, d)), \
        rows.new_empty((m, n), dtype=torch.float32), rows.new_empty((m, n))


def install():
    if core.forward not in (ORIGINAL_FORWARD, forward):
        raise RuntimeError('A different KS3 forward overlay is already installed')
    core.forward = forward
