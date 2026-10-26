"""Combine exact BF16 gradient gather with the validated32-row geometry VJP.

Original forward/products/packing, Torch gate sum and BF16 coordinate-gradient
storage remain unchanged. Delegate untuned backward to the original kernel.
"""
import torch
import triton as tr
from wr_reproduction import ks3_moe as core

ORIGINAL_BACKWARD = core.backward
EXECUTED_MODES = set()


@torch.library.custom_op('wr_ks3_bf16_gather_vjp32::backward', mutates_args={'bank'})
def backward(grad: torch.Tensor, y: torch.Tensor, x: torch.Tensor, c: torch.Tensor,
             h: torch.Tensor, gate: torch.Tensor, src: torch.Tensor, offs: torch.Tensor,
             tiles: torch.Tensor, up: torch.Tensor, down: torch.Tensor, g: torch.Tensor,
             bank: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if core.KS3_RANK != 1:
        raise ValueError('BF16 row32 gate covers independent rank1 only')
    if not core.BF16_TUNED_KERNELS:
        EXECUTED_MODES.add('original_untuned_backward')
        return ORIGINAL_BACKWARD(grad, y, x, c, h, gate, src, offs, tiles, up, down, g, bank, layer)
    EXECUTED_MODES.add('training_bf16_exact_gather_vjp32')
    if up.shape[0] == 8:
        core.EXECUTED_MOE_BACKWARDS.add(layer)
    from wr_reproduction.ks3_bf16_gather import gather
    gg, ds = gather(grad, y, src, gate)
    v = core.mm(gg, down, tiles); m, n = c.shape; e = up.shape[0]
    dc = torch.empty_like(c, dtype=torch.bfloat16)
    part = torch.empty((tr.cdiv(m, 32), n, 8), device=c.device, dtype=torch.float32)
    core._vjp[(tr.cdiv(m, 32), tr.cdiv(n, 32))](
        v, c, g, dc, part, tiles, src, m, n, grad.shape[0], BM=32, RANK=1, num_warps=4)
    dg = torch.empty_like(g)
    core._reduce[(e, n)](part, dg, offs, n, part.shape[0], tr.next_power_of_2(part.shape[0]),
                        BM=32, RANK=1, num_warps=4)
    from wr_reproduction.ks3_bf16_gemm import dw
    dw(h, gg, bank[(2*layer+1)*e:(2*layer+2)*e], offs)
    dw(dc, x, bank[2*layer*e:(2*layer+1)*e], offs)
    dx = core.mm(dc, up, tiles, dtype=grad.dtype)
    return dx, ds, dg


@backward.register_fake
def _(grad, y, x, c, h, gate, src, offs, tiles, up, down, g, bank, layer):
    return x.new_empty(x.shape), x.new_empty((x.shape[0],), dtype=torch.float32), torch.empty_like(g)


def install():
    if core.backward not in (ORIGINAL_BACKWARD, backward):
        raise RuntimeError('A different KS3 backward overlay is already installed')
    core.backward = backward
