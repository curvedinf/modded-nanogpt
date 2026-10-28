"""Reuse PR393's fused ANVIL machinery at the KS3 bank's original margin.

The original expert path bypasses this machinery because its normalization
margin is 1.1, while the dense fused helper hard-codes 1.05. Keep the expert
margin, maps, rails and update rule; change only their implementation.
"""
import torch
from track_1_short.optim import anvil
from track_1_short.perf.kernels import anvil_opt as fused

ORIGINAL = anvil.anvil_cascade


@torch.compile(dynamic=False, fullgraph=True)
def normalize(x, gram, margin: float):
    trace = gram.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
    divisor = trace.sqrt() * margin + 1e-6
    return (x.float() / divisor).bfloat16(), (gram.float() / divisor.square()).bfloat16()


def cascade(grad_chunk, velocity, momentum_t, split_baddbmm,
            fast_beta_t, fast_weight_t, margin=1.05):
    if grad_chunk.size(-2) <= grad_chunk.size(-1):
        return ORIGINAL(grad_chunk, velocity, momentum_t, split_baddbmm,
                        fast_beta_t, fast_weight_t, margin)
    x = torch.empty_like(grad_chunk)
    fused.pre_fused(grad_chunk, velocity, x, momentum_t, fast_beta_t, fast_weight_t)
    n = x.size(-1)
    gram = torch.empty((*x.shape[:-2], n, n), device=x.device, dtype=x.dtype)
    fused.gram_tall(x, gram)
    x, gram = normalize(x, gram, margin)
    poly = torch.empty_like(gram)
    out = torch.empty_like(x)
    for k, (a, b, c) in enumerate(anvil.ANVIL_MAPS):
        if k:
            fused.gram_tall(x, gram)
        fused.poly_gram(gram, c, b, poly)
        if split_baddbmm:
            fused.gemm_xb_add(x, poly, out, a)
        else:
            torch.baddbmm(x, x, poly, beta=a, out=out)
        x, out = out, x
    return x


def install_dimension_tiles():
    # PR393's original table covers only its 768-wide residual stream. These
    # tiles also divide 512/640; keep every existing entry and arithmetic rule.
    for width in (512,640):
        for family in ('XTX','XXT','baAA'):
            fused._GRAM_CFG.setdefault((family,width),(64,128,128,4,3))


def install():
    install_dimension_tiles()
    anvil.anvil_cascade = cascade
