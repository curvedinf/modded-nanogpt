"""Larger execution groups for unchanged document/window attention."""
import torch
from ks3_research.attention import _FusedAccumulationSDPA, BufferedSelfAttention
from mi300.portable import varlen_attention


class _Fused128SDPA(_FusedAccumulationSDPA):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.group_size=128
        return _FusedAccumulationSDPA.forward(ctx,q,k,v,seqlens,window,scale)


def fused128_attention(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    return _Fused128SDPA.apply(q,k,v,seqlens,window,scale)


class Fused128SelfAttention(BufferedSelfAttention):
    attention=staticmethod(fused128_attention)
