"""256-chunk execution groups for the unchanged attention operation."""
import torch
from ks3_research.attention import _FusedAccumulationSDPA,BufferedSelfAttention
from mi300.portable import varlen_attention


class _Fused256SDPA(_FusedAccumulationSDPA):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.group_size=256
        return _FusedAccumulationSDPA.forward(ctx,q,k,v,seqlens,window,scale)


def fused256_attention(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():
        return varlen_attention(q,k,v,seqlens,window,scale)
    return _Fused256SDPA.apply(q,k,v,seqlens,window,scale)


class Fused256SelfAttention(BufferedSelfAttention):
    attention=staticmethod(fused256_attention)
