"""Differentiable BF16 replacements for the record's Hopper-only kernels.

Query chunks bound attention storage without truncating context. Checkpointing bounds
MLP and softmax activation memory; recomputation changes throughput, not the objective.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def relu_squared_mlp(x, up, down):
    def compute(x, up, down):
        return F.relu(F.linear(x, up)).square() @ down
    if torch.is_grad_enabled():
        return checkpoint(compute, x, up, down, use_reentrant=False)
    return compute(x, up, down)


def qk_norm_rope(qk, f1, f2, heads, rotary_dim, paired, key_offset):
    t, _, d = qk.shape
    # The source kernel normalizes in fp32 with fp32 epsilon and rounds only after RoPE.
    z = qk.float()
    z = z * (z.square().mean(-1, keepdim=True) + 1.1920928955078125e-7).rsqrt()
    q0, k0 = z.split(heads, dim=1)
    if paired:
        offsets = torch.arange(heads, device=qk.device) % 2
        f1 = f1.view(t, 2, d)[:, offsets]
        f2 = f2.view(t, 2, d)[:, offsets]
    else:
        f1, f2 = f1[:, None], f2[:, None]
    rotate = lambda a: (f1.float() * a + f2.float() * a.unflatten(-1, (-1, 2)).flip(-1).flatten(-2))
    q, k = rotate(q0), rotate(k0)
    if key_offset:
        k = torch.cat((k[..., :rotary_dim], torch.cat((k[:1, :, rotary_dim:], k0[:-1, :, rotary_dim:]))), -1)
    if paired:
        q, k = q.reshape(2*t, heads//2, d), k.reshape(2*t, heads//2, d)
    return q.to(qk.dtype), k.to(qk.dtype)


def _attend(q, k, v, q_start, k_start, window, scale):
    qi = torch.arange(q.shape[0], device=q.device) + q_start
    ki = torch.arange(k.shape[0], device=q.device) + k_start
    allowed = (ki[None] <= qi[:, None]) & (ki[None] >= qi[:, None] - window)
    # SDPA backends want matching Q/K/V widths. Zero padding is exact with explicit scale.
    width = max(q.shape[-1], v.shape[-1])
    qp, kp, vp = [F.pad(a, (0, width-a.shape[-1])).transpose(0, 1)[None] for a in (q,k,v)]
    out = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=allowed, dropout_p=0.0, scale=scale)
    return out[0].transpose(0, 1)[..., :v.shape[-1]]


def varlen_attention(q, k, v, seqlens, window, scale):
    boundaries = seqlens.detach().cpu().tolist()
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        for lo in range(start, end, 256):
            hi = min(lo+256, end)
            kl = max(start, lo-window)
            values = (q[lo:hi], k[kl:hi], v[kl:hi], lo, kl, window, scale)
            if torch.is_grad_enabled():
                outputs.append(checkpoint(_attend, *values, use_reentrant=False))
            else:
                outputs.append(_attend(*values))
    return torch.cat(outputs)


def training_loss(x, weight, targets, prefix_table, mtp_weights, prefix_weight, sampled):
    n = x.shape[0]
    if sampled is None:
        candidate_ids = torch.arange(weight.shape[1], device=x.device)
        mapped, prefixes = targets, prefix_table[targets]
    else:
        candidate_ids = (sampled.vocab_pos >= 0).nonzero().flatten()
        mapped, prefixes = sampled.target_pos[:n], sampled.prefix_pos[:n]
    # Match MTP shifts across the packed token stream, including at document boundaries.
    shifted = []
    for offset in range(mtp_weights.numel()):
        shifted.append(F.pad(mapped[offset:], (0, offset), value=-1))
    labels = torch.stack(shifted, 1)
    def slab(h, w, ids, ys, ps, mw, pw):
        logits = h @ w.index_select(1, ids)
        capped = 23 * torch.sigmoid((logits.float() + 5) / 7.5)
        logp = F.log_softmax(capped, dim=-1)
        loss = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
        for i in range(ys.shape[1]):
            y = ys[:, i]
            loss = loss - mw[i] * logp.gather(1, y.clamp_min(0)[:, None]).squeeze(1) * (y >= 0)
        loss = loss - pw[0] * logp.gather(1, ps.clamp_min(0)[:, None]).squeeze(1) * (ps >= 0)
        return loss
    chunks = []
    for lo in range(0, n, 1024):
        hi = min(lo+1024, n)
        values = (x[lo:hi], weight, candidate_ids, labels[lo:hi], prefixes[lo:hi], mtp_weights, prefix_weight)
        chunks.append(checkpoint(slab, *values, use_reentrant=False))
    return torch.cat(chunks)
