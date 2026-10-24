"""PR393/PR379 copy-sink pointer, same Triton math; namespaced for the MI300 port. See provenance.json."""
import torch
import triton
import triton.language as tl
from torch import Tensor

COPY_BLOCK = 2048      # training copy band: one to two blocks (queries of block b see key blocks b-1 and b)


# ---------------------------------------------------------------------------------------------
# Fused (flash-style) copy-sink pointer: scores are computed tile by tile on-chip and tiles outside the
# query's own document are skipped entirely. Needs the copy dim D in {64, 128, 256}.

@triton.jit
def _copy_fwd_kernel(Q, K, S0, DOC, TOK, TGT, DSTART, P_OUT, A_OUT, LSE_OUT, N, L, LB, scale,
                     D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    # NK shifts share one attention: shift kk copies the token kk after the source (TOK row kk = input[j + kk]) for
    # target row kk (= the target kk steps ahead), counting only sources with j + kk <= query (no future tokens).
    m0 = tl.program_id(0) * BM
    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
    s0 = tl.load(S0 + offs_m)
    doc_q = tl.load(DOC + offs_m)
    band_lo = (m0 // L) * L - L  # all queries of the tile share an L-block (BM divides L)
    lo = tl.maximum(tl.maximum(tl.load(DSTART + m0), band_lo), m0 - LB)
    lo = (tl.maximum(lo, 0) // BN) * BN
    m_i = s0  # the sink is the running max's starting point, so rows without sources are well defined
    l_i = tl.full([BM], 1.0, tl.float32)
    acc = tl.zeros([BM, 4], tl.float32)
    col = tl.arange(0, 4)
    for n0 in range(lo, m0 + BM, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        doc_k = tl.load(DOC + offs_n)
        valid = (offs_n[None, :] < offs_m[:, None]) & (doc_k[None, :] == doc_q[:, None]) & (offs_n[None, :] >= band_lo) \
            & (offs_m[:, None] - offs_n[None, :] <= LB)
        s = tl.where(valid, s, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        for kk in tl.static_range(NK):
            tok = tl.load(TOK + kk * N + offs_n)
            tgt = tl.load(TGT + kk * N + offs_m)
            hit = (tok[None, :] == tgt[:, None]) & (offs_n[None, :] + kk <= offs_m[:, None])
            acc += tl.where(col[None, :] == kk, tl.sum(tl.where(hit, p, 0.0), axis=1)[:, None], 0.0)
        m_i = m_new
    for kk in tl.static_range(NK):
        tl.store(P_OUT + kk * N + offs_m, tl.sum(tl.where(col[None, :] == kk, acc, 0.0), axis=1) / l_i)
    tl.store(A_OUT + offs_m, tl.exp(s0 - m_i) / l_i)
    tl.store(LSE_OUT + offs_m, m_i + tl.log(l_i))


@triton.jit
def _copy_bwd_dq_kernel(Q, K, DOC, TOK, TGT, DSTART, LSE, GP, DROW, DQ, N, L, LB, scale,
                        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    m0 = tl.program_id(0) * BM
    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
    doc_q = tl.load(DOC + offs_m)
    lse = tl.load(LSE + offs_m)
    drow = tl.load(DROW + offs_m)
    band_lo = (m0 // L) * L - L
    lo = tl.maximum(tl.maximum(tl.load(DSTART + m0), band_lo), m0 - LB)
    lo = (tl.maximum(lo, 0) // BN) * BN
    dq = tl.zeros([BM, D], tl.float32)
    for n0 in range(lo, m0 + BM, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        doc_k = tl.load(DOC + offs_n)
        valid = (offs_n[None, :] < offs_m[:, None]) & (doc_k[None, :] == doc_q[:, None]) & (offs_n[None, :] >= band_lo) \
            & (offs_m[:, None] - offs_n[None, :] <= LB)
        a = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        gm = tl.zeros([BM, BN], tl.float32)
        for kk in tl.static_range(NK):
            tok = tl.load(TOK + kk * N + offs_n)
            tgt = tl.load(TGT + kk * N + offs_m)
            gp = tl.load(GP + kk * N + offs_m)
            hit = (tok[None, :] == tgt[:, None]) & (offs_n[None, :] + kk <= offs_m[:, None])
            gm += tl.where(hit, gp[:, None], 0.0)
        ds = a * (gm - drow[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + offs_m[:, None] * D + offs_d[None, :], dq * scale)


@triton.jit
def _copy_bwd_dk_kernel(Q, K, DOC, TOK, TGT, DEND, LSE, GP, DROW, DK, N, L, LB, scale,
                        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NK: tl.constexpr):
    n0 = tl.program_id(0) * BN
    offs_n = n0 + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])
    doc_k = tl.load(DOC + offs_n)
    # queries that can see these keys: later positions of the same document, inside the band
    # (query block qb sees keys from block qb-1 onwards, so keys of block kb are seen up to block kb+1)
    hi = tl.minimum(tl.load(DEND + n0 + BN - 1), (n0 // L + 2) * L)
    hi = tl.minimum(tl.minimum(hi, N), n0 + BN + LB)
    lo = (n0 // BM) * BM
    dk = tl.zeros([BN, D], tl.float32)
    for m0 in range(lo, hi, BM):
        offs_m = m0 + tl.arange(0, BM)
        q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])
        doc_q = tl.load(DOC + offs_m)
        lse = tl.load(LSE + offs_m)
        drow = tl.load(DROW + offs_m)
        band_lo = (m0 // L) * L - L
        s = tl.dot(q, tl.trans(k)) * scale
        valid = (offs_n[None, :] < offs_m[:, None]) & (doc_k[None, :] == doc_q[:, None]) & (offs_n[None, :] >= band_lo) \
            & (offs_m[:, None] - offs_n[None, :] <= LB)
        a = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        gm = tl.zeros([BM, BN], tl.float32)
        for kk in tl.static_range(NK):
            tok = tl.load(TOK + kk * N + offs_n)
            tgt = tl.load(TGT + kk * N + offs_m)
            gp = tl.load(GP + kk * N + offs_m)
            hit = (tok[None, :] == tgt[:, None]) & (offs_n[None, :] + kk <= offs_m[:, None])
            gm += tl.where(hit, gp[:, None], 0.0)
        ds = a * (gm - drow[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + offs_n[:, None] * D + offs_d[None, :], dk * scale)


def _copy_tiles(D):
    """BM, BN, num_warps, num_stages of the copy kernels."""
    return (64, 64, 4, 3) if D <= 128 else (64, 32, 8, 2)


@torch.library.custom_op("control393::cplm_copy_fwd", mutates_args=())
def cplm_copy_fwd(q: Tensor, k: Tensor, s0: Tensor, doc: Tensor, tok: Tensor, tgt: Tensor, dstart: Tensor,
                  dend: Tensor, L: int, LB: int) -> tuple[Tensor, Tensor, Tensor]:
    """dend is read by the backward only; it rides through the op so setup_context can save it."""
    N, D = q.shape
    NK = tok.shape[0]  # tok, tgt: [NK, N] (see _copy_fwd_kernel)
    BM, BN, nw, ns = _copy_tiles(D)
    assert N % L == 0 and L % BM == 0 and L % BN == 0 and D in (64, 128, 256) and 1 <= NK <= 4
    q, k, s0 = q.contiguous(), k.contiguous(), s0.contiguous()
    P = torch.empty((NK, N), dtype=torch.float32, device=q.device)
    A = torch.empty(N, dtype=torch.float32, device=q.device)
    LSE = torch.empty_like(A)
    _copy_fwd_kernel[(N // BM,)](q, k, s0, doc.contiguous(), tok.contiguous(), tgt.contiguous(), dstart.contiguous(),
                                 P, A, LSE, N, L, LB, D ** -0.5, D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
    return P, A, LSE


@cplm_copy_fwd.register_fake
def _(q, k, s0, doc, tok, tgt, dstart, dend, L, LB):
    N, NK = q.shape[0], tok.shape[0]
    return (q.new_empty((NK, N), dtype=torch.float32), q.new_empty(N, dtype=torch.float32),
            q.new_empty(N, dtype=torch.float32))


@torch.library.custom_op("control393::cplm_copy_bwd", mutates_args=())
def cplm_copy_bwd(q: Tensor, k: Tensor, doc: Tensor, tok: Tensor, tgt: Tensor, dstart: Tensor, dend: Tensor,
                  LSE: Tensor, gP: Tensor, drow: Tensor, L: int, LB: int) -> tuple[Tensor, Tensor]:
    N, D = q.shape
    NK = tok.shape[0]
    BM, BN, nw, ns = _copy_tiles(D)
    q, k, gP, drow = q.contiguous(), k.contiguous(), gP.contiguous(), drow.contiguous()
    doc, tok, tgt, dstart, dend = (t.contiguous() for t in (doc, tok, tgt, dstart, dend))
    dq = torch.empty((N, D), dtype=torch.float32, device=q.device)
    dk = torch.empty_like(dq)
    _copy_bwd_dq_kernel[(N // BM,)](q, k, doc, tok, tgt, dstart, LSE, gP, drow, dq, N, L, LB, D ** -0.5,
                                    D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
    _copy_bwd_dk_kernel[(N // BN,)](q, k, doc, tok, tgt, dend, LSE, gP, drow, dk, N, L, LB, D ** -0.5,
                                    D=D, BM=BM, BN=BN, NK=NK, num_warps=nw, num_stages=ns)
    return dq, dk


@cplm_copy_bwd.register_fake
def _(q, k, doc, tok, tgt, dstart, dend, LSE, gP, drow, L, LB):
    return q.new_empty(q.shape, dtype=torch.float32), q.new_empty(q.shape, dtype=torch.float32)


def _copy_setup_context(ctx, inputs, output):
    q, k, s0, doc, tok, tgt, dstart, dend, L, LB = inputs
    P, A, LSE = output
    ctx.save_for_backward(q, k, doc, tok, tgt, dstart, dend, LSE, P, A)
    ctx.L, ctx.LB = L, LB
    ctx.set_materialize_grads(False)


def _copy_backward(ctx, gP, gA, gLSE):
    q, k, doc, tok, tgt, dstart, dend, LSE, P, A = ctx.saved_tensors
    gP = torch.zeros_like(P) if gP is None else gP.float()
    gA = torch.zeros_like(A) if gA is None else gA.float()
    drow = (gP * P).sum(0) + gA * A  # d loss / d (log-normalizer), per row
    ds0 = A * (gA - drow)            # sink score gradient
    dq, dk = torch.ops.control393.cplm_copy_bwd(q, k, doc, tok, tgt, dstart, dend, LSE, gP, drow, ctx.L, ctx.LB)
    return dq.to(q.dtype), dk.to(k.dtype), ds0, None, None, None, None, None, None, None


cplm_copy_fwd.register_autograd(_copy_backward, setup_context=_copy_setup_context)


def copy_sink_probs(q, k, k_sink, input_seq, target_seq, seqlens, L):
    """Copy-sink pointer on a packed 1-D stream (PR #379's fused path: no shifts, no lookback cap).

    q, k [N, d] copy queries/keys (RMS-normed); k_sink [d]; input_seq/target_seq [N]; seqlens = cu_seqlens
    ([0, doc ends..., padded with N]). Query i attends to the strictly-past positions j of its own document inside the
    band (key blocks b-1 and b of L tokens for a query in block b) plus the learned sink. Returns
    p_copy [N] = sum_j attn[i, j] * [input_seq[j] == target_seq[i]] and a_sink [N]."""
    N, d = q.shape
    dev = input_seq.device
    cu = seqlens.long().clamp(max=N)
    j = torch.searchsorted(cu, torch.arange(N, device=dev), right=True)  # boundaries <= each position
    cu_ext = torch.cat([cu.new_zeros(1), cu, cu.new_full((1,), N)])
    doc, dstart, dend = j.to(torch.int32), cu_ext[j].to(torch.int32), cu_ext[j + 1].to(torch.int32)
    s0 = (q.float() @ k_sink.float()) * d ** -0.5
    tok = input_seq.to(torch.int32).view(1, N).contiguous()
    tgt = target_seq.to(torch.int32).view(1, N).contiguous()
    P, A, _ = torch.ops.control393.cplm_copy_fwd(q, k, s0, doc, tok, tgt, dstart, dend, L, N)
    return P[0], A


