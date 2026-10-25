"""PR393 gradient gather with extra native layouts; original E5 outputs retained.

Kernel arithmetic/loop/order are copied from pinned426ab710 moe_mlp.py.
Only additional native stores follow each original pair of stores.
"""
import torch
import triton
import triton.language as tl
from wr_reproduction.fused_fp8 import _native

@triton.jit
def _grad_gather_native_kernel(g_ptr, y_ptr, src_ptr, s_ptr, f8_ptr, t_ptr, ds_ptr, scale_ptr, n, M,
                              D: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, GN, GTN):
    """For padded rows p of this program, t = src[p] (n: a zero row):
         f8[p] = e5m2(grad[t] * s[t] / scale) and its transpose   (the gradient of y, the ungated output)
         ds[p] = sum_d grad[t, d] * y_pad[p, d]                   (the gradient of the gate s[t])
    One program covers whole rows (the D loop), so ds needs no atomics and is deterministic."""
    pm = tl.program_id(0)
    rows = pm * BM + tl.arange(0, BM)
    s_row = tl.load(src_ptr + rows)
    valid = s_row < n
    gate = tl.load(s_ptr + s_row, mask=valid, other=0.0)
    inv = 1.0 / tl.load(scale_ptr)
    g_base = g_ptr + s_row.to(tl.int64)[:, None] * D
    y_base = y_ptr + rows.to(tl.int64)[:, None] * D
    ds = tl.zeros((BM,), dtype=tl.float32)
    for d0 in tl.static_range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        g = tl.load(g_base + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        y = tl.load(y_base + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        ds += tl.sum(g * y, axis=1)
        v = g * gate[:, None]
        q = tl.maximum(tl.minimum(v * inv, 57344.0), -57344.0).to(tl.float8e5)
        tl.store(f8_ptr + rows.to(tl.int64)[:, None] * D + cols[None, :], q)
        tl.store(t_ptr + cols.to(tl.int64)[:, None] * M + rows[None, :], tl.trans(q))
        native = _native(q.to(tl.uint8, bitcast=True), True, True)
        tl.store(GN + rows.to(tl.int64)[:, None] * D + cols[None, :], native)
        tl.store(GTN + cols.to(tl.int64)[:, None] * M + rows[None, :], tl.trans(native))
    tl.store(ds_ptr + rows, ds)


def gather(grad,y,src,gate,scale):
    n,d=grad.shape;m=src.numel()
    assert grad.dtype==torch.bfloat16 and grad.is_contiguous() and y.shape==(m,d) and y.is_contiguous()
    assert gate.dtype==torch.float32 and gate.shape==(n,) and src.dtype==torch.int32
    assert m%64==0 and d%128==0
    assert scale.dtype==torch.float32 and scale.numel()==1
    f8=torch.empty((m,d),device=grad.device,dtype=torch.float8_e5m2)
    ft=torch.empty((d,m),device=grad.device,dtype=f8.dtype)
    native=torch.empty_like(f8,dtype=torch.float8_e5m2fnuz)
    nt=torch.empty_like(ft,dtype=torch.float8_e5m2fnuz)
    ds=torch.empty(m,device=grad.device,dtype=torch.float32)
    _grad_gather_native_kernel[(m//64,)](grad,y,src,gate,f8,ft,ds,scale,n,m,
                                        D=d,BM=64,BD=128,GN=native,GTN=nt,num_warps=4)
    return f8,ft,ds,native,nt
