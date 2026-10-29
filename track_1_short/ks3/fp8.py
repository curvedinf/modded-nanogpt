"""H100 KS3 FP8 products with fused producers and reusable dual layouts.

Geometry remains the checked FP16-forward/FP32-STE formulation. Evaluation
uses the preserved BF16 path. Upstream delayed activation/gradient scaling
and owner-sharded weight refresh remain on the original training clock.
"""
import os
import torch, triton as tr
from track_1_short.ks3.fp8_kernels import publish, pack_input
from track_1_short.ks3.native_geometry import _geometry_prepared, _vjp_prepared
from track_1_short.ks3.prepared_geometry import _prepare
from track_1_short.ks3 import core
GS32 = True
DW_IMPL = 'tma'
VJP_ROWS = 128
VJP_WARPS = 4
VJP_COLS = 32
GATHER_IMPL = 'tiled'
TMA_STORES = True
V_DTYPE = torch.float16
C_DTYPE = torch.float16
ROUND_C = True
FUSE_GEOMETRY = False
MM_IMPL = 'tma'
PACKED_TRANSPORT = True
LAYOUT_TILE = 64
CUBIC_ONLY = True
TMA_OUTPUTS = True
FUSED_AMAX = True
COMPACT_PART = False
FUSE_CUBIC_VJP = False
DPRE_E4M3 = False
DPRE_DTYPE = torch.float8_e5m2
SKIP_MOE = frozenset()

def store_descriptors(row_major, column_major, rows, cols=32):
    from triton.tools.tensor_descriptor import TensorDescriptor
    assert row_major.shape[0] % rows == 0 and row_major.shape[1] % cols == 0
    return (TensorDescriptor.from_tensor(row_major, [rows, cols]), TensorDescriptor.from_tensor(column_major, [cols, rows]))

def update_amax(partial, out):
    from track_1_short.ks3.fp8_kernels import merge_amax
    merge_amax[tr.cdiv(partial.numel(), 1024),](partial, out, partial.numel(), num_warps=4)

def mm(a, w, table, sa, sw, dtype=torch.float32, mode=0, **kw):
    from track_1_short.ks3.tma_mm import mm as tma_mm
    return tma_mm(a, w, table, sa, sw, dtype, mode=mode, **kw)

def dw(a, b, out, offs, sa, sb, flag):
    from track_1_short.perf.kernels.moe_mlp import _grouped_weight_grad_impl
    e, h, d = out.shape
    _grouped_weight_grad_impl(a, b, offs, sa, sb, e, out.view(e * h, d), flag)

def gather(grad, y, src, scale, gg, gt, dgate, gs, rows, m):
    from track_1_short.ks3.fp8_kernels import gather_gate_tiled
    gather_gate_tiled[tr.cdiv(m, 64),](grad, y, src, scale, gg, gt, dgate, gs, rows, m, D=grad.shape[-1], num_warps=4, enable_fp_fusion=False)

def prepare_geometry(g, n):
    assert g.shape[-1] == 8
    prepared = torch.empty(8, n, 9, device=g.device)
    _prepare[tr.cdiv(8 * n, 256),](g, prepared, 8 * n)
    return prepared

@torch.library.custom_op('h100_ks3_fp8::forward', mutates_args={'pa'})
def forward(rows: torch.Tensor, src: torch.Tensor, table: torch.Tensor, up: torch.Tensor, down: torch.Tensor, us: torch.Tensor, ws: torch.Tensor, xs: torch.Tensor, ps: torch.Tensor, pa: torch.Tensor, g: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = src.numel()
    n = up.shape[1]
    d = rows.shape[-1]
    x = torch.empty(m, d, device=rows.device, dtype=torch.float8_e4m3fn)
    xt = torch.empty(d, m, device=rows.device, dtype=x.dtype)
    pack_input[tr.cdiv(m, 128), tr.cdiv(d, 32)](rows, x, xt, src, xs, m, rows.shape[0], D=d, GS32=GS32, num_warps=4)
    rank = (g.shape[-1] - 2) // 6
    prepared = prepare_geometry(g, n)
    if CUBIC_ONLY and (rank != 1 or (FUSE_GEOMETRY and MM_IMPL != 'tma')):
        raise ValueError('Cubic FP8 requires rank1 and separate geometry or TMA fusion')
    h = torch.empty(m, n, device=rows.device, dtype=x.dtype)
    ht = torch.empty(n, m, device=rows.device, dtype=x.dtype)
    c = mm(x, up.transpose(1, 2), table, xs, us, dtype=C_DTYPE)
    maxima = torch.empty(tr.cdiv(m, 64) * tr.cdiv(n, 32), device=rows.device)
    hd, htd = store_descriptors(h, ht, 64)
    _geometry_prepared[tr.cdiv(m, 64), tr.cdiv(n, 32)](c, g, h, table, m, n, HT=ht, SCALE=ps, AMAX=maxima, FP8=True, PARTIAL_AMAX=True, BM=64, RANK=rank, GS32=GS32, PREPARED=prepared, H_DESC=hd, HT_DESC=htd, CUBIC_ONLY=CUBIC_ONLY, num_warps=4)
    update_amax(maxima, pa)
    out = torch.empty_like(rows)
    y = mm(h, down.transpose(1, 2), table, ps, ws, dtype=rows.dtype, mode=2, SRC=src, GATE=scale, SCATTER=out, ROWS=rows.shape[0])
    return (out, y, c, xt, ht, prepared)

@forward.register_fake
def _(rows, src, table, up, down, us, ws, xs, ps, pa, g, scale):
    m = src.numel()
    n = up.shape[1]
    return (torch.empty_like(rows), rows.new_empty((m, rows.shape[-1])), rows.new_empty((m, n), dtype=C_DTYPE), rows.new_empty((rows.shape[-1], m), dtype=torch.float8_e4m3fn), rows.new_empty((n, m), dtype=torch.float8_e4m3fn), g.new_empty((8, n, 9)))

@torch.library.custom_op('h100_ks3_fp8::backward_core', mutates_args={'bank', 'da'})
def backward_core(grad: torch.Tensor, y: torch.Tensor, c: torch.Tensor, xt: torch.Tensor, ht: torch.Tensor, scale: torch.Tensor, src: torch.Tensor, offs: torch.Tensor, table: torch.Tensor, down: torch.Tensor, g: torch.Tensor, prepared: torch.Tensor, us: torch.Tensor, ws: torch.Tensor, xs: torch.Tensor, gs: torch.Tensor, ps: torch.Tensor, ds: torch.Tensor, da: torch.Tensor, bank: torch.Tensor, flag: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = c.shape
    rank = (g.shape[-1] - 2) // 6
    gg = torch.empty(m, grad.shape[-1], device=c.device, dtype=torch.float8_e5m2)
    gt = torch.empty(grad.shape[-1], m, device=c.device, dtype=gg.dtype)
    dgate = torch.empty(m, device=c.device)
    gather(grad, y, src, scale, gg, gt, dgate, gs, grad.shape[0], m)
    bm = VJP_ROWS
    v = mm(gg, down, table, gs, ws, dtype=V_DTYPE)
    dc = torch.empty(m, n, device=c.device, dtype=DPRE_DTYPE)
    dct = torch.empty(n, m, device=c.device, dtype=dc.dtype)
    part = torch.empty(tr.cdiv(m, bm), n, 8, device=c.device)
    maxima = torch.empty(tr.cdiv(m, bm) * tr.cdiv(n, VJP_COLS), device=c.device)
    dcd, dctd = store_descriptors(dc, dct, bm, VJP_COLS)
    _vjp_prepared[tr.cdiv(m, bm), tr.cdiv(n, VJP_COLS)](v, c, g, dc, part, table, src, m, n, grad.shape[0], DCT=dct, SCALE=ds, AMAX=maxima, FP8=True, PARTIAL_AMAX=True, BM=bm, RANK=rank, PREPARED=prepared, DC_DESC=dcd, DCT_DESC=dctd, CUBIC_ONLY=CUBIC_ONLY, COMPACT_PART=COMPACT_PART, DPRE_E4M3=DPRE_E4M3, BC=VJP_COLS, num_warps=VJP_WARPS)
    update_amax(maxima, da)
    dg = torch.empty_like(g)
    core._reduce[8, n](part, dg, offs, n, part.shape[0], tr.next_power_of_2(part.shape[0]), BM=bm, RANK=rank, num_warps=4)
    dw(ht, gt, bank[(2 * layer + 1) * 8:(2 * layer + 2) * 8], offs, ps, gs, flag)
    dw(dct, xt, bank[2 * layer * 8:(2 * layer + 1) * 8], offs, ds, xs, flag)
    return (dc, dgate, dg)

@backward_core.register_fake
def _(grad, y, c, xt, ht, scale, src, offs, table, down, g, prepared, us, ws, xs, gs, ps, ds, da, bank, flag, layer):
    return (c.new_empty(c.shape, dtype=DPRE_DTYPE), c.new_empty((c.shape[0],), dtype=torch.float32), torch.empty_like(g))

@torch.library.custom_op('h100_ks3_fp8::dx', mutates_args=())
def dx(dc: torch.Tensor, up: torch.Tensor, table: torch.Tensor, src: torch.Tensor, dl: torch.Tensor, router: torch.Tensor, ds: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
    from track_1_short.ks3.tma_mm import mm as tma_mm
    return tma_mm(dc, up, table, ds, us, torch.bfloat16, mode=4, SRC=src, DLOG=dl, ROUTER=router, ROWS=dl.shape[0])

@dx.register_fake
def _(dc, up, table, src, dl, router, ds, us):
    return dc.new_empty((dl.shape[0], router.shape[-1]), dtype=torch.bfloat16)

def routed_backward(ctx, grad, dprobs, dcounts):
    rows, router, post, probs, gate, expert, dest, src, offs, table, y, c, xt, ht, scale, g, prepared = ctx.saved_tensors
    up, upt, down, downt = ctx.caches
    us, ws, xs, gs, ps, pa, ds, da = ctx.scales
    dc, dsp, dg = backward_core(grad.contiguous(), y, c, xt, ht, scale, src, offs, table, downt.transpose(1, 2), g, prepared, us, ws, xs, gs, ps, ds, da, ctx.bank, ctx.flag, ctx.layer)
    ds_gate = dsp[dest.long()]
    dp = torch.zeros_like(probs) if dprobs is None else dprobs.float()
    dp = dp.scatter_add(1, expert[:, None], (ds_gate * post.float()).reshape(-1, 1))
    dl = (probs * (dp - (probs * dp).sum(-1, keepdim=True))).to(torch.bfloat16)
    result = dx(dc, upt.transpose(1, 2), table, src, dl, router, ds, us)
    return (result, dl.T @ rows, (ds_gate * gate).sum().to(post.dtype).reshape(post.shape), None, None, dg, None, None, None, None, None)

class Routed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rows, router, post, w1, w2, g, caches, scales, bank, flag, layer):
        from track_1_short.perf.kernels import moe_mlp as moe
        probs = torch.softmax(torch.nn.functional.linear(rows, router).float(), -1)
        gate, expert = probs.max(-1)
        dest, src, offs, table, counts = moe.moe_dispatch(expert, 8)
        scale = gate * post.float().reshape(())
        up, upt, down, downt = caches
        us, ws, xs, gs, ps, pa, ds, da = scales
        out, y, c, xt, ht, prepared = forward(rows, src, table, up, down, us, ws, xs, ps, pa, g, scale)
        ctx.save_for_backward(rows, router, post, probs, gate, expert, dest, src, offs, table, y, c, xt, ht, scale, g, prepared)
        ctx.caches = caches
        ctx.scales = scales
        ctx.bank = bank
        ctx.flag = flag
        ctx.layer = layer
        ctx.mark_non_differentiable(counts)
        return (out, probs, counts)
    backward = staticmethod(routed_backward)

def attach(base):
    import torch.distributed as dist
    from triton.backends.nvidia.compiler import get_ptxas
    if tuple(map(int, get_ptxas(90).version.split('.'))) < (13, 3):
        raise RuntimeError('H100 FP8 dual stores require ptxas >=13.3; use run.sh or set TRITON_PTXAS_PATH.')
    from track_1_short.model.moe import MOE_INDEX
    from track_1_short.perf.cuda_graphs import fp8_refresh_graphs as refresh

    class FP8GPT(base):

        def _refresh_delayed_scale(self, amax_buf, scale_buf, gain, floor):
            return base._refresh_delayed_scale(amax_buf, scale_buf, gain, floor)

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            n, d = self.ks3_up.shape[-2:]
            for name, shape in [('ks3_f8_up', (5, 8, n, d)), ('ks3_f8_up_t', (5, 8, d, n)), ('ks3_f8_down', (5, 8, d, n)), ('ks3_f8_down_t', (5, 8, n, d))]:
                self.register_buffer(name, torch.empty(shape, device=self.device, dtype=torch.float8_e4m3fn), persistent=False)
            self.register_buffer('ks3_weight_scales', torch.ones((5, 2, 8), device=self.device), persistent=False)
            self.register_buffer('ks3_weight_amax', torch.ones((5, 2, 8), device=self.device), persistent=False)
            self.register_buffer('ks3_fp8_transport', torch.empty((80, n * d + 64), device=self.device, dtype=torch.float8_e4m3fn), persistent=False)
            self.register_buffer('ks3_fp8_partial', torch.ones((80, n * d // 4096), device=self.device), persistent=False)

        @torch.no_grad()
        def _quantize_moe(self, lo, hi, exact):
            n, d = self.ks3_up.shape[-2:]
            scales = self.ks3_weight_scales.view(-1)
            if exact:
                flat = self.moe_bank[lo:hi].reshape(hi - lo, -1)
                maximum = torch.maximum(flat.amax(1), -flat.amin(1)).float()
            else:
                maximum = self.ks3_fp8_partial[lo:hi].amax(1)
            scales[lo:hi].copy_(maximum.clamp_min(1e-12) / 448.0)
            publish[n * d // 4096, hi - lo](self.moe_bank, self.ks3_fp8_transport, scales, self.ks3_fp8_partial, lo, n, D=d, PACK_SCALE=PACKED_TRANSPORT, GS32=GS32, num_warps=4)
            if lo == 0 and hi == 80:
                self.build_moe_cols()

        def gather_moe_f8(self):
            lo, hi = (self._moe_lo, self._moe_hi)
            q = self.ks3_fp8_transport.view(torch.uint8)
            s = self.ks3_weight_scales.view(-1)
            return [('moe_bank', dist.all_gather_into_tensor(q, q[lo:hi], async_op=True).get_future())]

        @torch.no_grad()
        def build_moe_cols(self):
            n, d = self.ks3_up.shape[-2:]
            from track_1_short.ks3.weight_layouts import build
            build(self.ks3_fp8_transport, self.ks3_f8_up, self.ks3_f8_up_t, self.ks3_f8_down, self.ks3_f8_down_t, n, self.ks3_weight_scales, packed=PACKED_TRANSPORT, tile=LAYOUT_TILE)

        @torch.no_grad()
        def prepare_ks3_eval(self):
            return super().prepare_ks3_eval()

        def _moe(self, layer, normed, weights, post):
            if layer in SKIP_MOE:
                return (torch.zeros_like(normed), normed.new_zeros((), dtype=torch.float32))
            if not self.training:
                return super()._moe(layer, normed, weights, post)
            k = MOE_INDEX[layer]
            w1, w2, router = weights[k]
            rows = normed.reshape(-1, normed.shape[-1])
            if post is None:
                post = rows.new_ones(())
            caches = (self.ks3_f8_up[k], self.ks3_f8_up_t[k], self.ks3_f8_down[k], self.ks3_f8_down_t[k])
            scales = (self.ks3_weight_scales[k, 0], self.ks3_weight_scales[k, 1], self._mlp_x_scale, self._mlp_grad_scale, self._moe_post_scales[k:k + 1], self._moe_post_amax[k], self._moe_dpre_scales[k:k + 1], self._moe_dpre_amax[k])
            out, probs, counts = Routed.apply(rows, router, post, w1, w2, self.ks3_geometry[k], caches, scales, self.moe_bank_grad, self.moe_grad_accumulate, k)
            return (out.reshape(normed.shape), 8 * (counts.float() / rows.shape[0] * probs.mean(0)).sum())
    refresh.MOE_LOCAL_OUTPUTS = ('ks3_fp8_transport', 'ks3_weight_scales', 'ks3_fp8_partial')
    refresh.MOE_COLS_OUTPUTS = ('ks3_f8_up', 'ks3_f8_up_t', 'ks3_f8_down', 'ks3_f8_down_t')
    refresh.MOE_COLS_OUTPUTS += ('ks3_weight_scales',)
    return FP8GPT
