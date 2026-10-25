"""Optional FP8 products for dense KS3, reusing the verified routed kernels.

Dense and routed precision may differ. This module changes only KS3 dense
projections. E4M3/E5M2 originals and reusable native dual layouts share scales;
the frozen eight-rank activation-state implementation owns delayed statistics.
"""
import torch
import triton as tr

from wr_reproduction import ks3_moe_fp8 as fp
from wr_reproduction.ks3_moe import _pack


def configure():
    # The optional dense path currently requires BF16 routed KS3, so these
    # existing FP8 dispatch flags have no effect on the routed experts.
    fp.PREPARED_ACTIVATIONS = True
    fp.PREPARED_DUAL = True
    fp.PREPARED_GATHER = True


def init_buffers(model, features, *, training_gs32=True):
    if type(training_gs32) is not bool:
        raise ValueError('Dense GS32 selection must be boolean')
    model.ks3_dense_training_gs32 = training_gs32
    configure()
    for name, shape in [('up', (5, 1, features, 768)), ('up_t', (5, 1, 768, features)),
                        ('down', (5, 1, 768, features)), ('down_t', (5, 1, features, 768))]:
        for prefix, dtype in [('f8', torch.float8_e4m3fn), ('native', torch.float8_e4m3fnuz)]:
            model.register_buffer('ks3_dense_' + prefix + '_' + name,
                                  torch.empty(shape, device=model.device, dtype=dtype), persistent=False)
    for name in ('scales', 'amax'):
        model.register_buffer('ks3_dense_weight_' + name,
                              torch.ones((5, 2, 1), device=model.device), persistent=False)
    # Retained MLPs occupy the first entries; dense KS3 gets the remaining
    # entries. Frozen RankStates already clones and refreshes these four exact
    # buffer names for each original logical rank, including hybrid mixtures.
    from track_1_short.model.gpt import FP8_GRAD_SCALE
    for name, shape, initial in [('post_scales', (5,), 1.), ('post_amax', (5, 1), 0.),
                                 ('dpre_scales', (5,), FP8_GRAD_SCALE), ('dpre_amax', (5, 1), 0.)]:
        model.register_buffer('_mlp_' + name,
                              torch.full(shape, initial, device=model.device, dtype=torch.float32), persistent=False)


@torch.no_grad()
def refresh_weights(model, exact):
    for slot in range(5):
        if slot in model.retained_slots:
            continue
        for projection, suffix in ((0, 'up'), (1, 'down')):
            source = model.mlp_bank[slot, projection]
            if projection:
                source = source.T
            scale = model.ks3_dense_weight_scales[slot, projection]
            amax = model.ks3_dense_weight_amax[slot, projection]
            largest = source.float().abs().amax() if exact else amax
            scale.copy_(largest.clamp_min(1e-12) / 448.)
            amax.zero_()
            row = getattr(model, 'ks3_dense_f8_' + suffix)[slot, 0]
            col = getattr(model, 'ks3_dense_f8_' + suffix + '_t')[slot, 0]
            native = getattr(model, 'ks3_dense_native_' + suffix)[slot, 0]
            native_t = getattr(model, 'ks3_dense_native_' + suffix + '_t')[slot, 0]
            r, c = source.shape
            _pack[(tr.cdiv(r, 128), tr.cdiv(c, 32))](
                source, row, r, c, *source.stride(), SCALE=scale, OT=col, AMAX=amax,
                FP8=True, GS32=model.ks3_dense_training_gs32, NATIVE_OUT=native, NATIVE_T=native_t,
                ROWS_PER_TILE=128, num_warps=4)


@torch.library.custom_op('wr_ks3_dense_fp8::forward', mutates_args={'pa'})
def dense_forward(rows: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                  us: torch.Tensor, ws: torch.Tensor, xs: torch.Tensor, ps: torch.Tensor,
                  pa: torch.Tensor, geometry: torch.Tensor, slot: int, training_gs32: bool
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor, torch.Tensor]:
    from wr_reproduction.ks3_feedforward import COVERAGE
    COVERAGE['forward'].add(slot)
    m = tr.cdiv(rows.shape[0], 128) * 128
    src = torch.arange(m, device=rows.device, dtype=torch.int32)
    tiles = torch.zeros(m // 128, device=rows.device, dtype=torch.int32)
    offsets = torch.full((1,), m, device=rows.device, dtype=torch.int32)
    y, c, xt, ht = fp.forward(rows, src, tiles, up, down, us, ws, xs, ps, pa,
                               geometry.unsqueeze(0), training_gs32)
    return y, c, xt, ht, src, tiles, offsets


@dense_forward.register_fake
def _(rows, up, down, us, ws, xs, ps, pa, geometry, slot, training_gs32):
    m, d, n = tr.cdiv(rows.shape[0], 128) * 128, rows.shape[1], up.shape[1]
    return (rows.new_empty((m, d)), rows.new_empty((m, n), dtype=torch.float32),
            rows.new_empty((d, m), dtype=torch.float8_e4m3fnuz),
            rows.new_empty((n, m), dtype=torch.float8_e4m3fnuz),
            rows.new_empty((m,), dtype=torch.int32), rows.new_empty((m // 128,), dtype=torch.int32),
            rows.new_empty((1,), dtype=torch.int32))


@torch.library.custom_op('wr_ks3_dense_fp8::backward', mutates_args={'da'})
def dense_backward(grad: torch.Tensor, y: torch.Tensor, c: torch.Tensor, xt: torch.Tensor,
                   ht: torch.Tensor, src: torch.Tensor, tiles: torch.Tensor, offsets: torch.Tensor,
                   upt: torch.Tensor, downt: torch.Tensor, us: torch.Tensor, ws: torch.Tensor,
                   xs: torch.Tensor, gs: torch.Tensor, ps: torch.Tensor, ds: torch.Tensor,
                   da: torch.Tensor, geometry: torch.Tensor, slot: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from wr_reproduction.ks3_feedforward import COVERAGE
    COVERAGE['backward'].add(slot)
    bank = grad.new_empty((2, c.shape[1], grad.shape[1]))
    flag = torch.zeros((1,), device=grad.device, dtype=torch.int32)
    gate = torch.ones(grad.shape[0], device=grad.device, dtype=torch.float32)
    dx, _, dg = fp.backward(grad.contiguous(), y, c, xt, ht, gate, src, offsets, tiles,
                            upt, downt, us, ws, xs, gs, ps, ds, da, bank, flag,
                            geometry.unsqueeze(0), 0)
    return dx[:grad.shape[0]].contiguous(), bank, dg.squeeze(0)


@dense_backward.register_fake
def _(grad, y, c, xt, ht, src, tiles, offsets, upt, downt, us, ws, xs, gs, ps, ds, da, geometry, slot):
    return torch.empty_like(grad), grad.new_empty((2, c.shape[1], grad.shape[1])), torch.empty_like(geometry)


class DenseFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, w1, w2, geometry, caches, scales, slot, training_gs32):
        up, upt, down, downt = caches
        us, ws, xs, gs, ps, pa, ds, da = scales
        y, c, xt, ht, src, tiles, offsets = dense_forward(rows, up, down, us, ws, xs, ps, pa, geometry, slot, training_gs32)
        ctx.save_for_backward(y, c, xt, ht, src, tiles, offsets, upt, downt, geometry)
        # Delayed amax buffers mutate during every site's backward. Keep them
        # outside save_for_backward, as the existing routed FP8 path does, so
        # another slot sharing the same base buffer cannot invalidate autograd.
        ctx.scales = scales
        ctx.slot = slot
        return y[:rows.shape[0]]

    @staticmethod
    def backward(ctx, grad):
        y, c, xt, ht, src, tiles, offsets, upt, downt, geometry = ctx.saved_tensors
        us, ws, xs, gs, ps, pa, ds, da = ctx.scales
        dx, bank, dg = dense_backward(grad, y, c, xt, ht, src, tiles, offsets,
                                      upt, downt, us, ws, xs, gs, ps, ds, da, geometry, ctx.slot)
        return dx, bank[0], bank[1], dg, None, None, None, None


def apply(model, slot, scale_slot, rows, w1, w2):
    caches = tuple(getattr(model, 'ks3_dense_native_' + name)[slot]
                   for name in ('up', 'up_t', 'down', 'down_t'))
    scales = (model.ks3_dense_weight_scales[slot, 0], model.ks3_dense_weight_scales[slot, 1],
              model._mlp_x_scale, model._mlp_grad_scale,
              model._mlp_post_scales[scale_slot:scale_slot+1], model._mlp_post_amax[scale_slot],
              model._mlp_dpre_scales[scale_slot:scale_slot+1], model._mlp_dpre_amax[scale_slot])
    return DenseFP8.apply(rows, w1, w2, model.ks3_dense_geometry[slot], caches, scales, slot,
                          model.ks3_dense_training_gs32)
