"""Unquantized CPU oracle for a future independent-rank-one KS3 hypothesis.

This module is not imported by the frozen probe training path. Keep the pinned
rank-one features and add gamma_i * tanh(x_i / 4) * tanh(x_j / 4), j = i xor 1.
Each gamma_i = bound * tanh(raw_pair_i) starts at zero. The added feature is
bounded in absolute value by ``bound``; no coordinates or spines are averaged.
GPU GS32/FP16 forward and STE backward still require separate implementation
and correctness gates before any convergence run.
"""
import math

import torch
from torch.nn import functional as F


def partner(x):
    """Swap adjacent coordinates without merging either independent pathway."""
    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("Pairwise features require a positive even width")
    return x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).reshape_as(x)


def _validate(coordinates, raw_pair, bound):
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("Interaction bound must be finite and positive")
    if coordinates.ndim < 1 or raw_pair.ndim < 1:
        raise ValueError("Coordinates and pair coefficients need a feature axis")
    if coordinates.shape[-1] != raw_pair.shape[-1]:
        raise ValueError("Pair coefficients must preserve the coordinate width")
    if coordinates.shape[-1] == 0 or coordinates.shape[-1] % 2:
        raise ValueError("Pairwise features require a positive even width")
    if torch.broadcast_shapes(coordinates.shape, raw_pair.shape) != coordinates.shape:
        raise ValueError("Pair coefficients may broadcast only over input rows")


class _PairwiseTerm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coordinates, raw_pair, bound):
        _validate(coordinates, raw_pair, bound)
        ctx.save_for_backward(coordinates, raw_pair)
        ctx.bound = bound
        b = torch.tanh(coordinates * .25)
        return bound * torch.tanh(raw_pair) * b * partner(b)

    @staticmethod
    def backward(ctx, grad):
        x, raw = ctx.saved_tensors
        # Recompute from saved inputs so the explicit VJP also has a valid
        # second derivative. Both members of every pair contribute to dx.
        b = torch.tanh(x * .25)
        a = torch.tanh(raw)
        weighted = grad * (ctx.bound * a)
        dx = .25 * (1 - b.square()) * (weighted * partner(b) + partner(weighted * b))
        da = (grad * ctx.bound * (1 - a.square()) * b * partner(b)).sum_to_size(raw.shape)
        return dx, da, None


def pairwise_term(coordinates, raw_pair, bound=1.):
    return _PairwiseTerm.apply(coordinates, raw_pair, bound)


def independent_rank_one(coordinates, geometry):
    """Pinned KS3 reference equations, with eight geometry values per chart."""
    if geometry.shape[-2:] != (coordinates.shape[-1], 8):
        raise ValueError("Geometry must contain six coefficients, scale and shift per coordinate")
    dtype = torch.float64 if coordinates.dtype == torch.float64 else torch.float32
    x = coordinates.to(dtype).clamp(-65504., 65504.)
    p0, p1, p2, p3, p4, p5, raw_scale, raw_shift = geometry.to(dtype).unbind(-1)
    t = 4. * torch.tanh(x / 4.)
    scale = F.softplus(raw_scale) + 1e-4
    shift = .5 * torch.tanh(raw_shift)
    z = torch.tanh(scale * (t + shift))
    cubic = 2. * p0 - 2. * p1 + p2 + p3
    quadratic = -3. * p0 + 3. * p1 - 2. * p2 - p3
    spine = ((cubic * t + quadratic) * t + p2) * t + p0
    warped = ((4. * p5 * z + 2. * p4) * z - 3. * p5) * z - p4
    return F.relu(x + spine + warped).square()


def coupled_rank_one(coordinates, geometry, raw_pair, bound=1.):
    base = independent_rank_one(coordinates, geometry)
    x = coordinates.to(base.dtype).clamp(-65504., 65504.)
    return base + pairwise_term(x, raw_pair.to(base.dtype), bound)


def block(inputs, up, down, geometry, raw_pair, bound=1.):
    coordinates = F.linear(inputs, up)
    features = coupled_rank_one(coordinates, geometry, raw_pair, bound)
    return F.linear(features.to(down.dtype), down)
