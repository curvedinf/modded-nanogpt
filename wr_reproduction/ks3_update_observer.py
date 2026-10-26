"""Sparse, read-only KS3 gradients at the actual optimizer entry boundary.

This optional extension does not edit the frozen probe/evaluation implementation.
Two original updates after a probe cover both Adam cadences. Samples exclude
inactive dense bank slots and separate coefficients, scale, shift, up and down.
"""
import math
import time
import torch


def evenly(count, device, cap=4096):
    n = min(count, cap)
    return torch.arange(n, device=device, dtype=torch.int64) * (count - 1) // max(n - 1, 1)


def specifications(model):
    """Return bounded indices in the original tensors, without copying a bank."""
    result = []
    dense_slots = sorted(set(range(5)) - set(getattr(model, 'retained_slots', range(5))))
    for name, branch in (('ks3_geometry', 'routed'), ('ks3_dense_geometry', 'dense')):
        if not hasattr(model, name):
            continue
        p = getattr(model, name)
        if p.shape[-1] != 8:
            raise ValueError('This diagnostic is explicitly rank one')
        if branch == 'dense':
            per_slot = p.shape[1]
            count = len(dense_slots) * per_slot
            selected = evenly(count, p.device, 512)
            slots = torch.tensor(dense_slots, device=p.device)
            groups = slots[selected // per_slot] * per_slot + selected % per_slot
        else:
            count = p.numel() // 8
            groups = evenly(count, p.device, 512)
        for component, columns in (('coefficients', range(6)), ('raw_scale', [6]), ('raw_shift', [7])):
            columns = list(columns)
            indices = (groups[:, None] * 8 + torch.tensor(columns, device=p.device)).reshape(-1)
            result.append(dict(key=branch + '.' + component, parameter=name, tensor=p,
                               indices=indices, total_elements=count * len(columns),
                               branch=branch, component=component, active_slots=dense_slots if branch == 'dense' else None))
    for name, branch in (('moe_bank', 'routed'), ('mlp_bank', 'dense')):
        if not hasattr(model, name) or (branch == 'dense' and not dense_slots):
            continue
        p = getattr(model, name)
        if branch == 'routed':
            # Original layout [5 layers, 2 directions, 8 experts, features, 768].
            per_side = p.numel() // 10
            slots = torch.arange(5, device=p.device)
        else:
            per_side = p.shape[-2] * p.shape[-1]
            slots = torch.tensor(dense_slots, device=p.device)
        total = len(slots) * per_side
        selected = evenly(total, p.device)
        for direction, component in enumerate(('up_projection', 'down_projection')):
            indices = (slots[selected // per_side] * 2 + direction) * per_side + selected % per_side
            result.append(dict(key=branch + '.' + component, parameter=name, tensor=p,
                               indices=indices, total_elements=total, branch=branch,
                               component=component, active_slots=dense_slots if branch == 'dense' else None))
    return result


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class UpdateObserver:
    def __init__(self, model, manager, log):
        self.model, self.manager, self.log = model, manager, log
        self.remaining = 0
        self.total_seconds = 0.
        self.reports = []
        self.specs = None
        self.original = manager.step_optimizers
        manager.step_optimizers = self.step

    def arm(self, updates=2):
        if self.remaining:
            raise RuntimeError('Overlapping sparse diagnostic windows')
        self.remaining = updates

    @torch.no_grad()
    def capture(self, do_adam, allow_missing=False):
        if self.specs is None:
            self.specs = specifications(self.model)
        values = []
        for spec in self.specs:
            p, index = spec['tensor'], spec['indices']
            cfg = self.manager.optimizer.param_cfgs[p]
            applied = cfg.optim != 'adam' or do_adam
            if applied and p.grad is None and not allow_missing:
                raise RuntimeError('Missing reduced gradient at optimizer entry: ' + spec['key'])
            version, pointer = p._version, p.data_ptr()
            grad = p.grad
            grad_identity = None if grad is None else (grad.data_ptr(), grad._version)
            previous = p.detach().reshape(-1)[index].float().clone()
            gradient = None if grad is None else grad.detach().reshape(-1)[index].float().clone()
            assert (p._version, p.data_ptr()) == (version, pointer)
            assert grad is p.grad
            assert grad is None or (grad.data_ptr(), grad._version) == grad_identity
            values.append((spec, previous, gradient, applied))
        return values

    @torch.no_grad()
    def finish(self, captured):
        tensors, rows = [], []
        for spec, previous, gradient, applied in captured:
            p = spec['tensor']
            delta = p.detach().reshape(-1)[spec['indices']].float() - previous
            grad_rms = previous.new_tensor(float('nan')) if gradient is None else gradient.square().mean().sqrt()
            tensors.append(torch.stack((previous.square().mean().sqrt(), delta.square().mean().sqrt(), grad_rms)))
            rows.append(dict(key=spec['key'], parameter=spec['parameter'], branch=spec['branch'],
                             component=spec['component'], active_slots=spec['active_slots'],
                             sample_elements=previous.numel(), total_elements=spec['total_elements'],
                             optimizer_applied=applied, gradient_available=gradient is not None))
        numbers = torch.stack(tensors).cpu().tolist()
        for row, (parameter, update, gradient) in zip(rows, numbers):
            if not math.isfinite(parameter) or not math.isfinite(update) or (row['gradient_available'] and not math.isfinite(gradient)):
                raise RuntimeError('Nonfinite KS3 diagnostic: ' + row['key'])
            row.update(parameter_rms=parameter, update_rms=update,
                       relative_update_rms=update / max(parameter, 1e-12),
                       gradient_sample_rms=gradient if row['gradient_available'] else None,
                       gradient_l2_estimate=gradient * row['total_elements'] ** .5 if row['gradient_available'] else None,
                       gradient_missing_reason=None if row['gradient_available'] else 'Adam gradient remains in original rank-local accumulation on this even update')
        return {row.pop('key'): row for row in rows}

    def step(self, step, *args, **kwargs):
        if not self.remaining:
            return self.original(step, *args, **kwargs)
        device = next(self.model.parameters()).device
        # Draining already queued training is reported separately; it remains on
        # the continuous run clock. No pause or duration is subtracted from it.
        t = time.perf_counter(); synchronize(device)
        training_drain = time.perf_counter() - t
        t = time.perf_counter()
        do_adam = self.manager.is_adam_step(step)
        captured = self.capture(do_adam)
        synchronize(device)
        before_seconds = time.perf_counter() - t
        result = self.original(step, *args, **kwargs)
        t = time.perf_counter(); synchronize(device)
        optimizer_drain = time.perf_counter() - t
        t = time.perf_counter()
        measurements = self.finish(captured)
        self.remaining -= 1
        row = dict(event='ks3_gradient_update', step=step + 1, original_step=step,
                   do_adam=do_adam, measurements=measurements,
                   gradient_boundary='after all-eight-rank reduction; before original step_optimizers',
                   sampling='deterministic bounded active-slot samples; geometry components separated',
                   approximate_norms=True, single_original_update=True,
                   pre_sample_training_drain_seconds=training_drain,
                   post_optimizer_drain_seconds=optimizer_drain,
                   extra_seconds_before_optimizer=before_seconds,
                   timing='All sampling, drains and logging remain on continuous training clock; drains wait for original queued training work')
        if hasattr(self.model, 'post_lambdas'):
            row['feedforward_post_multipliers_bf16'] = self.model.post_lambdas[:, 1].detach().bfloat16().float().cpu().tolist()
            row['activation_caveat'] = 'Core projected-output moments precede these post multipliers and routed gate weights.'
        self.reports.append(row)
        self.total_seconds += before_seconds
        self.log(**row)
        self.total_seconds += time.perf_counter() - t
        return result
