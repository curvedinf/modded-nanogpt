"""Opt-in geometry averaging on the same ticks as PR393's MoE projections.

Geometry is replicated, so its averages remain replicated and need no extra
collectives. Unlike projection matrices, nonlinear geometry parameters are
not Frobenius-rescaled after averaging.
"""
import torch


class GeometryTail:
    def __init__(self, parameter, total_steps, *, avg_window=250, avg_period=4,
                 avg_rate=4.0 / 53.0, blend_window=298, blend_period=4,
                 blend_rate=2.0 / (298 // 4 + 1), blend=0.55 - 0.2007):
        self.parameter = parameter
        self.total_steps = total_steps
        self.avg_window, self.avg_period, self.avg_rate = avg_window, avg_period, avg_rate
        self.blend_window, self.blend_period = blend_window, blend_period
        self.blend_rate, self.blend = blend_rate, blend
        self.avg = torch.empty_like(parameter, dtype=torch.float32)
        self.bank = torch.empty_like(parameter, dtype=torch.float32)
        self.avg_seeded = self.bank_seeded = False

    @torch.no_grad()
    def tick(self, step):
        n = self.total_steps
        if step >= n - self.avg_window and (n - 1 - step) % self.avg_period == 0:
            if self.avg_seeded:
                self.avg.lerp_(self.parameter, self.avg_rate)
            else:
                self.avg.copy_(self.parameter)
                self.avg_seeded = True
        if step >= n - self.blend_window and (step % self.blend_period == 1 or step >= n - 2):
            if self.bank_seeded:
                self.bank.lerp_(self.parameter, self.blend_rate)
            else:
                self.bank.copy_(self.parameter)
                self.bank_seeded = True

    @torch.no_grad()
    def ship(self):
        assert self.avg_seeded and self.bank_seeded, 'Geometry tail did not receive both windows'
        self.parameter.copy_(torch.lerp(self.avg, self.bank, self.blend))


def install():
    from track_1_short import tail_average as original
    base = original.TailAverages
    if getattr(base, '_ks3_geometry_tail_installed', False):
        return
    old_init, old_tick, old_ship = base.__init__, base.tick, base.ship

    def init(self, optimizer, rank, value_embed_updates, total_steps):
        old_init(self, optimizer, rank, value_embed_updates, total_steps)
        parameter = optimizer._param_by_label['ks3_geometry']
        assert optimizer.param_cfgs[parameter].comms == 'replicated'
        self.ks3_geometry_tail = GeometryTail(
            parameter, total_steps,
            avg_window=original.TAIL_AVG_WINDOW, avg_period=original.TAIL_AVG_PERIOD,
            avg_rate=original.TAIL_AVG_RATE, blend_window=original.BANK_BLEND_WINDOW,
            blend_period=original.BANK_BLEND_PERIOD, blend_rate=original.BANK_BLEND_RATE,
            blend=original.BANK_BLEND - original.RICHARDSON_W)

    def tick(self, step):
        old_tick(self, step)
        self.ks3_geometry_tail.tick(step)

    def ship(self):
        old_ship(self)
        self.ks3_geometry_tail.ship()

    base.__init__, base.tick, base.ship = init, tick, ship
    base._ks3_geometry_tail_installed = True
