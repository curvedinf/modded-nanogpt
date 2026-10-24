"""Existing ANVIL/Adam schedules, with FP32 KS3 masters and trainable geometry."""
import torch
from mi300.optim.anvil import (AnvilAndAdam, AnvilBank, anvil_bank_update,
                              anvil_cascade, SPLIT_BADDBMM_MIN_ROWS)
from mi300.perf.bank_scalars import BankScalarStaging
from mi300.training import TrainingManager
from mi300.training import value_embed_betas_and_wd_mul
from mi300.schedule import get_rail_beta
from mi300.optim.anvil import RAIL_ENGAGE_STEP, RAIL_FAST_BETA, RAIL_FAST_WEIGHT


class ContinuingManager(TrainingManager):
    def effective_lr_multiplier(self, step):
        scales = getattr(self, 'lr_stage_scales', (1.,)*len(self.schedule.boundaries))
        stage = next((i for i,(_,end) in enumerate(self.schedule.boundaries) if step<end),len(scales)-1)
        return self.schedule.get_lr(step)*scales[stage]

    def step_optimizers(self, step, row_prefetch, deferred_labels):
        # Preserve the original manager's ordering and GLOBAL update counters.
        # Only the rail schedule coordinate saturates: extrapolating its cooldown
        # beyond total_steps would eventually make momentum negative.
        step_lr = self.effective_lr_multiplier(step)
        rail_beta = get_rail_beta(min(step, self.schedule.total_steps), self.schedule.total_steps)
        if step >= RAIL_ENGAGE_STEP:
            self.optimizer.set_rails(fast_beta=RAIL_FAST_BETA, fast_weight=RAIL_FAST_WEIGHT)
        else:
            self.optimizer.set_rails(fast_beta=rail_beta, fast_weight=1.0)
        for param, cfg in self.optimizer.param_cfgs.items():
            cfg.lr = cfg.initial_lr * step_lr
            if cfg.optim == 'anvil':
                cfg.momentum = rail_beta
            elif cfg.label == 'value_embeds':
                cfg.adam_betas, cfg.wd_mul = value_embed_betas_and_wd_mul(step)
        event = row_prefetch.optimizer_event(step, lr=self.adam_defaults['lr'] * step_lr,
                                            weight_decay=self.adam_defaults['weight_decay'],
                                            eps=self.adam_defaults['eps'])
        deferred = self.optimizer.step(do_adam=self.is_adam_step(step), sparse_update=event,
                                       deferred_labels=deferred_labels)
        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()
        return deferred


class KS3Optimizer(AnvilAndAdam):
    def _init_banks(self):
        anvil = [(p, c) for p, c in self.param_cfgs.items() if c.optim == 'anvil']
        device = anvil[0][0].device
        self.bank_scalars = BankScalarStaging([c.chunk_size for _, c in anvil], device)
        self.banks = {}
        for i, (param, cfg) in enumerate(anvil):
            expected = torch.float32 if cfg.label == 'mlp_bank' else torch.bfloat16
            assert param.dtype == expected, (cfg.label, param.dtype)
            shape = (cfg.chunk_size, *cfg.reshape[1:])
            self.banks[cfg.label] = AnvilBank(
                label=cfg.label, grad=torch.empty(shape, dtype=param.dtype, device=device),
                **self.live_bank_state(cfg.label), **self.bank_scalars.scalars(i),
                eff_lr=self.bank_scalars.eff_lr(i), beta2=cfg.beta2,
                split_baddbmm=shape[-2] > SPLIT_BADDBMM_MIN_ROWS,
                red_dim=-1 if shape[-2] >= shape[-1] else -2)


@torch.no_grad()
def update_bank(bank):
    if bank.p_slice.dtype == torch.bfloat16:
        return anvil_bank_update(bank)
    # The original cascade mutates its FP32 argument; preserve the reduce-scatter buffer.
    update = anvil_cascade(bank.grad.clone(), bank.velocity, bank.momentum,
                          bank.split_baddbmm, bank.fast_beta, bank.fast_weight)
    update = AnvilAndAdam._rail_equalizer(update, bank.lane_energy, bank.beta2, bank.red_dim)
    p = bank.p_slice
    aligned = (bank.velocity[1] * p) >= 0
    p.copy_(p - p*aligned*bank.eff_wd*bank.eff_lr - update.float()*bank.eff_lr)


def make_manager(model, schedule, lr_scale=0.8, geometry_lr_mul=0.125, lr_stage_scales=None,
                 optimizer_backend='eager'):
    extra = {name: dict(optim='adam', comms='replicated', adam_betas=[.9,.99],
                        lr_mul=geometry_lr_mul, wd_mul=0.)
             for name in ('ks3_coefficients', 'ks3_scale', 'ks3_shift')}
    manager = ContinuingManager(model, schedule, update_bank, extra_param_table=extra,
                              optimizer_type=KS3Optimizer)
    manager.lr_stage_scales = tuple(lr_stage_scales or (1.,)*len(schedule.boundaries))
    # Include the row-sparse tables, whose base learning rate is read separately.
    manager.adam_defaults['lr'] *= lr_scale
    for cfg in manager.optimizer.param_cfgs.values():
        cfg.initial_lr *= lr_scale
        cfg.lr *= lr_scale
    if optimizer_backend=='bank_graphs':
        from ks3_research.optimizer_graphs import BankGraphs
        for bank in manager.optimizer.banks.values():
            bank.grad.zero_()
        manager.optimizer.stage_bank_scalars()
        graphs=BankGraphs(manager.optimizer,update_bank)
        manager.optimizer.bank_update=graphs.update
        manager.optimizer.bank_graphs=graphs
    elif optimizer_backend!='eager':
        raise ValueError('Unknown optimizer backend')
    return manager
