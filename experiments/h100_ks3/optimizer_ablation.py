"""Explicit learning-rate ablations for the PR393 CPLM pointer parameters."""
import math

COPY_LABELS = ('copy_qk_bank', 'copy_k_sink', 'copy_qk_gain')


def configure_copy(table, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Copy learning-rate multiplier must be positive and finite')
    for label in COPY_LABELS:
        entry = table[label]
        assert entry['optim'] == 'adam' and entry['comms'] == 'replicated'
        entry['lr_mul'] = entry.get('lr_mul', 1.) * multiplier


def install_copy(multiplier):
    from track_1_short import training
    original = training.AnvilAndAdam

    def optimizer(*args, **kwargs):
        configure_copy(kwargs['param_table'], multiplier)
        return original(*args, **kwargs)

    training.AnvilAndAdam = optimizer


def install_router(multiplier):
    """Change only the learning rate of the five top-one MoE routers."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Router learning-rate multiplier must be positive and finite')
    from track_1_short import training
    original = training.AnvilAndAdam

    def optimizer(*args, **kwargs):
        entry = kwargs['param_table']['moe_router']
        assert entry['optim'] == 'adam' and entry['comms'] == 'replicated'
        entry['lr_mul'] = entry.get('lr_mul', 1.) * multiplier
        return original(*args, **kwargs)

    training.AnvilAndAdam = optimizer


def install_schedule(multiplier):
    """Scale the common schedule, including its floor and sparse-table updates."""
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Schedule multiplier must be positive and finite')
    from track_1_short import schedule
    original = schedule.TrainingSchedule.get_lr

    def get_lr(self, step):
        return original(self, step) * multiplier

    schedule.TrainingSchedule.get_lr = get_lr


def install_floor(value):
    """Change the cooldown endpoint without scaling the early learning rate."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Learning-rate floor must be positive and finite')
    from track_1_short import schedule
    schedule.LR_FLOOR = value
