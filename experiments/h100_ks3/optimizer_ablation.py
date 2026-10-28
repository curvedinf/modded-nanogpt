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
