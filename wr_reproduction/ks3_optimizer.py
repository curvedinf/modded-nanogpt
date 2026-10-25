"""Explicit KS3-only optimizer overrides; original shape/schedule logic remains."""
import math


def configure(table,geometry_lr,projection_lr_multiplier=1.):
    if not math.isfinite(projection_lr_multiplier) or projection_lr_multiplier<=0:
        raise ValueError('Projection LR multiplier must be finite and positive')
    table['ks3_geometry']=dict(optim='adam',comms='replicated',adam_betas=[.9,.99],
                               lr_mul=geometry_lr,wd_mul=0.)
    if projection_lr_multiplier!=1.:
        original=table['moe_bank']
        table['moe_bank']=dict(original,lr_mul=original.get('lr_mul',1.)*projection_lr_multiplier)
