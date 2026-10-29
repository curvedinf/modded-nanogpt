"""Explicit KS3-only optimizer overrides; original shape/schedule logic remains."""
import math


def validate_geometry_betas(value):
    if (not isinstance(value,(tuple,list)) or len(value)!=2
            or any(isinstance(x,bool) or not isinstance(x,(int,float))
                   or not math.isfinite(x) or not 0<=x<1 for x in value)):
        raise ValueError('KS3 geometry Adam betas must be two finite numbers in [0,1)')
    return [float(x) for x in value]


def configure(table,geometry_lr,projection_lr_multiplier=1.,geometry_betas=(.9,.99)):
    betas=validate_geometry_betas(geometry_betas)
    if not math.isfinite(projection_lr_multiplier) or projection_lr_multiplier<=0:
        raise ValueError('Projection LR multiplier must be finite and positive')
    table['ks3_geometry']=dict(optim='adam',comms='replicated',adam_betas=betas,
                               lr_mul=geometry_lr,wd_mul=0.)
    if projection_lr_multiplier!=1.:
        original=table['moe_bank']
        table['moe_bank']=dict(original,lr_mul=original.get('lr_mul',1.)*projection_lr_multiplier)
