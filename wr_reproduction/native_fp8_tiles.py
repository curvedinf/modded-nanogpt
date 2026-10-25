"""Measured choices for fully prepared KS3 native FP8 operands only."""


def projection_tiles(rows,n,k,prepared,policy='legacy'):
    if policy not in ('legacy','mi300-v2'):raise ValueError('Unknown native FP8 tile policy')
    original=(128,128,128,8,1,32) if n==768 else (64,128,128,4,1,16)
    if policy=='mi300-v2' and prepared and rows>=8192 and (k,n)==(1024,768):
        return (64,128,128,4,1,16)
    return original


def weight_gradient_tiles(rows,h,n,prepared,policy='legacy'):
    if policy not in ('legacy','mi300-v2'):raise ValueError('Unknown native FP8 tile policy')
    if policy=='mi300-v2' and prepared and rows>=8192 and (h,n)==(1024,768):
        return (128,256,128,8,1,32)
    return (64,128,64,4,1,16)
