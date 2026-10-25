"""Measured BF16 projection choices; original expert boundaries stay128."""


def select_projection_tiles(m,n,k,row_major_weight,policy='mi300-v1'):
    if m<=32768:
        baseline=(64,128,128,4,1,16) if n==768 else (128,128,64,8,1,32)
    elif n==768:
        baseline=(128,128,64,8,1,32) if row_major_weight else (64,128,64,4,1,16)
    else:
        baseline=(64,128,64,4,1,16) if row_major_weight else (64,128,128,4,1,16)
    if policy=='mi300-v1':return baseline
    if policy!='mi300-v2':raise ValueError('Unknown BF16 projection tile policy')
    # Only the1024-feature shape was swept. Tiny diagnostics retain v1 too.
    if m<8192 or (n,k) not in ((1024,768),(768,1024)):return baseline
    if m<=32768:
        if n==1024 and row_major_weight:return (128,256,64,8,1,32)
    else:
        if n==1024 and not row_major_weight:return (64,256,64,8,1,32)
        if n==768:return (128,128,64,8,2,32)
    return baseline
