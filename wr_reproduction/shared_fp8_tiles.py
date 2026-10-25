"""Opt-in MI300 tiles derived from the registered fifteen-case FP8 sweep.

This is hardware dispatch only. Small, grouped and non-row/column operands use
legacy dispatch. Original FP8 encodings, layouts, scales and epilogues remain.
"""


def select_tiles(m,n,k,a_stride,b_stride,grouped=False):
    if grouped or m<768 or min(n,k)<32 or a_stride[1]!=1 or b_stride[0]!=1:
        return None
    if n==768 and k>=8192 and m<=1536:
        return (64,64,128,4,1,32)
    if n==768 and k>=8192 and m<=4096:
        return (64,128,128,4,1,32)
    if n==768 and k<4096:
        return (128,128,128,8,1,32)
    return (128,256,128,8,1,32)


def select_tiles_v2(m,n,k,a_stride,b_stride,grouped=False):
    """Selective gains from ks3-shared-fp8-expanded-sweep-20261025T100101-631bcf.

    Producer epilogues retain v1: larger tiles did not improve their paired
    measurements. Shape classes interpolate the original logical batch sizes;
    grouped, small and unsupported-layout calls retain the original fallback.
    """
    baseline=select_tiles(m,n,k,a_stride,b_stride,grouped)
    if baseline is None:return None
    if n==2816 and k==768:return baseline
    if n==768 and k>=8192 and m<=1536:
        return (64,64,256,4,1,32)
    if n==768 and k>=8192 and m<=4096:
        return (64,128,256,4,1,32) if k>=49152 else baseline
    if (n==384 and k==768) or (m==10240 and n==768 and k==4096):
        return (128,256,128,4,1,32)
    return (256,256,128,8,1,32)


def select_packed_head_tiles(m,n,k,a_stride,b_stride,grouped=False):
    """Two-stage pipeline measured on eight original head logits/dX shapes."""
    if grouped or m<16384 or a_stride[1]!=1 or b_stride[0]!=1:return None
    if (k==768 and n>=10240) or (n==768 and k>=10240):
        return (256,256,128,8,2,32)
    return None


def select_packed_shared_tiles(m,n,k,a_stride,b_stride,grouped=False):
    """Measured packed-decoder pipelines for attention and dense MLP only."""
    if grouped or m<16384 or a_stride[1]!=1 or b_stride[0]!=1:return None
    if (n,k)==(2816,768):return (256,128,128,8,2,32)
    if (n,k) in ((2304,768),(768,1536),(768,2816)):
        return (256,256,128,8,2,32)
    return None
