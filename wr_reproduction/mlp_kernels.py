"""MI300 replacements for the record's Hopper MLP kernel entry points only.

The original autograd class, caches, delayed scales, post-lambda gradients and
model code remain untouched. These kernels preserve the source epilogues:
BF16 pre rounding, BF16 relu-square rounding, E4M3 post quantization, and BF16
dpre rounding before E5M2 gradient quantization. All training GEMMs are FP8.
"""
import torch
from wr_reproduction.fp8 import mm

def linear_relu_square(a,b,aux=None,a_f8=None,b_f8=None,dequant_scale_ptr=None,
                       emit_f8=False,post_scale=None,post_amax=None,
                       emit_t=False,store_pre=True,store_post_bf=True,
                       dpre_scale=None,dpre_amax=None):
    if b_f8 is None:
        if aux is not None:raise ValueError('Record backward is FP8-only')
        pre=(a@b.T).to(torch.bfloat16)
        post=pre.clamp_min(0).square()
        return pre if store_pre else None,post if store_post_bf else None,None,None
    from wr_reproduction.fused_fp8 import lrs_op
    if a_f8 is None or dequant_scale_ptr is None:raise ValueError('Missing original FP8 operands/scales')
    if aux is None:
        if not emit_f8:raise ValueError('Record FP8 training must produce cached FP8 intermediates')
        q,qt,pre,post=lrs_op(a_f8,b_f8.T,dequant_scale_ptr,post_scale,post_scale,
            post_amax,post_scale,True,emit_t,store_pre,store_post_bf)
        return pre if store_pre else None,post if store_post_bf else None,q,qt if emit_t else None
    q,qt,_,_=lrs_op(a_f8,b_f8.T,dequant_scale_ptr,post_scale,dpre_scale,
        dpre_amax,aux,False,True,False,False)
    return q,qt

def install(upstream_mlp):
    """Patch kernel dispatch in memory; never rewrite the pinned source."""
    upstream_mlp.linear_relu_square=linear_relu_square
    for name,transpose_b,fast in [('dp_f8',False,True),('wg2_f8',True,False),
                                  ('wg1_f8',True,False),('dx_f8',False,False)]:
        def kernel(a,b,a_scale,b_scale,_transpose=transpose_b,_fast=fast):
            return mm(a,b.T if _transpose else b,a_scale,b_scale,
                      out_dtype=torch.bfloat16,use_fast_accum=_fast)
        getattr(upstream_mlp,name+'_op').register_kernel('cuda',kernel)
