"""Checkpointed dense ReLU-square MLP with original BF16 rounding and GEMMs.

Only fresh, private linear/gradient intermediates are overwritten. The saved
input and weight views remain intact. Canonical evaluation uses the original
implementation; these kernels serve training and its explicit parity probes.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from mi300.portable import relu_squared_mlp


@triton.jit(do_not_specialize=['N'])
def _relu_square(PRE,N,B:tl.constexpr=1024):
    rows=tl.program_id(0)*B+tl.arange(0,B)
    z=tl.load(PRE+rows,rows<N,other=0.).to(tl.float32)
    positive=tl.where(z<=0.,0.,z)  # Preserve NaN propagation.
    tl.store(PRE+rows,positive*positive,rows<N)


@triton.jit(do_not_specialize=['N'])
def _relu_square_and_gradient(PRE,GRAD,N,B:tl.constexpr=1024):
    rows=tl.program_id(0)*B+tl.arange(0,B)
    z=tl.load(PRE+rows,rows<N,other=0.).to(tl.float32)
    g=tl.load(GRAD+rows,rows<N,other=0.).to(tl.float32)
    positive=tl.where(z<=0.,0.,z)
    # Eager PowBackward multiplies by the BF16-rounded 2*relu(z), then
    # ReluBackward zeros nonpositive rows. Keep that intermediate rounding.
    twice=(2.*positive).to(PRE.dtype.element_ty).to(tl.float32)
    derivative=tl.where(z<=0.,0.,g*twice)
    tl.store(PRE+rows,positive*positive,rows<N)
    tl.store(GRAD+rows,derivative,rows<N)


def activation_(pre):
    assert pre.is_contiguous() and pre.dtype==torch.bfloat16
    _relu_square[(triton.cdiv(pre.numel(),1024),)](pre,pre.numel(),num_warps=4)
    return pre


def activation_and_gradient_(pre,gradient):
    assert pre.is_contiguous() and gradient.is_contiguous()
    assert pre.shape==gradient.shape and pre.dtype==gradient.dtype==torch.bfloat16
    _relu_square_and_gradient[(triton.cdiv(pre.numel(),1024),)](
        pre,gradient,pre.numel(),num_warps=4)
    return pre,gradient


class _DensePointwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,up,down):
        ctx.input_shape=x.shape
        flat=x.reshape(-1,x.shape[-1])
        ctx.save_for_backward(flat,up,down)
        activated=activation_(F.linear(flat,up))
        return (activated@down).reshape(*x.shape[:-1],down.shape[-1])

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        x,up,down=ctx.saved_tensors
        gradient=gradient.reshape(-1,down.shape[-1]).contiguous()
        pre=F.linear(x,up)
        grad_pre=gradient@down.T
        activated,grad_pre=activation_and_gradient_(pre,grad_pre)
        grad_down=activated.T@gradient
        del activated,pre
        grad_x=grad_pre@up
        grad_up=grad_pre.T@x
        return grad_x.reshape(ctx.input_shape),grad_up,grad_down


def dense_pointwise_mlp(x,up,down):
    if not torch.is_grad_enabled() or x.device.type!='cuda' or x.dtype!=torch.bfloat16:
        return relu_squared_mlp(x,up,down)
    return _DensePointwise.apply(x,up,down)
