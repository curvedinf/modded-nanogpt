"""Opt-in native ROCm varlen/local flash attention; canonical evaluation retained.

Uses the installed ATen interface because torch.nn.attention.varlen lacks the
explicit local-window and scale arguments needed by this model. Unsupported
providers raise; there is no silent fallback during a measured trial.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from ks3_research.attention import BufferedSelfAttention
from control393.eval_attention import bucketed_attention


def clean_boundaries(values,tokens):
    if not values or values[0]!=0 or values[-1]!=tokens or tokens<=0:
        raise ValueError('Cumulative lengths must span the complete token batch')
    if any(b<a for a,b in zip(values,values[1:])):
        raise ValueError('Cumulative lengths must be nondecreasing')
    unique=[values[0]]
    for value in values[1:]:
        if value!=unique[-1]:unique.append(value)
    return unique,max(b-a for a,b in zip(unique,unique[1:]))


class _NativeFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        if q.dtype not in (torch.bfloat16,torch.float16) or window<0:
            raise ValueError('Native flash requires BF16/FP16 and a nonnegative local window')
        boundaries,max_len=clean_boundaries(seqlens.detach().cpu().tolist(),q.shape[0])
        cu=torch.tensor(boundaries,device=q.device,dtype=torch.int32)
        ctx.qdim,ctx.vdim=q.shape[-1],v.shape[-1]
        width=max(ctx.qdim,ctx.vdim)
        qp,kp,vp=[F.pad(x,(0,width-x.shape[-1])).contiguous() for x in (q,k,v)]
        output,lse,seed,offset,_=torch.ops.aten._flash_attention_forward(
            qp,kp,vp,cu,cu,max_len,max_len,0.,True,False,
            scale=scale,window_size_left=window,window_size_right=0)
        ctx.save_for_backward(qp,kp,vp,cu,output,lse,seed,offset)
        ctx.max_len,ctx.window,ctx.scale=max_len,window,scale
        return output[...,:ctx.vdim]

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        q,k,v,cu,output,lse,seed,offset=ctx.saved_tensors
        gradient=F.pad(gradient,(0,v.shape[-1]-ctx.vdim)).contiguous()
        dq,dk,dv=torch.ops.aten._flash_attention_backward(
            gradient,q,k,v,output,lse,cu,cu,ctx.max_len,ctx.max_len,0.,True,seed,offset,
            scale=ctx.scale,window_size_left=ctx.window,window_size_right=0)
        return dq[...,:ctx.qdim],dk[...,:ctx.qdim],dv[...,:ctx.vdim],None,None,None


def native_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():return bucketed_attention(q,k,v,seqlens,window,scale)
    return _NativeFlash.apply(q,k,v,seqlens,window,scale)


class NativeFlashEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(native_or_bucketed_eval)


class _NativeEfficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        boundaries,max_len=clean_boundaries(seqlens.detach().cpu().tolist(),q.shape[0])
        cu=torch.tensor(boundaries,device=q.device,dtype=torch.int32)
        ctx.qdim,ctx.vdim=q.shape[-1],v.shape[-1]
        width=max(ctx.qdim,ctx.vdim)
        qp,kp,vp=[F.pad(x,(0,width-x.shape[-1])).contiguous()[None] for x in (q,k,v)]
        output,lse,seed,offset,_,_=torch.ops.aten._efficient_attention_forward(
            qp,kp,vp,None,cu,cu,max_len,max_len,0.,1,True,scale=scale,window_size=window+1)
        ctx.save_for_backward(qp,kp,vp,cu,output,lse,seed,offset)
        ctx.max_len,ctx.window,ctx.scale=max_len,window,scale
        return output[0,...,:ctx.vdim]

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        q,k,v,cu,output,lse,seed,offset=ctx.saved_tensors
        gradient=F.pad(gradient,(0,v.shape[-1]-ctx.vdim)).contiguous()[None]
        dq,dk,dv,_=torch.ops.aten._efficient_attention_backward(
            gradient,q,k,v,None,output,cu,cu,ctx.max_len,ctx.max_len,lse,0.,seed,offset,1,False,
            scale=ctx.scale,window_size=ctx.window+1)
        return dq[0,...,:ctx.qdim],dk[0,...,:ctx.qdim],dv[0,...,:ctx.vdim],None,None,None


def efficient_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():return bucketed_attention(q,k,v,seqlens,window,scale)
    return _NativeEfficient.apply(q,k,v,seqlens,window,scale)


class NativeEfficientEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(efficient_or_bucketed_eval)
