"""Verified SDPA forward with tiled local backward and exclusive gradient writes.

Original MI300 implementation of logsumexp and Q/K/V derivatives. Rejected
standalone-forward prototypes are preserved in run144-146 source archives.
Algorithmic reference: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
No native provider window argument is trusted. Every tile masks causal position,
left window, document identity, and the final partial token block explicitly.
Canonical evaluation retains the already verified equal-shape SDPA path.
"""
import torch
from itertools import accumulate
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from ks3_research.attention import BufferedSelfAttention
from control393.eval_attention import bucketed_attention
from control393.bucket_training import shape_groups,group_metadata,pack_group,flatten_selected,grouped_sdpa
from ks3_research.attention import chunks_for
from control393.attention_metadata import host_boundaries,paired_boundaries,batched_group_metadata


def grouped_training_forward(q,k,v,seqlens,window,scale,compact=False,async_metadata=False):
    """Use verified training groups so irregular document tails share launches."""
    boundaries=host_boundaries(seqlens) if async_metadata else seqlens.detach().cpu().tolist()
    chunks=chunks_for(boundaries,window)
    offsets=[0,*accumulate(hi-kl for _,hi,kl in chunks)]
    result=torch.empty_like(v)
    layouts=(batched_group_metadata(q.device,chunks,offsets,window,q.shape[1],compact)
             if async_metadata else
             (group_metadata(q.device,chunks,ids,offsets,window,q.shape[1],compact)
              for ids in shape_groups(chunks)))
    for metadata in layouts:
        qp,kp,vp=pack_group(q,k,v,metadata,compact)
        mask,qd,qt=metadata[5],metadata[7],metadata[8]
        output=grouped_sdpa(qp,kp,vp,mask,scale)
        result.index_copy_(0,qd,flatten_selected(output,qt,v.shape[-1]))
    return result


@triton.jit(do_not_specialize=['N'])
def _row_delta(OUT,GRAD,DELTA,N,H:tl.constexpr,DV:tl.constexpr,B:tl.constexpr=64):
    rows=tl.program_id(0)*B+tl.arange(0,B);head=tl.program_id(1);dims=tl.arange(0,DV)
    offsets=(rows[:,None]*H+head)*DV+dims[None,:]
    out=tl.load(OUT+offsets,rows[:,None]<N,other=0.).to(tl.float32)
    grad=tl.load(GRAD+offsets,rows[:,None]<N,other=0.).to(tl.float32)
    tl.store(DELTA+rows*H+head,tl.sum(out*grad,1),rows<N)


@triton.jit(do_not_specialize=['N'])
def _forward_stats(Q,K,DOC,LSE,N,H:tl.constexpr,DQ:tl.constexpr,
                   WINDOW:tl.constexpr,SCALE:tl.constexpr,B:tl.constexpr=64):
    block=tl.program_id(0);head=tl.program_id(1)
    rows=block*B+tl.arange(0,B);dims=tl.arange(0,DQ)
    query=tl.load(Q+(rows[:,None]*H+head)*DQ+dims[None,:],rows[:,None]<N,other=0.)
    docs=tl.load(DOC+rows,rows<N,other=-1)
    maximum=tl.full((B,),-1.e6,tl.float32);denominator=tl.zeros((B,),tl.float32)
    first=tl.maximum(0,block*B-WINDOW)//B*B
    for start in range(first,(block+1)*B,B):
        cols=start+tl.arange(0,B)
        key=tl.load(K+(cols[None,:]*H+head)*DQ+dims[:,None],cols[None,:]<N,other=0.)
        key_docs=tl.load(DOC+cols,cols<N,other=-2)
        allowed=(rows[:,None]<N)&(cols[None,:]<N)&(cols[None,:]<=rows[:,None])
        allowed=allowed&(cols[None,:]>=rows[:,None]-WINDOW)&(docs[:,None]==key_docs[None,:])
        score=tl.dot(query,key)*(SCALE*1.4426950408889634)
        score=tl.where(allowed,score,-1.e6)
        next_max=tl.maximum(maximum,tl.max(score,1))
        probability=tl.where(allowed,tl.exp2(score-next_max[:,None]),0.)
        denominator=denominator*tl.exp2(maximum-next_max)+tl.sum(probability,1)
        maximum=next_max
    tl.store(LSE+rows*H+head,maximum+tl.log2(denominator),rows<N)


@triton.jit(do_not_specialize=['N'])
def _query_gradient(Q,K,V,DOC,LSE,GRAD,DELTA,GQ,N,H:tl.constexpr,DQ:tl.constexpr,
                    DV:tl.constexpr,WINDOW:tl.constexpr,SCALE:tl.constexpr,B:tl.constexpr=64):
    block=tl.program_id(0);head=tl.program_id(1)
    rows=block*B+tl.arange(0,B);dq=tl.arange(0,DQ);dv=tl.arange(0,DV)
    query=tl.load(Q+(rows[:,None]*H+head)*DQ+dq[None,:],rows[:,None]<N,other=0.)
    grad=tl.load(GRAD+(rows[:,None]*H+head)*DV+dv[None,:],rows[:,None]<N,other=0.)
    docs=tl.load(DOC+rows,rows<N,other=-1)
    lse=tl.load(LSE+rows*H+head,rows<N,other=0.)
    delta=tl.load(DELTA+rows*H+head,rows<N,other=0.)
    accumulated=tl.zeros((B,DQ),tl.float32)
    first=tl.maximum(0,block*B-WINDOW)//B*B
    for start in range(first,(block+1)*B,B):
        cols=start+tl.arange(0,B)
        key=tl.load(K+(cols[:,None]*H+head)*DQ+dq[None,:],cols[:,None]<N,other=0.)
        value=tl.load(V+(cols[None,:]*H+head)*DV+dv[:,None],cols[None,:]<N,other=0.)
        key_docs=tl.load(DOC+cols,cols<N,other=-2)
        allowed=(rows[:,None]<N)&(cols[None,:]<N)&(cols[None,:]<=rows[:,None])
        allowed=allowed&(cols[None,:]>=rows[:,None]-WINDOW)&(docs[:,None]==key_docs[None,:])
        score=tl.dot(query,tl.trans(key))*(SCALE*1.4426950408889634)
        probability=tl.where(allowed,tl.exp2(score-lse[:,None]),0.)
        dp=tl.dot(grad,value)
        ds=(probability*(dp-delta[:,None])).to(query.dtype)
        accumulated=tl.dot(ds,key,accumulated)
    tl.store(GQ+(rows[:,None]*H+head)*DQ+dq[None,:],accumulated*SCALE,rows[:,None]<N)


@triton.jit(do_not_specialize=['N'])
def _key_value_gradient(Q,K,V,DOC,LSE,GRAD,DELTA,GK,GV,N,H:tl.constexpr,DQ:tl.constexpr,
                        DV:tl.constexpr,WINDOW:tl.constexpr,SCALE:tl.constexpr,B:tl.constexpr=64):
    block=tl.program_id(0);head=tl.program_id(1)
    cols=block*B+tl.arange(0,B);dq=tl.arange(0,DQ);dv=tl.arange(0,DV)
    key=tl.load(K+(cols[:,None]*H+head)*DQ+dq[None,:],cols[:,None]<N,other=0.)
    value=tl.load(V+(cols[:,None]*H+head)*DV+dv[None,:],cols[:,None]<N,other=0.)
    key_docs=tl.load(DOC+cols,cols<N,other=-2)
    gk=tl.zeros((B,DQ),tl.float32);gv=tl.zeros((B,DV),tl.float32)
    last=tl.minimum(N,(block+1)*B+WINDOW)
    for start in range(block*B,last,B):
        rows=start+tl.arange(0,B)
        query=tl.load(Q+(rows[:,None]*H+head)*DQ+dq[None,:],rows[:,None]<N,other=0.)
        grad=tl.load(GRAD+(rows[:,None]*H+head)*DV+dv[None,:],rows[:,None]<N,other=0.)
        docs=tl.load(DOC+rows,rows<N,other=-1)
        lse=tl.load(LSE+rows*H+head,rows<N,other=0.)
        delta=tl.load(DELTA+rows*H+head,rows<N,other=0.)
        allowed=(rows[:,None]<N)&(cols[None,:]<N)&(cols[None,:]<=rows[:,None])
        allowed=allowed&(cols[None,:]>=rows[:,None]-WINDOW)&(docs[:,None]==key_docs[None,:])
        score=tl.dot(query,tl.trans(key))*(SCALE*1.4426950408889634)
        probability=tl.where(allowed,tl.exp2(score-lse[:,None]),0.)
        dp=tl.dot(grad,tl.trans(value))
        ds=(probability*(dp-delta[:,None])).to(query.dtype)
        gk=tl.dot(tl.trans(ds),query,gk)
        gv=tl.dot(tl.trans(probability.to(query.dtype)),grad,gv)
    tl.store(GK+(cols[:,None]*H+head)*DQ+dq[None,:],gk*SCALE,cols[:,None]<N)
    tl.store(GV+(cols[:,None]*H+head)*DV+dv[None,:],gv,cols[:,None]<N)


class _TiledAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        if q.dtype!=torch.bfloat16 or q.shape[-1] not in (64,128) or v.shape[-1] not in (64,128):
            raise ValueError('Initial tiled path requires production BF16 widths64/128')
        q,k,v=(x.contiguous() for x in (q,k,v));n,h,dq=q.shape;dv=v.shape[-1]
        ctx.window,ctx.scale=window,scale
        if window==0:
            ctx.save_for_backward(q,k,v)
            return v.clone()
        if window<0:raise ValueError('Local window must be nonnegative')
        doc=torch.bucketize(torch.arange(n,device=q.device),seqlens[1:].contiguous(),right=True).int()
        # Revision5 uses the incumbent training grouping, including bounded
        # groups for irregular document tails. Canonical evaluation is unchanged.
        output=grouped_training_forward(q,k,v,seqlens,window,scale,getattr(ctx,'compact',False),
                                        getattr(ctx,'async_metadata',False))
        lse=torch.empty((n,h),device=q.device,dtype=torch.float32)
        _forward_stats[(triton.cdiv(n,64),h)](q,k,doc,lse,n,h,dq,window,scale,
                                             num_warps=4,num_stages=1)
        ctx.save_for_backward(q,k,v,doc,output,lse)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        if ctx.window==0:
            q,k,v=ctx.saved_tensors
            return torch.zeros_like(q),torch.zeros_like(k),gradient,None,None,None
        q,k,v,doc,output,lse=ctx.saved_tensors;n,h,dq=q.shape;dv=v.shape[-1]
        gradient=gradient.contiguous();delta=torch.empty_like(lse)
        gq,gk,gv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
        grid=(triton.cdiv(n,64),h)
        _row_delta[grid](output,gradient,delta,n,h,dv,num_warps=4,num_stages=1)
        _query_gradient[grid](q,k,v,doc,lse,gradient,delta,gq,n,h,dq,dv,ctx.window,ctx.scale,
                              num_warps=4,num_stages=1)
        _key_value_gradient[grid](q,k,v,doc,lse,gradient,delta,gk,gv,n,h,dq,dv,ctx.window,ctx.scale,
                                  num_warps=4,num_stages=1)
        return gq,gk,gv,None,None,None


def tiled_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():return bucketed_attention(q,k,v,seqlens,window,scale)
    return _TiledAttention.apply(q,k,v,seqlens,window,scale)


class TiledEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(tiled_or_bucketed_eval)


class _CompactTiledAttention(_TiledAttention):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.compact=True
        return _TiledAttention.forward(ctx,q,k,v,seqlens,window,scale)


def compact_tiled_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():return bucketed_attention(q,k,v,seqlens,window,scale)
    return _CompactTiledAttention.apply(q,k,v,seqlens,window,scale)


class CompactTiledEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(compact_tiled_or_bucketed_eval)


class _AsyncTiledAttention(_TiledAttention):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.async_metadata=True
        return _TiledAttention.forward(ctx,q,k,v,seqlens,window,scale)


def async_tiled_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():return bucketed_attention(q,k,v,seqlens,window,scale)
    return _AsyncTiledAttention.apply(q,k,v,seqlens,window,scale)


class AsyncTiledEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(async_tiled_or_bucketed_eval)
    paired_seqlens=staticmethod(paired_boundaries)


def async_tiled_or_metadata_eval(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():
        from control393.eval_attention import metadata_bucketed_attention
        return metadata_bucketed_attention(q,k,v,seqlens,window,scale)
    return _AsyncTiledAttention.apply(q,k,v,seqlens,window,scale)


class AsyncTiledMetadataEvalSelfAttention(AsyncTiledEvalSelfAttention):
    attention=staticmethod(async_tiled_or_metadata_eval)
