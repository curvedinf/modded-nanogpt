"""MI300 causal/window varlen attention with mixed QK/V widths and device-only metadata.

The kernels preserve document masks and the original attention scale/window.
Backward derives from the locally verified control393 tiled attention kernels.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

@triton.jit
def _forward(Q,K,V,DOC,O,LSE,N:tl.constexpr,H:tl.constexpr,DQ:tl.constexpr,DV:tl.constexpr,
             WINDOW:tl.constexpr,SCALE:tl.constexpr,B:tl.constexpr=64):
    block=tl.program_id(0);head=tl.program_id(1)
    rows=block*B+tl.arange(0,B);dq=tl.arange(0,DQ);dv=tl.arange(0,DV)
    q=tl.load(Q+(rows[:,None]*H+head)*DQ+dq[None,:],rows[:,None]<N,other=0.)
    docs=tl.load(DOC+rows,rows<N,other=-1)
    maximum=tl.full((B,),-1.e6,tl.float32);denom=tl.zeros((B,),tl.float32)
    acc=tl.zeros((B,DV),tl.float32)
    first=tl.maximum(0,block*B-WINDOW)//B*B
    for start in range(first,(block+1)*B,B):
        cols=start+tl.arange(0,B)
        k=tl.load(K+(cols[None,:]*H+head)*DQ+dq[:,None],cols[None,:]<N,other=0.)
        v=tl.load(V+(cols[:,None]*H+head)*DV+dv[None,:],cols[:,None]<N,other=0.)
        kd=tl.load(DOC+cols,cols<N,other=-2)
        mask=(rows[:,None]<N)&(cols[None,:]<N)&(cols[None,:]<=rows[:,None])
        mask=mask&(cols[None,:]>=rows[:,None]-WINDOW)&(docs[:,None]==kd[None,:])
        score=tl.where(mask,tl.dot(q,k)*(SCALE*1.4426950408889634),-1.e6)
        newmax=tl.maximum(maximum,tl.max(score,1))
        p=tl.where(mask,tl.exp2(score-newmax[:,None]),0.)
        alpha=tl.exp2(maximum-newmax)
        denom=denom*alpha+tl.sum(p,1)
        acc=acc*alpha[:,None]
        acc=tl.dot(p.to(v.dtype),v,acc)
        maximum=newmax
    tl.store(O+(rows[:,None]*H+head)*DV+dv[None,:],acc/denom[:,None],rows[:,None]<N)
    tl.store(LSE+rows*H+head,maximum+tl.log2(denom),rows<N)

@triton.jit(do_not_specialize=['N'])
def _row_delta(OUT,GRAD,DELTA,N,H:tl.constexpr,DV:tl.constexpr,B:tl.constexpr=64):
    rows=tl.program_id(0)*B+tl.arange(0,B);head=tl.program_id(1);dims=tl.arange(0,DV)
    offsets=(rows[:,None]*H+head)*DV+dims[None,:]
    out=tl.load(OUT+offsets,rows[:,None]<N,other=0.).to(tl.float32)
    grad=tl.load(GRAD+offsets,rows[:,None]<N,other=0.).to(tl.float32)
    tl.store(DELTA+rows*H+head,tl.sum(out*grad,1),rows<N)


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


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,cu,window,scale):
        q,k,v=[x.contiguous() for x in (q,k,v)]
        n,h,dq=q.shape;dv=v.shape[-1]
        doc=torch.bucketize(torch.arange(n,device=q.device),cu[1:].contiguous(),right=True).int()
        out=torch.empty_like(v);lse=torch.empty((n,h),device=q.device,dtype=torch.float32)
        _forward[(triton.cdiv(n,64),h)](q,k,v,doc,out,lse,n,h,dq,dv,window,scale,B=64,num_warps=4,num_stages=1)
        ctx.save_for_backward(q,k,v,doc,out,lse);ctx.window=window;ctx.scale=scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        q,k,v,doc,output,lse=ctx.saved_tensors;n,h,dq=q.shape;dv=v.shape[-1]
        gradient=gradient.contiguous();delta=torch.empty_like(lse)
        gq,gk,gv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
        grid=(triton.cdiv(n,64),h)
        _row_delta[grid](output,gradient,delta,n,h,dv,B=64,num_warps=4,num_stages=1)
        _query_gradient[grid](q,k,v,doc,lse,gradient,delta,gq,n,h,dq,dv,ctx.window,ctx.scale,B=64,num_warps=4,num_stages=1)
        _key_value_gradient[grid](q,k,v,doc,lse,gradient,delta,gk,gv,n,h,dq,dv,ctx.window,ctx.scale,B=64,num_warps=4,num_stages=1)
        return gq,gk,gv,None,None,None


def flash_attn_varlen_func(q,k,v,*,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,
                          causal,softmax_scale,window_size):
    if not causal or window_size[1]!=0:raise ValueError('Unsupported record attention invocation')
    if q.shape[-1] not in (64,128) or v.shape[-1] not in (64,128):raise ValueError('Unexpected record head width')
    window=q.shape[0] if window_size[0]<0 else window_size[0]
    return _Attention.apply(q,k,v,cu_seqlens_q,window,softmax_scale)
