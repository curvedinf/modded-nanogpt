"""Shape-grouped attention with original-order, non-atomic K/V gradients.

Complete query chunks share exact context lengths. Partial query chunks are
grouped in small context-sorted batches. Every chunk keeps its document mask.
Backward first stores disjoint chunk gradients, then adds overlapping contexts
in reverse original chunk order, rounding after every addition as before.
"""
from collections import defaultdict
from itertools import accumulate
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.autograd.function import once_differentiable
import triton as tr
import triton.language as tl
from ks3_research.attention import chunks_for,BufferedSelfAttention
from mi300.portable import varlen_attention


def grouped_sdpa(q,k,v,mask,scale):
    # Registered all-stage probes151-155 exposed incorrect native SDPA
    # outputs for some heterogeneous mask shapes, unaffected by contiguous
    # head masks. Isolate those groups onto the explicit math implementation.
    if mask.ndim==4:
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scale)
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scale)


def shape_groups(chunks):
    full=defaultdict(list);partial=[]
    for i,(lo,hi,kl) in enumerate(chunks):
        if hi-lo==256:full[hi-kl].append(i)
        else:partial.append(i)
    groups=[ids[start:start+128] for ids in full.values() for start in range(0,len(ids),128)]
    partial.sort(key=lambda i:chunks[i][1]-chunks[i][2])
    groups.extend(partial[start:start+32] for start in range(0,len(partial),32))
    return groups


def metadata_plan(chunks,ids,offsets):
    """CPU-only immutable group coordinates and flattened gather payload."""
    positions=np.asarray([chunks[i] for i in ids],dtype=np.int64)
    lo,hi,kl=(positions[:,i,None] for i in range(3))
    qlen=int((hi-lo).max());klen=int((hi-kl).max());count=len(ids)
    if np.all(hi-lo==qlen) and np.all(hi-kl==klen):
        coordinates=np.column_stack((positions,np.asarray(offsets)[ids]))
        return count,qlen,klen,True,coordinates.reshape(-1),None
    qr=np.arange(qlen,dtype=np.int64);kr=np.arange(klen,dtype=np.int64)
    qa=lo+qr;ka=kl+kr
    qt=np.flatnonzero((qa<hi).reshape(-1));kt=np.flatnonzero((ka<hi).reshape(-1))
    arrays=[positions.reshape(-1),np.minimum(qa,hi-1).reshape(-1),
            np.minimum(ka,hi-1).reshape(-1),qa.reshape(-1)[qt],qt,
            (np.asarray(offsets,dtype=np.int64)[ids,None]+kr).reshape(-1)[kt],kt]
    return count,qlen,klen,False,np.concatenate(arrays),[a.size for a in arrays]


def metadata_from_payload(plan,joined,window,heads=1,compact=False):
    count,qlen,klen,uniform,_,sizes=plan
    device=joined.device
    if uniform:
        # All masks are the same in local coordinates; keep the original
        # 2D mask and exact unpadded gather layout.
        p=joined.view(count,4)
        qr=torch.arange(qlen,device=device);kr=torch.arange(klen,device=device)
        qi=(p[:,0,None]+qr).flatten();ki=(p[:,2,None]+kr).flatten()
        kd=(p[:,3,None]+kr).flatten()
        local_q=qr+klen-qlen
        mask=(kr[None,:]<=local_q[:,None])&(kr[None,:]>=local_q[:,None]-window)
        return count,qlen,klen,qi,ki,mask,None,qi,None,kd,None
    p,qi,ki,qd,qt,kd,kt=joined.split(sizes)
    p=p.view(count,3)
    qa=p[:,0,None]+torch.arange(qlen,device=device)
    ka=p[:,2,None]+torch.arange(klen,device=device)
    valid=qa<p[:,1,None]
    allowed=(ka[:,None,:]<=qa[:,:,None])&(ka[:,None,:]>=qa[:,:,None]-window)
    allowed=allowed&valid[:,:,None]&(ka[:,None,:]<p[:,1,None,None])
    # Retained for the backend-isolation experiment; probe155 showed that
    # contiguous head masks alone do not correct the native implementation.
    mask=allowed[:,None]
    if not compact:mask=mask.expand(-1,heads,-1,-1).contiguous()
    return count,qlen,klen,qi,ki,mask,valid,qd,qt,kd,kt


def group_metadata(device,chunks,ids,offsets,window,heads=1,compact=False):
    plan=metadata_plan(chunks,ids,offsets)
    return metadata_from_payload(plan,torch.from_numpy(plan[4]).to(device),window,heads,compact)


def pack_group(q,k,v,metadata,compact=False):
    count,qlen,klen,qi,ki=metadata[:5]
    heads=q.shape[1];width=max(q.shape[-1],v.shape[-1]);packed=[]
    # ROCm's 64-wide path with heterogeneous per-batch masks failed the
    # independent FP64 real-document oracle (registered probe151). Zero
    # padding to128 preserves the explicit dot-product scale and mathematics.
    if metadata[5].ndim==4 and not compact:width=max(width,128)
    for source,indices,length in ((q,qi,qlen),(k,ki,klen),(v,ki,klen)):
        x=source.index_select(0,indices).view(count,length,heads,source.shape[-1]).transpose(1,2)
        packed.append(F.pad(x,(0,width-source.shape[-1])))
    return packed


def flatten_selected(value,rows,width):
    flat=value.transpose(1,2).reshape(-1,value.shape[1],value.shape[-1])
    return (flat if rows is None else flat.index_select(0,rows))[...,:width]


@tr.jit
def _reverse_context_reduce(DEST,PACKED,STARTS,OFFSETS,FIRST,LAST,
                            HEADS:tl.constexpr,WIDTH:tl.constexpr,
                            D0,D1,D2,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    index=tl.arange(0,BLOCK);valid=index<HEADS*WIDTH
    first=tl.load(FIRST+row);last=tl.load(LAST+row)
    value=tl.full((BLOCK,),0,tl.float32).to(DEST.dtype.element_ty)
    for reverse in range(last-first):
        chunk=last-1-reverse
        lo=tl.load(STARTS+chunk);offset=tl.load(OFFSETS+chunk)
        add=tl.load(PACKED+(offset+row-lo)*HEADS*WIDTH+index,valid,other=0.)
        value=(value.to(tl.float32)+add.to(tl.float32)).to(DEST.dtype.element_ty)
    tl.store(DEST+row*D0+(index//WIDTH)*D1+(index%WIDTH)*D2,value,valid)


def reduction_metadata(chunks,offsets,template):
    values=torch.tensor([[c[2] for c in chunks],[c[1] for c in chunks],offsets[:-1]],device=template.device)
    starts,ends,offsets=values.unbind(0)
    rows=torch.arange(template.shape[0],device=template.device)
    first=torch.searchsorted(ends,rows,right=True)
    last=torch.searchsorted(starts,rows,right=True)
    return starts,offsets,first,last


def reverse_context_reduce(packed,chunks,offsets,template,metadata=None):
    if template.device.type!='cuda':
        result=torch.zeros_like(template)
        for i in reversed(range(len(chunks))):
            _,hi,lo=chunks[i]
            result[lo:hi].add_(packed[offsets[i]:offsets[i]+hi-lo])
        return result
    if template.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('GPU ordered reduction supports FP32/BF16')
    result=torch.empty_like(template)
    starts,offsets,first,last=metadata if metadata is not None else reduction_metadata(chunks,offsets,template)
    heads,width=template.shape[1:]
    _reverse_context_reduce[(template.shape[0],)](result,packed,starts,offsets,first,last,
        heads,width,*result.stride(),tr.next_power_of_2(heads*width))
    return result


class _BucketTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,seqlens,window,scale):
        ctx.chunks=chunks_for(seqlens.detach().cpu().tolist(),window)
        ctx.offsets=[0,*accumulate(hi-kl for _,hi,kl in ctx.chunks)]
        ctx.layouts=[];ctx.window=window;ctx.scale=scale
        ctx.save_for_backward(q,k,v)
        result=torch.empty_like(v)
        for ids in shape_groups(ctx.chunks):
            metadata=group_metadata(q.device,ctx.chunks,ids,ctx.offsets,window,q.shape[1])
            ctx.layouts.append(metadata)
            qp,kp,vp=pack_group(q,k,v,metadata)
            _,_,_,_,_,mask,_,qd,qt,_,_=metadata
            y=grouped_sdpa(qp,kp,vp,mask,scale)
            result.index_copy_(0,qd,flatten_selected(y,qt,v.shape[-1]))
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,grad_output):
        q,k,v=ctx.saved_tensors
        dq=torch.zeros_like(q)
        pk=k.new_empty(ctx.offsets[-1],*k.shape[1:])
        pv=v.new_empty(ctx.offsets[-1],*v.shape[1:])
        for metadata in reversed(ctx.layouts):
            qp,kp,vp=pack_group(q,k,v,metadata)
            _,_,_,qi,_,mask,valid,qd,qt,kd,kt=metadata
            count,heads,qlen,width=qp.shape
            upstream=grad_output.index_select(0,qi).view(count,qlen,heads,v.shape[-1]).transpose(1,2)
            upstream=F.pad(upstream,(0,width-v.shape[-1]))
            if valid is not None:upstream.masked_fill_(~valid[:,None,:,None],0.)
            with torch.enable_grad():
                qp.requires_grad_();kp.requires_grad_();vp.requires_grad_()
                y=grouped_sdpa(qp,kp,vp,mask,ctx.scale)
                gq,gk,gv=torch.autograd.grad(y,(qp,kp,vp),upstream)
            dq.index_copy_(0,qd,flatten_selected(gq,qt,q.shape[-1]))
            pk.index_copy_(0,kd,flatten_selected(gk,kt,k.shape[-1]))
            pv.index_copy_(0,kd,flatten_selected(gv,kt,v.shape[-1]))
        metadata=reduction_metadata(ctx.chunks,ctx.offsets,k) if k.device.type=='cuda' else None
        return (dq,reverse_context_reduce(pk,ctx.chunks,ctx.offsets,k,metadata),
                reverse_context_reduce(pv,ctx.chunks,ctx.offsets,v,metadata),None,None,None)


def bucket_training_attention(q,k,v,seqlens,window,scale):
    if not torch.is_grad_enabled():return varlen_attention(q,k,v,seqlens,window,scale)
    return _BucketTraining.apply(q,k,v,seqlens,window,scale)


class BucketTrainingSelfAttention(BufferedSelfAttention):
    attention=staticmethod(bucket_training_attention)


def training_buckets_or_eval(q,k,v,seqlens,window,scale):
    if torch.is_grad_enabled():return bucket_training_attention(q,k,v,seqlens,window,scale)
    from control393.eval_attention import bucketed_attention
    return bucketed_attention(q,k,v,seqlens,window,scale)


class BucketTrainingEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(training_buckets_or_eval)
