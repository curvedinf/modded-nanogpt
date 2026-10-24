"""Batch equal-shaped canonical evaluation chunks without padding their contexts."""
from collections import defaultdict
import torch
import torch.nn.functional as F
from ks3_research.attention import BufferedSelfAttention, chunks_for
from control393.attention128 import fused128_attention


@torch.no_grad()
def bucketed_attention(q,k,v,seqlens,window,scale,group_size=128):
    buckets=defaultdict(list)
    for lo,hi,kl in chunks_for(seqlens.detach().cpu().tolist(),window):
        buckets[hi-lo,hi-kl].append((lo,kl))
    heads=q.shape[1];width=max(q.shape[-1],v.shape[-1])
    result=torch.empty_like(v)
    for (qlen,klen),chunks in buckets.items():
        qr=torch.arange(qlen,device=q.device)
        kr=torch.arange(klen,device=q.device)
        local_q=qr+klen-qlen
        allowed=(kr[None,:]<=local_q[:,None])&(kr[None,:]>=local_q[:,None]-window)
        for start in range(0,len(chunks),group_size):
            group=chunks[start:start+group_size];count=len(group)
            positions=torch.tensor(group,device=q.device)
            qi=(positions[:,0,None]+qr).flatten()
            ki=(positions[:,1,None]+kr).flatten()
            packed=[]
            for source,indices,length in ((q,qi,qlen),(k,ki,klen),(v,ki,klen)):
                x=source.index_select(0,indices).view(count,length,heads,source.shape[-1]).transpose(1,2)
                packed.append(F.pad(x,(0,width-source.shape[-1])))
            y=F.scaled_dot_product_attention(*packed,attn_mask=allowed,dropout_p=0.,scale=scale)
            y=y.transpose(1,2).reshape(-1,heads,width)[...,:v.shape[-1]]
            result.index_copy_(0,qi,y)
    return result


@torch.no_grad()
def metadata_bucketed_attention(q,k,v,seqlens,window,scale,group_size=128):
    """Same SDPA groups/masks, with one upload and views for singleton groups."""
    from control393.attention_metadata import host_boundaries
    buckets=defaultdict(list)
    for lo,hi,kl in chunks_for(host_boundaries(seqlens),window):
        buckets[hi-lo,hi-kl].append((lo,kl))
    groups=[(qlen,klen,chunks[start:start+group_size])
            for (qlen,klen),chunks in buckets.items()
            for start in range(0,len(chunks),group_size)]
    positions=[position for _,_,group in groups if len(group)>1 for position in group]
    uploaded=None
    if positions:
        host=torch.tensor(positions,dtype=torch.int64)
        if q.device.type=='cuda':host=host.pin_memory()
        uploaded=host.to(q.device,non_blocking=True)
    offset=0;previous=None
    heads=q.shape[1];width=max(q.shape[-1],v.shape[-1])
    result=torch.empty_like(v)
    for qlen,klen,group in groups:
        count=len(group)
        if previous!=(qlen,klen):
            qr=torch.arange(qlen,device=q.device);kr=torch.arange(klen,device=q.device)
            local_q=qr+klen-qlen
            allowed=(kr[None,:]<=local_q[:,None])&(kr[None,:]>=local_q[:,None]-window)
            previous=qlen,klen
        if count==1:
            lo,kl=group[0]
            sources=(q[lo:lo+qlen],k[kl:kl+klen],v[kl:kl+klen])
            packed=[source.unsqueeze(0).transpose(1,2) for source in sources]
        else:
            p=uploaded[offset:offset+count];offset+=count
            qi=(p[:,0,None]+qr).flatten();ki=(p[:,1,None]+kr).flatten()
            packed=[source.index_select(0,indices).view(count,length,heads,source.shape[-1]).transpose(1,2)
                    for source,indices,length in ((q,qi,qlen),(k,ki,klen),(v,ki,klen))]
        packed=[F.pad(x,(0,width-x.shape[-1])) if x.shape[-1]!=width else x for x in packed]
        y=F.scaled_dot_product_attention(*packed,attn_mask=allowed,dropout_p=0.,scale=scale)
        y=y.transpose(1,2).reshape(-1,heads,width)[...,:v.shape[-1]]
        if count==1:result[lo:lo+qlen].copy_(y)
        else:result.index_copy_(0,qi,y)
    return result


def training_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if torch.is_grad_enabled():
        return fused128_attention(q,k,v,seqlens,window,scale)
    return bucketed_attention(q,k,v,seqlens,window,scale)


class BucketedEvalSelfAttention(BufferedSelfAttention):
    attention=staticmethod(training_or_bucketed_eval)


def training256_or_bucketed_eval(q,k,v,seqlens,window,scale):
    if torch.is_grad_enabled():
        from control393.attention256 import fused256_attention
        return fused256_attention(q,k,v,seqlens,window,scale)
    return bucketed_attention(q,k,v,seqlens,window,scale)


class BucketedEval256SelfAttention(BufferedSelfAttention):
    attention=staticmethod(training256_or_bucketed_eval)
