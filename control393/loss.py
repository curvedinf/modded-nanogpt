"""PR393 copy-sink mixture and sampled positive LM-head dW, portable BF16."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from control393.copy_pointer import copy_sink_probs
COPY=50257

def sampled_rows(rows,counter,offset,device):
    group=torch.arange(rows//4,device=device,dtype=torch.int64)+offset//4
    z=(group ^ (((counter+72821)*0x9e3779b9)&0xffffffff))&0xffffffff
    z=((z^(z>>16))*0x85ebca6b)&0xffffffff
    z=((z^(z>>13))*0xc2b2ae35)&0xffffffff
    z=z^(z>>16)
    return torch.arange(rows//4,device=device)*4+(z&3)

class Head(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,labels,prefix,copy_col,sample,counter,offset):
        ctx.save_for_backward(x,w,labels,prefix)
        ctx.copy_col,ctx.sample,ctx.counter,ctx.offset=copy_col,sample,counter,offset
        return x@w

    @staticmethod
    def backward(ctx,g):
        x,w,labels,prefix=ctx.saved_tensors;g=g.contiguous()
        dx=g@w.T
        if not ctx.sample:dw=x.T@g
        else:
            selected=sampled_rows(x.shape[0],ctx.counter,ctx.offset,x.device)
            gp=g.index_select(0,selected).clamp_min(0)
            gp[:,ctx.copy_col]=0
            dw=(x.index_select(0,selected).T@gp).float()*4
            # Only label/prefix columns may have negative gradients; handle duplicates once.
            ids=torch.cat([labels,prefix[:,None]],1)
            for i in range(ids.shape[1]):
                col=ids[:,i];valid=(col>=0)&(col!=ctx.copy_col)
                for j in range(i):valid=valid&(col!=ids[:,j])
                safe=col.clamp_min(0)
                val=g.gather(1,safe[:,None])[:,0].float().clamp_max(0)*valid
                dw.index_add_(1,safe,(x.float()*val[:,None]).T)
            dw[:,ctx.copy_col]=(x.float()*g[:,ctx.copy_col,None].float()).sum(0)
            dw=dw.to(w.dtype)
        return dx,dw,None,None,None,None,None,None


class TiledHead(Head):
    @staticmethod
    def forward(ctx,x,w,labels,prefix,copy_col,sample,counter,offset):
        ctx.save_for_backward(x,w,labels,prefix)
        ctx.copy_col,ctx.sample,ctx.counter,ctx.offset=copy_col,sample,counter,offset
        return torch.cat([x[lo:lo+4096]@w for lo in range(0,x.shape[0],4096)])


def evaluation_logits(model,x):
    rows=getattr(model,'head_projection_rows',0)
    if not rows:return model.lm_head(x)
    return torch.cat([model.lm_head(x[lo:lo+rows]) for lo in range(0,x.shape[0],rows)])

def cplm_loss(m,x,inputs,targets,seqlens,mtp,prefix,sampled):
    x=x.reshape(-1,x.shape[-1]);n=x.shape[0]
    q,k=F.linear(x,m.copy_qk_bank.flatten(0,1).to(x.dtype)).chunk(2,-1)
    q=(F.rms_norm(q.float(),(128,))*m.copy_qk_gain).to(x.dtype)
    k=F.rms_norm(k.float(),(128,)).to(x.dtype)
    pc,asin=copy_sink_probs(q,k,m.copy_k_sink,inputs,targets,seqlens,2048 if m.training else 8192)
    if m.training:
        if sampled is None:
            weight=m.lm_head.weight;mapped=targets;ps=m.prefix_table[targets];cc=COPY
        else:
            if hasattr(sampled,'candidate_ids'):
                ids=sampled.candidate_ids;cc=sampled.copy_column
            else:
                ids=(sampled.vocab_pos>=0).nonzero().flatten()
                cc=int(sampled.vocab_pos[COPY])
            weight=m.lm_head.weight.index_select(1,ids)
            mapped=sampled.target_pos[:n];ps=sampled.prefix_pos[:n]
            assert cc>=0
        labels=torch.stack([F.pad(mapped[i:],(0,i),value=-1) for i in range(mtp.numel())],1)
        sampled_dw=m.sampled_head and sampled is not None and n>=32768
        counter=m.head_counter
        compiled_loss=getattr(m,'training_loss_backend','eager')=='compiled_cplm_v1'
        if compiled_loss:
            from control393.training_loss import compiled_slab_loss
            copy_column=torch.tensor(cc,device=x.device,dtype=torch.long)
        head=TiledHead if getattr(m,'head_projection_rows',0)==4096 else Head
        def slab(h,w,ys,pys,p,a,offset):
            logits=head.apply(h,w,ys,pys,cc,sampled_dw,counter,offset)
            if compiled_loss:
                return compiled_slab_loss(logits,ys,pys,p,a,mtp,prefix,copy_column)
            logp=F.log_softmax(23*torch.sigmoid((logits.float()+5)/7.5),-1)
            pl=logp.gather(1,ys[:,0,None])[:,0].exp();alpha=logp[:,cc].exp()
            mixed=pl*(1+alpha*a/(1-alpha).clamp_min(1e-6))+alpha*p
            loss=-mtp[0]*torch.log(mixed+1e-9)
            for i in range(1,ys.shape[1]):
                y=ys[:,i];loss=loss-mtp[i]*logp.gather(1,y.clamp_min(0)[:,None])[:,0]*(y>=0)
            return loss-prefix[0]*logp.gather(1,pys.clamp_min(0)[:,None])[:,0]*(pys>=0)
        return torch.cat([checkpoint(slab,x[lo:lo+m.loss_slab_tokens],weight,
            labels[lo:lo+m.loss_slab_tokens],ps[lo:lo+m.loss_slab_tokens],pc[lo:lo+m.loss_slab_tokens],
            asin[lo:lo+m.loss_slab_tokens],lo,use_reentrant=False) for lo in range(0,n,m.loss_slab_tokens)])
    parts=[];shifts=torch.arange(8,dtype=torch.uint8,device=x.device)
    if getattr(m,'eval_loss_backend','eager')=='fused_canonical_v1':
        from control393.eval_loss import canonical_cplm_loss,softcap_table
        if not hasattr(m,'_eval_softcap_table'):
            m._eval_softcap_table=softcap_table(x.device)
        for lo in range(0,n,m.loss_slab_tokens):
            hi=min(n,lo+m.loss_slab_tokens)
            parts.append(canonical_cplm_loss(evaluation_logits(m,x[lo:hi]),inputs[lo:hi],targets[lo:hi],
                m.canon_mask,pc[lo:hi],asin[lo:hi],m._eval_softcap_table,
                block=getattr(m,'eval_loss_block',4096)))
        return torch.cat(parts)
    for lo in range(0,n,m.loss_slab_tokens):
        hi=min(n,lo+m.loss_slab_tokens)
        logits=(23*torch.sigmoid((evaluation_logits(m,x[lo:hi])+5)/7.5)).float()
        dropped=(m.canon_mask[inputs[lo:hi],:,None]>>shifts&1).view(logits.shape).bool()
        dropped[:,COPY]=False
        logits=logits.masked_fill(dropped,-60.)
        lse=logits.logsumexp(-1)
        pl=(logits.gather(1,targets[lo:hi,None])[:,0]-lse).exp()
        alpha=(logits[:,COPY]-lse).exp()
        p=pl/(1-alpha).clamp_min(1e-6)*(1-alpha*(1-asin[lo:hi]))+alpha*pc[lo:hi]
        parts.append(-p.clamp_min(1e-30).log())
    return torch.cat(parts)
