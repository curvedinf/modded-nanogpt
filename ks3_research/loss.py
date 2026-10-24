"""Portable training objective with shared vocabulary gather and larger slabs.

Derived from mi300.portable.training_loss. Targets still shift across the packed
stream, including slab/document boundaries; validation is not routed here.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def shared_gather_loss(x,weight,targets,prefix_table,mtp_weights,prefix_weight,sampled,slab_tokens=4096,
                       fused_softcap=False):
    n=x.shape[0]
    if slab_tokens<1:
        raise ValueError('Loss slab must have a positive token count')
    if sampled is None:
        selected=weight
        mapped,prefixes=targets,prefix_table[targets]
    else:
        candidate_ids=(sampled.vocab_pos>=0).nonzero().flatten()
        selected=weight.index_select(1,candidate_ids)
        mapped,prefixes=sampled.target_pos[:n],sampled.prefix_pos[:n]
    labels=torch.stack([F.pad(mapped[i:],(0,i),value=-1) for i in range(mtp_weights.numel())],1)

    def slab(h,w,ys,ps,mw,pw):
        logits=h@w
        if fused_softcap:
            from ks3_research.softcap_ce import softcap_ce
            return softcap_ce(logits,ys,ps,mw,pw)
        capped=23*torch.sigmoid((logits.float()+5)/7.5)
        logp=F.log_softmax(capped,dim=-1)
        loss=torch.zeros(h.shape[0],dtype=torch.float32,device=h.device)
        for i in range(ys.shape[1]):
            y=ys[:,i]
            loss=loss-mw[i]*logp.gather(1,y.clamp_min(0)[:,None]).squeeze(1)*(y>=0)
        return loss-pw[0]*logp.gather(1,ps.clamp_min(0)[:,None]).squeeze(1)*(ps>=0)

    return torch.cat([checkpoint(slab,x[lo:lo+slab_tokens],selected,labels[lo:lo+slab_tokens],
                                 prefixes[lo:lo+slab_tokens],mtp_weights,prefix_weight,use_reentrant=False)
                      for lo in range(0,n,slab_tokens)])
