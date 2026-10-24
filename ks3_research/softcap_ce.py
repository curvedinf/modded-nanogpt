"""Row-fused FP32 softcap, MTP/prefix cross entropy, and logit gradient.

The mathematical objective is the existing portable training loss. Validation
continues to use the original canonical mask and BF16 softcap path.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _loss_forward(Z, LABELS, PREFIX, MW, PW, LOSS,
                  VOCAB: tl.constexpr, MTP: tl.constexpr, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK)
    z=tl.load(Z+row*VOCAB+cols,cols<VOCAB,other=0.).to(tl.float32)
    capped=23.*tl.sigmoid((z+5.)/7.5)
    capped=tl.where(cols<VOCAB,capped,float('-inf'))
    maximum=tl.max(capped,0)
    logsum=tl.log(tl.sum(tl.exp(capped-maximum),0))
    loss=tl.full((),0.,tl.float32)
    for i in tl.static_range(MTP):
        y=tl.load(LABELS+row*MTP+i)
        w=tl.load(MW+i)
        selected=tl.load(Z+row*VOCAB+tl.maximum(y,0)).to(tl.float32)
        selected=23.*tl.sigmoid((selected+5.)/7.5)
        loss+=tl.where(y>=0,w*((maximum-selected)+logsum),0.)
    y=tl.load(PREFIX+row)
    w=tl.load(PW)
    selected=tl.load(Z+row*VOCAB+tl.maximum(y,0)).to(tl.float32)
    selected=23.*tl.sigmoid((selected+5.)/7.5)
    loss+=tl.where(y>=0,w*((maximum-selected)+logsum),0.)
    tl.store(LOSS+row,loss)


@tr.jit
def _loss_backward(Z,LABELS,PREFIX,MW,PW,UPSTREAM,DZ,
                   VOCAB: tl.constexpr,MTP: tl.constexpr,BLOCK: tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK)
    z=tl.load(Z+row*VOCAB+cols,cols<VOCAB,other=0.).to(tl.float32)
    sigmoid=tl.sigmoid((z+5.)/7.5)
    capped=tl.where(cols<VOCAB,23.*sigmoid,float('-inf'))
    exp=tl.exp(capped-tl.max(capped,0))
    probability=exp/tl.sum(exp,0)
    mass=tl.full((),0.,tl.float32)
    selected=tl.full((BLOCK,),0.,tl.float32)
    for i in tl.static_range(MTP):
        y=tl.load(LABELS+row*MTP+i)
        w=tl.load(MW+i)
        mass+=tl.where(y>=0,w,0.)
        selected+=tl.where((cols==y)&(y>=0),w,0.)
    y=tl.load(PREFIX+row)
    w=tl.load(PW)
    mass+=tl.where(y>=0,w,0.)
    selected+=tl.where((cols==y)&(y>=0),w,0.)
    derivative=(probability*mass-selected)*((23./7.5)*sigmoid*(1.-sigmoid))
    derivative*=tl.load(UPSTREAM+row)
    tl.store(DZ+row*VOCAB+cols,derivative,cols<VOCAB)


class _SoftcapCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,labels,prefixes,mtp,prefix_weight):
        logits,labels,prefixes=logits.contiguous(),labels.contiguous(),prefixes.contiguous()
        rows,vocab=logits.shape
        ctx.save_for_backward(logits,labels,prefixes,mtp,prefix_weight)
        loss=torch.empty(rows,device=logits.device,dtype=torch.float32)
        block=tr.next_power_of_2(vocab)
        _loss_forward[(rows,)](logits,labels,prefixes,mtp,prefix_weight,loss,
                               vocab,labels.shape[1],block,num_warps=16 if block>=32768 else 8)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        logits,labels,prefixes,mtp,prefix_weight=ctx.saved_tensors
        rows,vocab=logits.shape
        gradient=torch.empty_like(logits)
        block=tr.next_power_of_2(vocab)
        _loss_backward[(rows,)](logits,labels,prefixes,mtp,prefix_weight,upstream.contiguous(),gradient,
                                vocab,labels.shape[1],block,num_warps=16 if block>=32768 else 8)
        return gradient,None,None,None,None


def softcap_ce(logits,labels,prefixes,mtp,prefix_weight):
    if mtp.requires_grad or prefix_weight.requires_grad:
        raise ValueError('Training objective schedule weights must be constants')
    if logits.device.type=='cuda':
        return _SoftcapCE.apply(logits,labels,prefixes,mtp,prefix_weight)
    # CPU path permits real wrapper/stream-boundary regression tests; registered
    # GPU diagnostics separately exercise the actual fused kernels.
    logp=(23*torch.sigmoid((logits.float()+5)/7.5)).log_softmax(-1)
    loss=torch.zeros(logits.shape[0],device=logits.device)
    for i in range(labels.shape[1]):
        y=labels[:,i]
        loss-=mtp[i]*logp.gather(1,y.clamp_min(0)[:,None]).squeeze(1)*(y>=0)
    return loss-prefix_weight[0]*logp.gather(1,prefixes.clamp_min(0)[:,None]).squeeze(1)*(prefixes>=0)
