"""Evaluation-only canonical CPLM reduction with exact BF16 softcap lookup.

The table evaluates all 65536 BF16 bit patterns with the original eager torch
expression. It preserves each intermediate BF16 rounding instead of silently
replacing it with a fused FP32 softcap. Only the FP32 logsumexp reduction order
changes. Training and the copy-pointer probabilities are untouched.
"""
import torch
import triton
import triton.language as tl


@torch.no_grad()
def softcap_table(device):
    values=torch.arange(65536,device=device,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return (23*torch.sigmoid((values+5)/7.5)).float()


@triton.jit
def _partial_lse(Z, TABLE, INPUT, MASK, OUT, V:tl.constexpr, STRIDE:tl.constexpr,
                 PARTS:tl.constexpr, COPY:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0);part=tl.program_id(1)
    cols=part*BLOCK+tl.arange(0,BLOCK)
    bits=tl.load(Z+row*STRIDE+cols,cols<V,other=0).to(tl.uint16,bitcast=True).to(tl.int32)
    value=tl.load(TABLE+bits)
    prev=tl.load(INPUT+row)
    packed=tl.load(MASK+prev*(V//8)+cols//8,cols<V,other=0).to(tl.int32)
    dropped=((packed>>(cols%8))&1)!=0
    value=tl.where(dropped&(cols!=COPY),-60.,value)
    value=tl.where(cols<V,value,float('-inf'))
    maximum=tl.max(value,0)
    lse=maximum+tl.log(tl.sum(tl.exp(value-maximum),0))
    tl.store(OUT+row*PARTS+part,lse)


@torch.no_grad()
def canonical_cplm_loss(logits,inputs,targets,mask,pc,asin,table,*,copy_col=50257,block=4096):
    if logits.dtype!=torch.bfloat16 or logits.stride(1)!=1:
        raise ValueError('Fused canonical evaluation requires contiguous BF16 vocabulary rows')
    n,vocab=logits.shape
    if vocab%8 or mask.shape[1]!=vocab//8 or not 0<=copy_col<vocab:
        raise ValueError('Canonical mask and copy column must match the full vocabulary')
    if block<256 or block&(block-1):
        raise ValueError('Reduction block must be a power of two >=256')
    parts=triton.cdiv(vocab,block)
    partial=torch.empty((n,parts),device=logits.device,dtype=torch.float32)
    _partial_lse[(n,parts)](logits,table,inputs,mask,partial,vocab,logits.stride(0),
                           parts,copy_col,block,num_warps=16 if block>=32768 else 4)
    lse=partial[:,0] if parts==1 else partial.logsumexp(-1)
    selected=torch.stack((logits.gather(1,targets[:,None])[:,0],logits[:,copy_col]),1)
    transformed=table[selected.contiguous().view(torch.int16).long()&65535]
    dropped=((mask[inputs,targets//8].int()>>(targets%8))&1).bool()&(targets!=copy_col)
    pl=(transformed[:,0].masked_fill(dropped,-60.)-lse).exp()
    alpha=(transformed[:,1]-lse).exp()
    p=pl/(1-alpha).clamp_min(1e-6)*(1-alpha*(1-asin))+alpha*pc
    return -p.clamp_min(1e-30).log()
