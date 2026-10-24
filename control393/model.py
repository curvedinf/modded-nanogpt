"""Full PR393 topology and top-1 MoE, using portable MI300 projections."""
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from control393.base_gpt import GPT, NUM_MLP_SLOTS
from control393.loss import cplm_loss

MOE_LAYERS=(1,3,5,6,8)
DENSE_SLOTS=(0,2,4,10,11)
EXPERTS=8
HIDDEN=1024

def norm_map():
    a=np.load(Path(__file__).with_name('norm_map_pr375.npy'))
    assert hashlib.sha256(a[:50257].astype('<i4').tobytes()).hexdigest()=='79fb57ac259495ba36f1afb9b0d5da35f3048b5067fa59c101d5a3c1ed15c2a2'
    return torch.from_numpy(a)

def routed_mlp(x,up,down,router):
    """Dropless top-1 with probability scaling and exact Switch auxiliary loss."""
    shape=x.shape; flat=x.reshape(-1,x.shape[-1])
    probs=F.softmax(F.linear(flat,router.to(flat.dtype)).float(),dim=-1)
    gates,experts=probs.max(-1)
    order=torch.argsort(experts,stable=True)
    counts=torch.bincount(experts,minlength=EXPERTS)
    sizes=counts.cpu().tolist()
    ordered=flat.index_select(0,order)
    parts=[];lo=0
    for e,count in enumerate(sizes):
        h=ordered[lo:lo+count];lo+=count
        def f(h,u,d):return F.relu(F.linear(h,u)).square()@d
        if torch.is_grad_enabled():parts.append(checkpoint(f,h,up[e],down[e],use_reentrant=False))
        else:parts.append(f(h,up[e],down[e]))
    ys=torch.cat(parts)
    y=torch.empty_like(ys).index_copy(0,order,ys)
    aux=EXPERTS*(counts.float()/flat.shape[0]*probs.mean(0)).sum()
    return (y*gates[:,None].to(y.dtype)).reshape(shape),aux

class MoEGPT(GPT):
    def __init__(self,*args,attention_backend='gathered_sdpa32_fused',loss_slab_tokens=4096,
                 sampled_head=True,**kwargs):
        super().__init__(*args,**kwargs)
        from ks3_research.attention import FusedAccumulationSelfAttention,Fused64SelfAttention
        from control393.attention128 import Fused128SelfAttention
        from control393.eval_attention import BucketedEvalSelfAttention,BucketedEval256SelfAttention
        from control393.bucket_training import BucketTrainingEvalSelfAttention
        from control393.tiled_attention import TiledEvalSelfAttention,CompactTiledEvalSelfAttention,AsyncTiledEvalSelfAttention
        from control393.tiled_attention import AsyncTiledMetadataEvalSelfAttention
        cls={'gathered_sdpa32_fused':FusedAccumulationSelfAttention,
             'gathered_sdpa64_fused':Fused64SelfAttention,
             'gathered_sdpa128_fused':Fused128SelfAttention,
             'gathered_sdpa128_evalbucket':BucketedEvalSelfAttention,
             'gathered_sdpa256_evalbucket':BucketedEval256SelfAttention,
             'gathered_sdpa128_trainbucket':BucketTrainingEvalSelfAttention,
             'gathered_sdpa128_tiledbackward':TiledEvalSelfAttention,
             'gathered_sdpa128_tiledcompact':CompactTiledEvalSelfAttention,
             'gathered_sdpa128_tiledasync':AsyncTiledEvalSelfAttention,
             'gathered_sdpa128_tiledasync_fasteval':AsyncTiledMetadataEvalSelfAttention}[attention_backend]
        for slot,layer in list(self.attn.items()):
            self.attn[slot]=cls(layer.num_heads,layer.head_dim,layer.qk_dim,layer.v_dim,
                               layer.val_max_seq_len,paired=layer.paired)
        dim=self.mlp_bank.shape[-1]
        self.copy_qk_bank=nn.Parameter(torch.empty(2,128,dim).uniform_(-math.sqrt(3/dim),math.sqrt(3/dim)))
        self.copy_k_sink=nn.Parameter(torch.randn(128)*128**-.5)
        self.copy_qk_gain=nn.Parameter(torch.ones(1))
        self.register_buffer('ng_norm_map',norm_map().to(self.device),persistent=True)
        self.attention_backend=attention_backend
        self.loss_backend='cplm_pr393';self.loss_slab_tokens=loss_slab_tokens
        self.sampled_head=sampled_head;self.head_counter=0
        self.copy_eval_block=8192
        for name,p in self.named_parameters():p.label=name.replace('.weight','')

    def init_mlp(self,dim):
        self.expert_hidden=HIDDEN
        super().init_mlp(dim)
        self.mlp_bank=nn.Parameter(self.mlp_bank.detach()[list(DENSE_SLOTS)].clone())
        self.mlp_bank.reshape=(2*len(DENSE_SLOTS),2816,dim)
        self.moe_bank=nn.Parameter(torch.empty(len(MOE_LAYERS)*2*EXPERTS,HIDDEN,dim))
        self.moe_bank.reshape=tuple(self.moe_bank.shape)
        with torch.no_grad():
            for k in range(len(MOE_LAYERS)):
                self.moe_bank[k*16:k*16+8].uniform_(-.5*math.sqrt(3/dim),.5*math.sqrt(3/dim))
                self.moe_bank[k*16+8:(k+1)*16].zero_()
        self.moe_router=nn.Parameter(.01*torch.randn(len(MOE_LAYERS),EXPERTS,dim))

    def _mlp_weights(self):
        w=self.mlp_bank.flatten(0,1).unbind(0)
        up,down=[None]*NUM_MLP_SLOTS,[None]*NUM_MLP_SLOTS
        for i,slot in enumerate(DENSE_SLOTS):up[slot],down[slot]=w[2*i:2*i+2]
        return up,down

    def _mlp(self,x,c_fc,c_proj,*,slot=None):
        if slot not in MOE_LAYERS:return super()._mlp(x,c_fc,c_proj,slot=slot)
        k=MOE_LAYERS.index(slot);u,d,r=self._expert_views[k]
        y,aux=routed_mlp(x,u,d,r)
        if self.training:self._moe_aux.append(aux)
        return y

    def forward(self,input_seq,target_seq,seqlens,*args,**kwargs):
        self._seqlens=seqlens;self._moe_aux=[]
        self._expert_views=[(*w.unbind(0),router) for w,router in zip(
            self.moe_bank.view(5,2,8,self.expert_hidden,-1).unbind(0),self.moe_router.unbind(0))]
        loss=super().forward(input_seq,target_seq,seqlens,*args,**kwargs)
        if self.training:
            loss=loss+.01*sum(self._moe_aux)
            self.head_counter+=8
        return loss

    def _loss(self,x,inputs,targets,mtp,prefix,sampled):
        return cplm_loss(self,x,inputs,targets,self._seqlens,mtp,prefix,sampled)

    def cast_matrix_weights_bf16(self):
        super().cast_matrix_weights_bf16()
        self.moe_bank.data=self.moe_bank.data.bfloat16()
        self.copy_qk_bank.data=self.copy_qk_bank.data.bfloat16()

    def ffn_parameters(self):return self.mlp_bank.numel()+self.moe_bank.numel()+self.moe_router.numel()
