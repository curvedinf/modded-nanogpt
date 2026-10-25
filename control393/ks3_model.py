"""Same PR393 MoE routing/topology and dense branches, with KS3 experts."""
import torch
from torch import nn
import torch.nn.functional as F
from control393.model import MoEGPT,MOE_LAYERS
from ks3_research.vendor.ks3_int8 import ks3_int8

class MoEKS3GPT(MoEGPT):
    def __init__(self,*args,charts=1024,rank=1,dynamic_rows=False,bf16_products=False,expert_unbind=False,
                 dense_pointwise=False,**kwargs):
        if rank!=1:raise ValueError('Initial MoE KS3 implementation supports rank1; retain routing topology when extending ranks')
        if dynamic_rows or bf16_products:
            raise ValueError('Use the user-selected upstream ks3_int8 kernel unchanged; legacy dynamic_rows/BF16 BLAS replacements are forbidden')
        self.ks3_charts=charts
        self.ks3_dynamic_rows=dynamic_rows
        self.ks3_bf16_products=bf16_products
        self.ks3_expert_unbind=expert_unbind
        self.dense_pointwise=dense_pointwise
        super().__init__(*args,**kwargs)
        self.register_buffer('ks3_tanh_lut',(torch.tanh(torch.arange(-128,128,device=self.device).float()*(3.2/127.))*127).round().clamp(-127,127).to(torch.int8),persistent=False)

    def init_mlp(self,dim):
        super().init_mlp(dim)
        c=self.ks3_charts
        if c<=1024:
            bank=self.moe_bank.detach().view(5,2,8,1024,dim)[:,:,:,:c].clone()
        else:
            with torch.random.fork_rng(devices=[]):
                bank=torch.empty(5,2,8,c,dim)
                bank[:,0].uniform_(-.5*(3/dim)**.5,.5*(3/dim)**.5);bank[:,1].zero_()
        self.moe_bank=nn.Parameter(bank.reshape(80,c,dim))
        self.moe_bank.reshape=(80,c,dim);self.expert_hidden=c
        self.ks3_coefficients=nn.Parameter(torch.zeros(5,8,c,1,6))
        self.ks3_scale=nn.Parameter(torch.full((5,8,c),.75))
        self.ks3_shift=nn.Parameter(torch.zeros(5,8,c))

    def cast_matrix_weights_bf16(self):
        master=self.moe_bank.data
        super().cast_matrix_weights_bf16()
        self.moe_bank.data=master

    def _mlp(self,x,c_fc,c_proj,*,slot=None):
        if slot not in MOE_LAYERS:
            if self.dense_pointwise:
                from control393.dense_pointwise import dense_pointwise_mlp
                return dense_pointwise_mlp(x,c_fc,c_proj)
            return super()._mlp(x,c_fc,c_proj,slot=slot)
        k=MOE_LAYERS.index(slot);up,down,router=self._expert_views[k]
        flat=x.reshape(-1,x.shape[-1])
        probs=F.softmax(F.linear(flat,router.to(flat.dtype)).float(),-1)
        gates,experts=probs.max(-1);order=torch.argsort(experts,stable=True)
        counts=torch.bincount(experts,minlength=8);sizes=counts.cpu().tolist()
        ordered=flat.index_select(0,order);parts=[];lo=0
        if self.ks3_expert_unbind:
            # One stacking backward per bank instead of eight full-bank
            # select backwards. These remain views of the same parameters.
            up,down=up.unbind(0),down.unbind(0)
            coefficients=self.ks3_coefficients[k].unbind(0)
            scales=self.ks3_scale[k].unbind(0)
            shifts=self.ks3_shift[k].unbind(0)
        for e,count in enumerate(sizes):
            h=ordered[lo:lo+count];lo+=count
            if self.ks3_expert_unbind:
                coeff,scale,shift=coefficients[e],scales[e],shifts[e]
            else:
                coeff=self.ks3_coefficients[k,e];scale=self.ks3_scale[k,e];shift=self.ks3_shift[k,e]
            if count:
                y=ks3_int8(h,up[e],down[e].T,coeff,scale,shift,
                    subspace_rank=1,nonlinear='fp16',tanh_lut=self.ks3_tanh_lut)
            else:y=h+(up[e].sum()+down[e].sum()+coeff.sum()+scale.sum()+shift.sum()).to(h.dtype)*0
            parts.append(y)
        ys=torch.cat(parts);y=torch.empty_like(ys).index_copy(0,order,ys)
        if self.training:self._moe_aux.append(8*(counts.float()/flat.shape[0]*probs.mean(0)).sum())
        return (y*gates[:,None].to(y.dtype)).reshape(x.shape)

    def ffn_parameters(self):
        return super().ffn_parameters()+sum(p.numel() for p in [self.ks3_coefficients,self.ks3_scale,self.ks3_shift])
