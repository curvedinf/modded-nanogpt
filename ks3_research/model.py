"""Full-width KS3 FFNs in the existing MI300 transformer.

The cell and its W8A8 kernels come from curvedinf/k-splanifolds; see vendor/PROVENANCE.json.
Chart count is a research variable, not a parameter-budget constraint.
"""
import math
import torch
from torch import nn
from mi300.model.gpt import GPT, NO_MLP_LAYERS, NUM_MLP_SLOTS
from ks3_research.vendor.ks3_int8 import ks3_int8

ACTIVE_SLOTS = tuple(i for i in range(NUM_MLP_SLOTS) if i not in NO_MLP_LAYERS)


class KS3GPT(GPT):
    def __init__(self, *args, charts=1536, rank=1, nonlinear='fp16', attention_backend='original_sdpa',
                 loss_backend='original',loss_slab_tokens=1024,ks3_backward='vendor',
                 ks3_forward='original', **kwargs):
        if charts < 1 or rank < 1:
            raise ValueError('KS3 chart count and rank must be positive')
        self.ks3_charts, self.ks3_rank = charts, rank
        self.ks3_nonlinear = nonlinear
        self.slot_to_bank = {slot: i for i, slot in enumerate(ACTIVE_SLOTS)}
        super().__init__(*args, **kwargs)
        if attention_backend in ('buffered_sdpa','grouped_sdpa8','gathered_sdpa8','gathered_sdpa32','gathered_sdpa32_vectorized','gathered_sdpa32_fused','gathered_sdpa64_fused'):
            from ks3_research.attention import (BufferedSelfAttention, GroupedSelfAttention,
                                                GatheredSelfAttention,Gathered32SelfAttention,
                                                VectorizedGatheredSelfAttention,FusedAccumulationSelfAttention,Fused64SelfAttention)
            layer_type = dict(buffered_sdpa=BufferedSelfAttention,grouped_sdpa8=GroupedSelfAttention,
                              gathered_sdpa8=GatheredSelfAttention,
                              gathered_sdpa32=Gathered32SelfAttention,
                              gathered_sdpa32_vectorized=VectorizedGatheredSelfAttention,
                              gathered_sdpa32_fused=FusedAccumulationSelfAttention,
                              gathered_sdpa64_fused=Fused64SelfAttention)[attention_backend]
            for slot, layer in list(self.attn.items()):
                self.attn[slot] = layer_type(layer.num_heads,layer.head_dim,layer.qk_dim,
                    layer.v_dim,layer.val_max_seq_len,paired=layer.paired)
        elif attention_backend != 'original_sdpa':
            raise ValueError(f'Unknown attention backend: {attention_backend}')
        self.attention_backend = attention_backend
        if loss_backend not in ('original','shared_gather','fused_softcap') or loss_slab_tokens<1:
            raise ValueError('Invalid training loss backend/slab size')
        self.loss_backend,self.loss_slab_tokens=loss_backend,loss_slab_tokens
        if ks3_backward not in ('vendor','fp32_blas','fp32_blas_all'):
            raise ValueError('Invalid KS3 backward backend')
        self.ks3_backward=ks3_backward
        if ks3_forward not in ('original','blas_output','blas_both','blas_fused_unpack') or (ks3_forward!='original' and ks3_backward!='fp32_blas_all'):
            raise ValueError('BLAS forward projections require the all-products BLAS backward')
        self.ks3_forward=ks3_forward
        self.register_buffer('ks3_tanh_lut',
            (torch.tanh(torch.arange(-128,128,device=self.device).float()*(3.2/127.))*127)
                .round().clamp(-127,127).to(torch.int8), persistent=False)

    def init_mlp(self, model_dim):
        # Consume the original initialization's draws so unrelated weights retain seed semantics.
        super().init_mlp(model_dim)
        features = self.ks3_charts * self.ks3_rank
        old = self.mlp_bank.detach()
        if features <= old.shape[2]:
            bank = old[list(ACTIVE_SLOTS), :, :features].clone()
        else:
            with torch.random.fork_rng(devices=[]):
                bank = torch.empty(len(ACTIVE_SLOTS), 2, features, model_dim)
                bank[:, 0].uniform_(-math.sqrt(3/model_dim)*.5, math.sqrt(3/model_dim)*.5)
                bank[:, 1].zero_()
        self.mlp_hdim = features
        self.mlp_bank = nn.Parameter(bank)
        self.mlp_bank.reshape = (2*len(ACTIVE_SLOTS), features, model_dim)
        # All live charts participate: six coefficients per coordinate and two warps per chart.
        self.ks3_coefficients = nn.Parameter(torch.zeros(len(ACTIVE_SLOTS), self.ks3_charts,
                                                        self.ks3_rank, 6))
        self.ks3_scale = nn.Parameter(torch.full((len(ACTIVE_SLOTS), self.ks3_charts), .75))
        self.ks3_shift = nn.Parameter(torch.zeros(len(ACTIVE_SLOTS), self.ks3_charts))

    def _mlp_weights(self):
        weights = self.mlp_bank.flatten(0, 1).unbind(0)
        fcs, projs = [None]*NUM_MLP_SLOTS, [None]*NUM_MLP_SLOTS
        for slot, bank in self.slot_to_bank.items():
            fcs[slot], projs[slot] = weights[2*bank:2*bank+2]
        return fcs, projs

    def _mlp(self, x_normed, c_fc, c_proj, *, slot=None):
        bank = self.slot_to_bank[slot]
        implementation=ks3_int8
        if self.ks3_backward=='fp32_blas':
            from ks3_research.backward import ks3_blas
            implementation=ks3_blas
        elif self.ks3_backward=='fp32_blas_all':
            from ks3_research.backward import ks3_blas_all
            implementation=ks3_blas_all
        if self.ks3_forward=='blas_output':
            from ks3_research.backward import ks3_blas_output
            implementation=ks3_blas_output
        elif self.ks3_forward=='blas_both':
            from ks3_research.backward import ks3_blas_both
            implementation=ks3_blas_both
        elif self.ks3_forward=='blas_fused_unpack':
            from ks3_research.backward import ks3_blas_fused_unpack
            implementation=ks3_blas_fused_unpack
        return implementation(x_normed, c_fc, c_proj.T, self.ks3_coefficients[bank],
                        self.ks3_scale[bank], self.ks3_shift[bank],
                        subspace_rank=self.ks3_rank, nonlinear=self.ks3_nonlinear,
                        tanh_lut=self.ks3_tanh_lut)

    def cast_matrix_weights_bf16(self):
        master = self.mlp_bank.data
        super().cast_matrix_weights_bf16()
        self.mlp_bank.data = master  # the W8A8 kernel owns quantization; masters stay FP32

    def _loss(self,x,input_seq,target_seq,mtp_weights,prefix_weight,sampled_loss):
        if not self.training or self.loss_backend=='original':
            return super()._loss(x,input_seq,target_seq,mtp_weights,prefix_weight,sampled_loss)
        from ks3_research.loss import shared_gather_loss
        return shared_gather_loss(x.view(-1,x.size(-1)),self.lm_head.weight,target_seq,self.prefix_table,
                                  mtp_weights,prefix_weight,sampled_loss,self.loss_slab_tokens,
                                  fused_softcap=self.loss_backend=='fused_softcap')

    def ffn_parameters(self):
        return sum(p.numel() for p in (self.mlp_bank, self.ks3_coefficients,
                                      self.ks3_scale, self.ks3_shift))
