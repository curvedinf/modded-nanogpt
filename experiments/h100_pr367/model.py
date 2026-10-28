"""PR367 retrieval model with five eight-expert, top-one cubic KS3 sites."""
import math
import os
import torch
import triton as tr
from torch import nn

from experiments.h100_ks3 import fp8
from experiments.h100_ks3.fp8_kernels import publish
from experiments.h100_ks3.weight_layouts import build as build_layouts
from wr_reproduction import ks3_moe as core

LAYERS = (1, 3, 5, 6, 8)
INDEX = {layer: index for index, layer in enumerate(LAYERS)}
WIDTH = 1024
DIM = 512
MAPS = ((8.057967458384, -22.052383339291, 15.540875280932),
        (3.929245019706, -2.754491513849, 0.497237282013),
        (3.477283617853, -2.503541192905, 0.479259394356),
        (2.507712624527, -1.782373685650, 0.401358534767))


class RoutedWithWeightGrad(fp8.Routed):
    """Expose the checked bank gradients to PR367's standard graph collector."""
    @staticmethod
    def backward(ctx, grad, dprobs, dcounts):
        result = list(fp8.routed_backward(ctx, grad, dprobs, dcounts))
        result[3] = ctx.bank[16 * ctx.layer:16 * ctx.layer + 8]
        result[4] = ctx.bank[16 * ctx.layer + 8:16 * ctx.layer + 16]
        return tuple(result)


@torch.compile(dynamic=False, fullgraph=True)
def normalize(x, gram):
    divisor = gram.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None].sqrt() * 1.1 + 1e-6
    return (x.float() / divisor).bfloat16(), (gram.float() / divisor.square()).bfloat16()


def install_optimizer():
    from track_1_short import training
    from track_1_short.optim import anvil
    from track_1_short.perf.cuda_graphs import optimizer_graphs
    from track_1_short.perf.kernels import anvil_opt as fused
    original_build = anvil.AnvilAndAdam._build_param_cfg

    def build(self, parameter, label):
        original_build(self, parameter, label)
        if label == 'moe_bank':
            cfg = self.param_cfgs[parameter]
            start = self.rank * cfg.chunk_size
            cfg.per_matrix_lr_mul = [2. if (start + i) % 16 >= 8 else 1. for i in range(cfg.chunk_size)]

    anvil.AnvilAndAdam._build_param_cfg = build
    original_optimizer = training.AnvilAndAdam

    def optimizer(*args, **kwargs):
        table = kwargs['param_table']
        table.update(moe_bank=dict(optim='anvil', comms='sharded', adam_betas=None),
            moe_router=dict(optim='adam', comms='replicated', adam_betas=[.9, .99], lr_mul=1., wd_mul=0.),
            ks3_geometry=dict(optim='adam', comms='replicated', adam_betas=[.9, .99], lr_mul=.125, wd_mul=0.))
        for name in ('scatter_order', 'work_order'):
            kwargs[name].extend(('moe_bank', 'moe_router', 'ks3_geometry'))
        return original_optimizer(*args, **kwargs)

    training.AnvilAndAdam = optimizer
    original_update = anvil.anvil_bank_update

    def update(bank):
        if bank.label != 'moe_bank':
            return original_update(bank)
        x = torch.empty_like(bank.grad)
        fused.pre_fused(bank.grad, bank.velocity, x, bank.momentum, bank.fast_beta, bank.fast_weight)
        gram = torch.empty((*x.shape[:-2], x.shape[-1], x.shape[-1]), device=x.device, dtype=x.dtype)
        fused.gram_tall(x, gram)
        x, gram = normalize(x, gram)
        poly, out = torch.empty_like(gram), torch.empty_like(x)
        for k, (a, b, c) in enumerate(MAPS):
            if k:
                fused.gram_tall(x, gram)
            fused.poly_gram(gram, c, b, poly)
            fused.gemm_xb_add(x, poly, out, a)
            x, out = out, x
        x = anvil.AnvilAndAdam._rail_equalizer(x, bank.lane_energy, bank.beta2, bank.red_dim)
        anvil.AnvilAndAdam._sign_aligned_decay_update(
            bank.p_slice.view(torch.uint16), bank.mantissa, x, bank.eff_wd, bank.eff_lr, bank.velocity[1])

    anvil.anvil_bank_update = optimizer_graphs.anvil_bank_update = update


def install():
    from track_1_short.model.gpt import GPT, FP8_EXACT_SCALE_CALLS, FP8_GRAD_SCALE, FP8_POST_HEADROOM, FP8_DPRE_HEADROOM
    from track_1_short.perf.cuda_graphs import fp8_refresh_graphs
    from triton.backends.nvidia.compiler import get_ptxas
    assert tuple(map(int, get_ptxas(90).version.split('.'))) >= (13, 3)
    assert fp8.GS32 and fp8.CUBIC_ONLY and fp8.TMA_OUTPUTS and fp8.MM_IMPL == 'tma'
    install_optimizer()
    from experiments.h100_ks3.geometry_tail import install as install_geometry_tail
    install_geometry_tail()
    fp8_refresh_graphs.MLP_OUTPUT_PREFIXES += (
        '_ks3_f8_', '_ks3_weight_', '_ks3_post_', '_ks3_dpre_', '_ks3_packed', '_ks3_partial')

    class KS3GPT(GPT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert kwargs['model_dim'] == DIM
            bank = torch.empty(80, WIDTH, DIM)
            for k in range(5):
                bank[16*k:16*k+8].uniform_(-math.sqrt(3) * .5 * DIM**-.5, math.sqrt(3) * .5 * DIM**-.5)
                bank[16*k+8:16*k+16].zero_()
            self.moe_bank = nn.Parameter(bank)
            self.moe_bank.reshape = (80, WIDTH, DIM)
            self.moe_bank.label = 'moe_bank'
            self.moe_router = nn.Parameter(torch.randn(5, 8, DIM) * .01)
            self.moe_router.label = 'moe_router'
            geometry = torch.zeros(5, 8, WIDTH, 8, device=self.device)
            geometry[..., 6] = .75
            self.ks3_geometry = nn.Parameter(geometry)
            self.ks3_geometry.label = 'ks3_geometry'
            shapes = dict(_ks3_f8_up=(5,8,WIDTH,DIM), _ks3_f8_up_t=(5,8,DIM,WIDTH),
                          _ks3_f8_down=(5,8,DIM,WIDTH), _ks3_f8_down_t=(5,8,WIDTH,DIM),
                          _ks3_packed=(80,WIDTH*DIM))
            for name, shape in shapes.items():
                self.register_buffer(name, torch.empty(shape, device=self.device, dtype=torch.float8_e4m3fn), persistent=False)
            for name, shape, value in (
                ('_ks3_weight_scales',(5,2,8),1.), ('_ks3_partial',(80,WIDTH*DIM//4096),1.),
                ('_ks3_post_scales',(5,),1.), ('_ks3_post_amax',(5,1),0.),
                ('_ks3_dpre_scales',(5,),FP8_GRAD_SCALE), ('_ks3_dpre_amax',(5,1),0.)):
                self.register_buffer(name, torch.full(shape, value, device=self.device), persistent=False)
            for name, shape in (('_ks3_up',(5,8,WIDTH,DIM)), ('_ks3_down',(5,8,DIM,WIDTH)), ('_ks3_grad',(80,WIDTH,DIM))):
                self.register_buffer(name, torch.empty(shape, device=self.device, dtype=torch.bfloat16), persistent=False)
            self.register_buffer('_ks3_overwrite', torch.zeros((), device=self.device, dtype=torch.int32), persistent=False)
            print('[ks3-pr367] five top1 MoE sites;8 experts;1024 cubic rank1 charts;FP8 products;GS32/BF16 evaluation', flush=True)

        def cast_matrix_weights_bf16(self):
            super().cast_matrix_weights_bf16()
            self.moe_bank.data = self.moe_bank.data.bfloat16()

        def rearm_fp8_bootstrap(self):
            super().rearm_fp8_bootstrap()
            self._ks3_partial.zero_()
            self._ks3_post_amax.zero_()
            self._ks3_dpre_amax.zero_()
            self._ks3_post_scales.fill_(1.)
            self._ks3_dpre_scales.fill_(FP8_GRAD_SCALE)

        @torch.no_grad()
        def quantize_mlp_fp8(self, refresh_lm):
            exact = self.mlp_quantize_calls < FP8_EXACT_SCALE_CALLS
            super().quantize_mlp_fp8(refresh_lm)
            maximum = self.moe_bank.float().abs().amax((1,2)) if exact else self._ks3_partial.amax(1)
            self._ks3_weight_scales.view(-1).copy_(maximum.clamp_min(1e-12)/448.)
            publish[(WIDTH*DIM//4096,80)](self.moe_bank,self._ks3_packed,self._ks3_weight_scales,self._ks3_partial,0,WIDTH,D=DIM,num_warps=4)
            build_layouts(self._ks3_packed,self._ks3_f8_up,self._ks3_f8_up_t,self._ks3_f8_down,self._ks3_f8_down_t,WIDTH,self._ks3_weight_scales,tile=64)
            self._refresh_delayed_scale(self._ks3_post_amax,self._ks3_post_scales,FP8_POST_HEADROOM/448.,1e-6)
            self._refresh_delayed_scale(self._ks3_dpre_amax,self._ks3_dpre_scales,FP8_DPRE_HEADROOM/57344.,1e-12)

        @torch.no_grad()
        def prepare_ks3_eval(self):
            bank = self.moe_bank.view(5,2,8,WIDTH,DIM)
            for k in range(5):
                self._ks3_up[k].copy_(core.pack(bank[k,0].reshape(8*WIDTH,DIM)).view(8,WIDTH,DIM))
                for e in range(8):
                    self._ks3_down[k,e].copy_(core.pack(bank[k,1,e].T))

        def _moe(self, layer, normed, post):
            from track_1_short.perf.kernels.moe_mlp import moe_dispatch
            k = INDEX[layer]
            rows = normed.reshape(-1,DIM)
            router = self.moe_router[k].bfloat16()
            if not self.training:
                probs = torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
                gate,expert = probs.max(-1)
                dest,src,_,table,counts = moe_dispatch(expert,8)
                y,_,_,_ = core.forward(rows,src,table,self._ks3_up[k],self._ks3_down[k],self.ks3_geometry[k])
                out = y[dest.long()]*gate[:,None].to(y.dtype)
            else:
                if post is None:
                    post = rows.new_ones(())
                w1,w2 = self.moe_bank.view(5,2,8,WIDTH,DIM)[k].unbind(0)
                caches = (self._ks3_f8_up[k],self._ks3_f8_up_t[k],self._ks3_f8_down[k],self._ks3_f8_down_t[k])
                scales = (self._ks3_weight_scales[k,0],self._ks3_weight_scales[k,1],self._mlp_x_scale,self._mlp_grad_scale,
                          self._ks3_post_scales[k:k+1],self._ks3_post_amax[k],self._ks3_dpre_scales[k:k+1],self._ks3_dpre_amax[k])
                out,probs,counts = RoutedWithWeightGrad.apply(rows,router,post,w1,w2,self.ks3_geometry[k],caches,scales,self._ks3_grad,self._ks3_overwrite,k)
            aux = 8*(counts.float()/rows.shape[0]*probs.mean(0)).sum()
            return out.reshape(normed.shape),aux

    return KS3GPT
