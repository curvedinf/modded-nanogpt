"""PR393 four-map ANVIL on ROCm; other optimizer equations remain unchanged."""
import torch
from mi300.optim.anvil import AnvilAndAdam, RAIL_SLOW_BETA, FROBENIUS_EPS
from mi300.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA
from ks3_research.optim import ContinuingManager
ANVIL_MAPS=[(8.057967458384,-22.052383339291,15.540875280932),(3.929245019706,-2.754491513849,.497237282013),(3.477283617853,-2.503541192905,.479259394356),(2.507712624527,-1.782373685650,.401358534767)]
@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def anvil_cascade(grad_chunk: torch.Tensor, velocity: torch.Tensor, momentum_t: torch.Tensor,
                  split_baddbmm: bool, fast_beta_t: torch.Tensor, fast_weight_t: torch.Tensor, margin: float = 1.05):
    """Twin-rail Nesterov momentum, then the ANVIL cascade that drives every singular value to ~1.

    velocity is one fp32 [2, *chunk] tensor: rail 0 fast, rail 1 slow. Their blend gets a Nesterov
    lookahead (momentum_t), is cast to bf16, normalized by its Frobenius norm, and whitened by
    ANVIL_MAPS. momentum_t, fast_beta_t and fast_weight_t are 0-D device tensors (AnvilBank), so
    per-step changes never recompile and a captured graph reads their current values.
    """
    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))
    velocity[1].lerp_(grad_chunk, 1 - RAIL_SLOW_BETA)
    w = fast_weight_t.to(grad_chunk.dtype)
    blend = w * velocity[0] + (1 - w) * velocity[1]
    g = grad_chunk.lerp_(blend, momentum)

    X = g.bfloat16().contiguous()
    is_tall = g.size(-2) > g.size(-1)

    # The first Gram is taken on the unnormalized X: its trace is ||X||_F^2, which gives the
    # normalization for free; X and the Gram are then rescaled instead of recomputing.
    if is_tall:
        # Tall: use Triton kernels with X^T @ X (small) and right multiplication
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        XTX(X, out=A)  # A = X.T @ X
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * margin + FROBENIUS_EPS
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm

        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XTX(X, out=A)  # A = X.T @ X
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b*A + c*(A@A)

            # Referencing X twice causes pytorch to make a defensive copy,
            # resulting in a cudaMemcpyAsync in baddbmm.
            # For large matrices (i.e., the mlp weights), it's faster to split
            # the operation into two kernels to avoid this.
            if split_baddbmm:
                XB_matmul(X, B, out=C)  # C = X @ B
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)  # C = a * X + X @ B

            X, C = C, X  # Swap references to avoid unnecessary copies
    else:
        # Wide: use Triton kernels with X @ X^T (small) and left multiplication
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        XXT(X, out=A)  # A = X @ X.mT
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * margin + FROBENIUS_EPS
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XXT(X, out=A)  # A = X @ X.mT
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

            if split_baddbmm:
                BX_matmul(B, X, out=C)  # C = B @ X
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

            X, C = C, X  # Swap references to avoid unnecessary copies

    return X

# -----------------------------------------------------------------------------
# Combined ANVIL + Adam Optimizer


class Optimizer(AnvilAndAdam):
    def _build_param_cfg(self,param,label):
        super()._build_param_cfg(param,label)
        if label=='moe_bank':
            self.param_cfgs[param].per_matrix_lr_mul=[2. if i%16>=8 else 1. for i in range(param.reshape[0])]

class KS3Optimizer(Optimizer):
    def _init_banks(self):
        from mi300.optim.anvil import AnvilBank,SPLIT_BADDBMM_MIN_ROWS
        from mi300.perf.bank_scalars import BankScalarStaging
        items=[(p,c) for p,c in self.param_cfgs.items() if c.optim=='anvil']
        self.bank_scalars=BankScalarStaging([c.chunk_size for _,c in items],items[0][0].device)
        self.banks={}
        for i,(p,c) in enumerate(items):
            shape=(c.chunk_size,*c.reshape[1:])
            self.banks[c.label]=AnvilBank(label=c.label,grad=torch.empty(shape,dtype=p.dtype,device=p.device),
                **self.live_bank_state(c.label),**self.bank_scalars.scalars(i),eff_lr=self.bank_scalars.eff_lr(i),
                beta2=c.beta2,split_baddbmm=shape[-2]>SPLIT_BADDBMM_MIN_ROWS,red_dim=-1 if shape[-2]>=shape[-1] else -2)

@torch.no_grad()
def bank_update(bank):
    grad=bank.grad.clone() if bank.grad.dtype==torch.float32 else bank.grad
    update=anvil_cascade(grad,bank.velocity,bank.momentum,bank.split_baddbmm,
                         bank.fast_beta,bank.fast_weight,1.1 if bank.label=='moe_bank' else 1.05)
    update=AnvilAndAdam._rail_equalizer(update,bank.lane_energy,bank.beta2,bank.red_dim)
    if bank.p_slice.dtype==torch.float32:
        p=bank.p_slice;aligned=(bank.velocity[1]*p)>=0
        p.copy_(p-p*aligned*bank.eff_wd*bank.eff_lr-update.float()*bank.eff_lr)
    else:
        AnvilAndAdam._sign_aligned_decay_update(bank.p_slice.view(torch.uint16),bank.mantissa,
                                               update,bank.eff_wd,bank.eff_lr,bank.velocity[1])

def make_manager(model,schedule,lr_scale=1.,lr_stage_scales=None,geometry_lr_mul=.125,expert_lr_mul=1.,optimizer_backend='eager'):
    adam=dict(optim='adam',comms='replicated',adam_betas=[.9,.99])
    extra={'moe_bank':dict(optim='anvil',comms='sharded',adam_betas=None),
           'moe_router':dict(adam,lr_mul=1.,wd_mul=0.),'copy_qk_bank':dict(adam),
           'copy_k_sink':dict(adam,wd_mul=0.),'copy_qk_gain':dict(adam,wd_mul=0.)}
    ks3=hasattr(model,'ks3_coefficients')
    if ks3:
        for name in ['ks3_coefficients','ks3_scale','ks3_shift']:extra[name]=dict(adam,lr_mul=geometry_lr_mul,wd_mul=0.)
    manager=ContinuingManager(model,schedule,bank_update,extra_param_table=extra,optimizer_type=KS3Optimizer if ks3 else Optimizer)
    manager.lr_stage_scales=tuple(lr_stage_scales or [1.]*5)
    manager.adam_defaults['lr']*=lr_scale
    for cfg in manager.optimizer.param_cfgs.values():
        factor=lr_scale*(expert_lr_mul if cfg.label=='moe_bank' else 1.)
        cfg.initial_lr*=factor;cfg.lr*=factor
    if optimizer_backend=='bank_graphs':
        from ks3_research.optimizer_graphs import BankGraphs
        for bank in manager.optimizer.banks.values():
            bank.grad.zero_()
        manager.optimizer.stage_bank_scalars()
        graphs=BankGraphs(manager.optimizer,bank_update)
        manager.optimizer.bank_update=graphs.update
        manager.optimizer.bank_graphs=graphs
    elif optimizer_backend!='eager':
        raise ValueError('Unknown PR393 optimizer backend')
    return manager
