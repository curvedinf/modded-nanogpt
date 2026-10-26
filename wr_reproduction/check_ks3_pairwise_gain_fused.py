"""Exact fused/unfused gain parity plus the independent gain/gradient oracle."""
from unittest.mock import patch
import torch
from wr_reproduction import ks3_moe as core, ks3_pairwise_gain as gain
from wr_reproduction import ks3_pairwise_gain_fused as fused
from wr_reproduction.check_ks3_pairwise import error


def check(settings):
    fused.install()
    from wr_reproduction.check_ks3_pairwise_gain import check as independent
    independent_result=independent(settings)
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(2169)
    e,n,d=8,1024,768;bound=settings['bound'];checks=[]
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.02).bfloat16()) for _ in range(e)])
    for tuned,tokens in ((True,1),(True,257),(True,4096),(True,16384),(True,49152),(False,257)):
        core.BF16_TUNED_KERNELS=tuned
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        assignments=torch.randint(e,(tokens,),device='cuda')
        if tokens==257:assignments.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        g=torch.randn(e,n,8,device='cuda')*.01;g[...,6]=.75
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        for nonzero in (False,True):
            raw=torch.randn(e,n,device='cuda')*.35 if nonzero else torch.zeros(e,n,device='cuda')
            with patch.object(gain.routed,'features',fused.ORIGINAL_FEATURES),patch.object(gain.routed,'feature_vjp',fused.ORIGINAL_VJP):
                old=gain.forward(x,src,tiles,up,down,g,raw,bound)
                oldb=gain.backward(grad,*old,gate,src,offs,tiles,up,down,g,raw,bound,bank,0)
                oldbank=bank.clone()
            new=gain.forward(x,src,tiles,up,down,g,raw,bound)
            newb=gain.backward(grad,*new,gate,src,offs,tiles,up,down,g,raw,bound,bank,0)
            names=('output','input_pack','coordinates','features','input_gradient','gate_gradient',
                   'geometry_gradient','gain_gradient','weight_gradient')
            comparisons={key:dict(bitwise_equal=torch.equal(a,b),relative_rms=error(a,b))
                         for key,a,b in zip(names,(*new,*newb,bank),(*old,*oldb,oldbank),strict=True)}
            assert all(x['bitwise_equal'] for x in comparisons.values()),(tuned,tokens,nonzero,comparisons)
            row=dict(tuned=tuned,tokens=tokens,nonzero_coefficients=nonzero,
                active_experts=int((counts>0).sum()),comparisons=comparisons)
            checks.append(row);print(row,flush=True)
    return dict(passed=True,independent_gain_and_original_gradient_oracle=independent_result,
        fused_unfused_checks=checks,all_fused_unfused_outputs_and_gradients_bitwise_equal=True,
        original_BF16_rounding_and_VJP_retained=True,numerical_seed=2169,
        full_graph_training_tested=False,full_convergence_verified=False)
