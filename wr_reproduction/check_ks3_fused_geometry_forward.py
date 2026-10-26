"""Exact original forward/gradient parity at real routed expert shapes."""
from unittest.mock import patch
import torch
from wr_reproduction import ks3_moe as core, ks3_fused_geometry_forward as fused
from wr_reproduction.check_ks3_pairwise import error


def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(3169);torch.set_num_threads(1);core.KS3_RANK=1
    e,n,d=8,1024,768;checks=[];timings=[]
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.025).bfloat16()) for _ in range(e)])
    original_backward=core.backward
    for tuned,tokens in ((True,1),(True,257),(True,16384),(True,49152),(False,257)):
        core.BF16_TUNED_KERNELS=tuned
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        assignments=torch.zeros(tokens,device='cuda',dtype=torch.int64) if tokens==257 else torch.randint(e,(tokens,),device='cuda')
        _,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        for zero in (True,False):
            g=torch.randn(e,n,8,device='cuda')*.015;g[...,6]=.75;g[...,7]=0.
            if zero:g[...,:6].zero_()
            original=fused.ORIGINAL_FORWARD(x,src,tiles,up,down,g)
            actual=fused.forward(x,src,tiles,up,down,g)
            names=('output','input_pack','coordinates','features')
            forward_checks={k:dict(bitwise=torch.equal(a,b),relative_rms=error(a,b))
                for k,a,b in zip(names,actual,original,strict=True)}
            assert all(v['bitwise'] for v in forward_checks.values()),(tuned,tokens,zero,forward_checks)
            old=core.backward(grad,*original,gate,src,offs,tiles,up,down,g,bank,0);old_bank=bank.clone()
            new=core.backward(grad,*actual,gate,src,offs,tiles,up,down,g,bank,0)
            grad_checks={k:dict(bitwise=torch.equal(a,b),relative_rms=error(a,b))
                for k,a,b in zip(('input','gate','geometry','weight'),(*new,bank),(*old,old_bank),strict=True)}
            assert all(v['bitwise'] for v in grad_checks.values()),(tuned,tokens,zero,grad_checks)
            xx=x.clone().requires_grad_();router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
            post=torch.tensor(.8,device='cuda',requires_grad=True);geometry=g.clone().requires_grad_()
            def routed(forward):
                with patch.object(core,'forward',forward):
                    y,_,_=core.RoutedKS3.apply(xx,router,post,up,down,up,down,geometry,bank,0,True)
                    grads=torch.autograd.grad(y,(xx,router,post,geometry),grad)
                    return y.detach(),[v.detach().clone() for v in grads],bank.clone()
            a=routed(fused.ORIGINAL_FORWARD);b=routed(fused.forward)
            assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2])
            assert all(torch.equal(v,w) for v,w in zip(a[1],b[1],strict=True))
            row=dict(tuned=tuned,tokens=tokens,zero_coefficients=zero,active_experts=int((counts>0).sum()),
                padded_rows=src.numel(),forward=forward_checks,gradients=grad_checks,
                routed_output_input_router_post_geometry_weight_bitwise_equal=True)
            checks.append(row);print(row,flush=True)
        if tuned and tokens>=16384:
            # One bounded forward timing pair at each real scheduled shape;
            # this is diagnostic only. Actual-model speed remains a separate gate.
            import triton
            old_ms=triton.testing.do_bench(lambda:fused.ORIGINAL_FORWARD(x,src,tiles,up,down,g),warmup=100,rep=300)
            new_ms=triton.testing.do_bench(lambda:fused.forward(x,src,tiles,up,down,g),warmup=100,rep=300)
            timings.append(dict(tokens=tokens,original_forward_ms=old_ms,fused_forward_ms=new_ms,ratio=new_ms/old_ms))
    fused.install();fused.install()
    assert core.forward is fused.forward and core.backward is original_backward
    assert fused.EXECUTED_MODES=={'training_fused_up_geometry','original_evaluation'}
    return dict(passed=True,numerical_seed=3169,checks=checks,block_timings=timings,
        new_parameters=0,new_persistent_buffers=0,original_selected_GEMM_tiles=True,
        original_GS32_groups_of32=True,original_FP16_nonlinearity=True,
        original_forward_and_all_gradients_bitwise=True,original_backward_and_evaluation=True,
        install_idempotent=True,actual_model_training_tested=False,full_convergence_verified=False,
        tolerance_scope='All stored coordinates, packed features, outputs and input/router/post/geometry/weight gradients must match original bitwise. No numerical tolerance relaxation.',
        oracle='Original separate up GEMM and geometry kernels; actual1/257/16384/49152-token shapes with zero/nonzero coefficients and empty experts. No new mathematical function.')
