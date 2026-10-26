"""Bitwise producer, block and routed gradient parity on the unchanged function."""
import torch
from wr_reproduction import ks3_moe as core,ks3_local_residual as local,ks3_local_residual_gather as fused
from wr_reproduction.check_ks3_bf16_gather import oracle


@torch.no_grad()
def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(1669);torch.set_num_threads(1)
    core.BF16_TUNED_KERNELS=True;core.KS3_RANK=1
    e,n,d=8,1024,768
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.025).bfloat16()) for _ in range(e)])
    g=torch.randn(e,n,8,device='cuda')*.015;g[...,6]=.75
    checks=[]
    for tokens in (1,257,16384,49152):
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(x)*.02;gate=torch.randn(tokens,device='cuda')*.4
        expert=torch.randint(e,(tokens,),device='cuda')
        if tokens==257:expert.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
        for zero in (True,False):
            raw=torch.zeros(e,n,3,device='cuda') if zero else torch.randn(e,n,3,device='cuda')*.2
            y,xp,c,h=local.forward(x,src,tiles,up,down,g,raw,1.)
            expected,ds,size=oracle(grad,y,src,gate);actual,got=fused.gather(grad,y,src,gate)
            assert torch.equal(actual,expected) and torch.equal(got,ds)
            bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
            args=(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,raw,1.,bank,0)
            before=fused.ORIGINAL_BACKWARD(*args);before_bank=bank.clone()
            after=fused.backward(*args)
            exact={name:torch.equal(a,b) for name,a,b in zip(('input','gate','geometry','local_coefficients','weights'),(*after,bank),(*before,before_bank))}
            assert all(exact.values()),(tokens,zero,exact)
            router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
            post=torch.tensor(.8,device='cuda',requires_grad=True)
            xx=x.clone().requires_grad_();gg=g.clone().requires_grad_();pp=raw.clone().requires_grad_()
            def routed(back):
                saved=local.backward;local.backward=back
                try:
                    out,_,_=local.RoutedLocalized.apply(xx,router,post,up,down,up,down,gg,pp,bank,0,1.,True)
                    gradients=torch.autograd.grad(out,(xx,router,post,gg,pp),grad)
                    return out.detach(),[v.detach().clone() for v in gradients],bank.clone()
                finally:local.backward=saved
            with torch.enable_grad():
                old=routed(fused.ORIGINAL_BACKWARD);new=routed(fused.backward)
            routed_exact={name:torch.equal(a,b) for name,a,b in zip(('input','router','post','geometry','local_coefficients'),new[1],old[1])}
            assert torch.equal(old[0],new[0]) and torch.equal(old[2],new[2]) and all(routed_exact.values())
            row=dict(tokens=tokens,zero_local_coefficients=zero,active_experts=int((counts>0).sum()),padded_rows=src.numel(),
                producer_gradient_and_gate_derivative_bitwise_equal=True,block_gradient_bitwise=exact,
                routed_output_and_projection_gradient_bitwise_equal=True,routed_gradient_bitwise=routed_exact)
            checks.append(row);print(row,flush=True)
    fused.install();installed=local.backward;fused.install();assert local.backward is installed
    core.BF16_TUNED_KERNELS=False
    # The actual eval path only calls the unchanged forward; retain exact
    # original behavior for any diagnostic untuned backward invocation too.
    a=fused.ORIGINAL_BACKWARD(*args);ab=bank.clone();b=local.backward(*args)
    assert all(torch.equal(v,w) for v,w in zip(a,b)) and torch.equal(ab,bank)
    return dict(passed=True,numerical_seed=1669,checks=checks,untuned_fallback_bitwise_equal=True,
        installation_idempotent=True,all_original_and_new_function_parameters_unchanged=True,
        original_BF16_gradient_boundary_unchanged=True,all_shared_kernels_unchanged=True,
        full_graph_training_tested=False,actual_model_speed_verified=False,full_convergence_verified=False)
