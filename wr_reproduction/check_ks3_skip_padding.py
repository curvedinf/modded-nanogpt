"""Prove the routing predicate and require bitwise original model math."""
import random
from unittest.mock import patch
import torch
from wr_reproduction import ks3_moe as core,ks3_skip_padding as skip
from wr_reproduction.bf16_tiles import select_projection_tiles


def cpu_check():
    rng=random.Random(3469)
    cases=[[1,0,0,0,0,0,0,0],[257,0,0,0,0,0,0,0],
           [0,1,127,128,129,1023,2048,0]]
    cases += [[rng.randrange(4097) for _ in range(8)] for _ in range(64)]
    checked=0
    for counts in cases:
        real=[]
        for count in counts:real += [True]*count+[False]*((-count)%128)
        real += [False]*1024
        for bm in (32,64,128):
            for lo in range(0,len(real),bm):
                if not real[lo]:assert not any(real[lo:lo+bm])
                checked+=1
    for m in (1024,1280,17280,50048):
        for n,k in ((1024,768),(768,1024)):
            for row_major in (True,False):
                bm,*_=select_projection_tiles(m,n,k,row_major,'mi300-v1')
                assert bm<=128 and 128%bm==0
    return dict(passed=True,layout_cases=len(cases),tile_predicates_checked=checked,
        first_sentinel_proves_whole_tile_invalid=True,all_original_projection_tiles_divide128=True,
        contract='Original dispatch: each expert begins at128 alignment, real rows precede sentinel padding, then unused capacity. No assumption about source-index order within the real rows.')


def check():
    cpu=cpu_check()
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(3469);torch.set_num_threads(1);core.KS3_RANK=1
    e,n,d=8,1024,768;checks=[]
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.025).bfloat16()) for _ in range(e)])
    for tokens in (1,257,16384,49152):
        core.BF16_TUNED_KERNELS=True
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        assignments=torch.zeros(tokens,device='cuda',dtype=torch.int64) if tokens==257 else torch.randint(e,(tokens,),device='cuda')
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        real=src<tokens;layout=real.cpu();assert bool((src>=0).all())
        block_counts={}
        for bm in (32,64,128):
            blocks=layout.view(-1,bm)
            assert not bool((~blocks[:,0]&blocks.any(1)).any())
            block_counts[str(bm)]=dict(total=blocks.shape[0],entirely_padding=int((~blocks[:,0]).sum()))
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        for zero in (True,False):
            g=torch.randn(e,n,8,device='cuda')*.015;g[...,4:6]*=2;g[...,6]=.75;g[...,7]=0.
            if zero:g[...,:6].zero_()
            baseline=skip.ORIGINAL_FORWARD(x,src,tiles,up,down,g)
            expected=skip.ORIGINAL_BACKWARD(grad,*baseline,gate,src,offs,tiles,up,down,g,bank,0)
            expected_bank=bank.clone()
            candidate=skip.forward(x,src,tiles,up,down,g)
            # Down-projection values for wholly unused rows intentionally become
            # zero. They are never gathered into model outputs or valid grads.
            assert torch.equal(candidate[0][real],baseline[0][real]),('valid output',tokens,zero)
            assert all(torch.equal(a,b) for a,b in zip(candidate[1:],baseline[1:])),('saved producers',tokens,zero)
            got=skip.backward(grad,*candidate,gate,src,offs,tiles,up,down,g,bank,0)
            exact=dict(input_gradient=torch.equal(got[0],expected[0]),gate_gradient=torch.equal(got[1],expected[1]),
                geometry_gradient=torch.equal(got[2],expected[2]),both_projection_weight_gradients=torch.equal(bank,expected_bank))
            assert all(exact.values()),('original backward',tokens,zero,exact)
            xx=x.clone().requires_grad_();router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
            post=torch.tensor(.8,device='cuda',requires_grad=True);geometry=g.clone().requires_grad_()
            def routed(new):
                with patch.object(core,'forward',skip.forward if new else skip.ORIGINAL_FORWARD), \
                     patch.object(core,'backward',skip.backward if new else skip.ORIGINAL_BACKWARD):
                    out,_,_=core.RoutedKS3.apply(xx,router,post,up,down,up,down,geometry,bank,0,True)
                    gradients=torch.autograd.grad(out,(xx,router,post,geometry),grad)
                    return out.detach(),[v.detach().clone() for v in gradients],bank.clone()
            a=routed(False);b=routed(True)
            assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2]),('routed output/weight',tokens,zero)
            assert all(torch.equal(u,v) for u,v in zip(a[1],b[1])),('routed gradients',tokens,zero)
            core.BF16_TUNED_KERNELS=False
            a=skip.ORIGINAL_FORWARD(x,src,tiles,up,down,g);b=skip.forward(x,src,tiles,up,down,g)
            assert all(torch.equal(u,v) for u,v in zip(a,b))
            old=skip.ORIGINAL_BACKWARD(grad,*a,gate,src,offs,tiles,up,down,g,bank,0);old_bank=bank.clone()
            new=skip.backward(grad,*b,gate,src,offs,tiles,up,down,g,bank,0)
            assert all(torch.equal(u,v) for u,v in zip(old,new)) and torch.equal(bank,old_bank)
            core.BF16_TUNED_KERNELS=True
            row=dict(tokens=tokens,zero_coefficients=zero,padded_rows=src.numel(),active_experts=int((counts>0).sum()),
                original_dispatch_predicate_checked=True,row_blocks=block_counts,
                valid_output_and_saved_producers_bitwise=True,original_backward_bitwise=exact,
                routed_output_input_router_post_geometry_and_weight_gradients_bitwise=True,
                original_evaluation_and_untuned_backward_bitwise=True)
            checks.append(row);print(row,flush=True)
    skip.install();skip.install()
    assert core._geometry is skip.ORIGINAL_GEOMETRY and core._vjp is skip.ORIGINAL_VJP
    assert skip.EXECUTED_MODES=={'training_skip_padded_forward','training_skip_padded_backward','original_evaluation','original_untuned_backward'}
    return dict(passed=True,cpu_checks=cpu,numerical_seed=3469,checks=checks,
        matrix_shapes=[list(x) for x in sorted(skip.MATRIX_SHAPES)],
        shape_fields=['tokens','capacity_rows','N','K','BM','BN','BK','warps','stages','mfma'],
        new_parameters=0,original_function_and_gradients_bitwise=True,original_geometry_and_VJP=True,
        original_evaluation=True,initialization_and_optimizer_unchanged=True,
        oracle='Bitwise original routed implementation including valid outputs, input/router/post/geometry and both projection-weight gradients. Original dispatch predicate checked for every row tile in every actual shape.',
        scope='Only entire sentinel projection blocks are skipped. Unused down-projection rows may differ; no model output or gradient may differ.',
        actual_model_training_tested=False,full_convergence_verified=False)
