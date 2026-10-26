"""Cached transpose refresh and exact original KS3 output/gradient checks."""
from unittest.mock import patch
import torch
from wr_reproduction import ks3_moe as core, ks3_bf16_dual_layout as dual
from wr_reproduction.check_ks3_pairwise import error


def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(2269);torch.set_num_threads(1);core.KS3_RANK=1;core.BF16_TUNED_KERNELS=True
    model=torch.nn.Module()
    model.register_buffer('moe_bank',(torch.randn(80,1024,768,device='cuda')*.025).bfloat16())
    model.register_buffer('ks3_up',torch.empty(5,8,1024,768,device='cuda',dtype=torch.bfloat16))
    model.register_buffer('ks3_down',torch.empty(5,8,768,1024,device='cuda',dtype=torch.bfloat16))
    before_rng=(torch.get_rng_state(),torch.cuda.get_rng_state());dual.attach(model)
    assert torch.equal(torch.get_rng_state(),before_rng[0]) and torch.equal(torch.cuda.get_rng_state(),before_rng[1])
    initial_ptrs={name:b.data_ptr() for name,b in model.named_buffers()}
    refresh_checks=[]
    for refresh_index in range(2):
        if refresh_index:model.moe_bank.add_(torch.randn_like(model.moe_bank)*.003)
        dual.ORIGINAL_REFRESH(model);up=model.ks3_up.clone();down=model.ks3_down.clone()
        dual.refresh(model)
        assert torch.equal(up,model.ks3_up) and torch.equal(down,model.ks3_down)
        assert torch.equal(up.transpose(-2,-1),model.ks3_bf16_up_t)
        assert torch.equal(down.transpose(-2,-1),model.ks3_bf16_down_t)
        assert initial_ptrs=={name:b.data_ptr() for name,b in model.named_buffers()}
        refresh_checks.append(dict(refresh_index=refresh_index,both_original_values_bitwise=True,
            both_transposes_bitwise=True,all_buffer_pointers_preserved=True))
    dual.install();dual.install();checks=[];e,n,d=8,1024,768
    for site,(tuned,tokens) in enumerate(((True,1),(True,257),(True,16384),(True,49152),(False,257))):
        core.BF16_TUNED_KERNELS=tuned
        up=model.ks3_up[site];down=model.ks3_down[site]
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        assignments=torch.zeros(tokens,device='cuda',dtype=torch.int64) if tokens==257 else torch.randint(e,(tokens,),device='cuda')
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        for zero in (True,False):
            g=torch.randn(e,n,8,device='cuda')*.015;g[...,6]=.75;g[...,7]=0.
            if zero:g[...,:6].zero_()
            with patch.object(core,'mm',dual.ORIGINAL_MM):
                old=core.forward(x,src,tiles,up,down,g)
                oldb=core.backward(grad,*old,gate,src,offs,tiles,up,down,g,bank,0);oldbank=bank.clone()
            new=core.forward(x,src,tiles,up,down,g)
            newb=core.backward(grad,*new,gate,src,offs,tiles,up,down,g,bank,0)
            names=('output','input_pack','coordinates','features','input_gradient','gate_gradient','geometry_gradient','weight_gradient')
            differences={k:dict(bitwise=torch.equal(a,b),relative_rms=error(a,b))
                for k,a,b in zip(names,(*new,*newb,bank),(*old,*oldb,oldbank),strict=True)}
            assert all(v['bitwise'] for v in differences.values()),(site,tokens,zero,differences)
            xx=x.clone().requires_grad_();router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
            post=torch.tensor(.8,device='cuda',requires_grad=True);geometry=g.clone().requires_grad_()
            def routed(mm):
                with patch.object(core,'mm',mm):
                    y,_,_=core.RoutedKS3.apply(xx,router,post,up,down,up,down,geometry,bank,0,True)
                    grads=torch.autograd.grad(y,(xx,router,post,geometry),grad)
                    return y.detach(),[v.detach().clone() for v in grads],bank.clone()
            a=routed(dual.ORIGINAL_MM);b=routed(dual.mm)
            assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2])
            assert all(torch.equal(v,w) for v,w in zip(a[1],b[1]))
            row=dict(tuned=tuned,tokens=tokens,site=site,zero_coefficients=zero,
                active_experts=int((counts>0).sum()),padded_rows=src.numel(),comparisons=differences,
                routed_output_input_router_post_geometry_weight_bitwise_equal=True)
            checks.append(row);print(row,flush=True)
    return dict(passed=True,numerical_seed=2269,cache_refresh_checks=refresh_checks,
        checks=checks,all_original_outputs_and_gradients_bitwise_equal=True,
        original_GS32_and_selected_tiles_retained=True,new_parameters=0,
        extra_buffer_bytes=2*5*8*1024*768*2,allocation_consumes_no_RNG=True,
        original_untuned_evaluation_bitwise=True,install_idempotent=True,
        full_graph_training_tested=False,full_convergence_verified=False)
