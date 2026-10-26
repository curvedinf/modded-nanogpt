"""Prepared geometry cache freshness and exact original output/gradient checks."""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from wr_reproduction import ks3_moe as core, ks3_prepared_geometry as prepared
from wr_reproduction.check_ks3_pairwise import error


def original_kernels():
    stack=ExitStack()
    stack.enter_context(patch.object(core,'_geometry',prepared.ORIGINAL_GEOMETRY))
    stack.enter_context(patch.object(core,'_vjp',prepared.ORIGINAL_VJP))
    return stack


def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(2369);torch.set_num_threads(1);core.KS3_RANK=1;core.BF16_TUNED_KERNELS=True
    model=torch.nn.Module()
    model.register_buffer('moe_bank',(torch.randn(80,1024,768,device='cuda')*.025).bfloat16())
    model.register_buffer('ks3_up',torch.empty(5,8,1024,768,device='cuda',dtype=torch.bfloat16))
    model.register_buffer('ks3_down',torch.empty(5,8,768,1024,device='cuda',dtype=torch.bfloat16))
    g=torch.randn(5,8,1024,8,device='cuda')*.015;g[...,6]=.75;g[...,7]=0.
    model.register_parameter('ks3_geometry',torch.nn.Parameter(g))
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state());prepared.attach(model)
    assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
    pointers={n:b.data_ptr() for n,b in model.named_buffers()};refresh_checks=[]
    # Original replicated Adam consolidates parameter storage after construction.
    original_geometry_pointer=model.ks3_geometry.data_ptr()
    flat=model.ks3_geometry.detach().clone().reshape(-1)
    model.ks3_geometry.data=flat.view_as(model.ks3_geometry)
    assert model.ks3_geometry.data_ptr()!=original_geometry_pointer
    prepared.bind(model)
    assert all(prepared.key(model.ks3_geometry[site]) in prepared.BINDINGS for site in range(5))
    for index in range(2):
        with torch.no_grad():
            if index:model.ks3_geometry.add_(torch.randn_like(model.ks3_geometry)*.01)
        prepared.ORIGINAL_REFRESH(model);up=model.ks3_up.clone();down=model.ks3_down.clone()
        prepared.refresh(model);cache=model.ks3_prepared_geometry.clone()
        assert torch.equal(up,model.ks3_up) and torch.equal(down,model.ks3_down)
        a=model.ks3_geometry.detach()[...,:6]
        scale=(a.abs().amax(-1)/127.).clamp(2.**-24,65504.).half().float()
        q=(a/scale[...,None]).round().clamp(-127,127)
        assert torch.equal(scale,cache[...,6]) and torch.equal(q,cache[...,:6])
        prepared.prepare(model)
        assert torch.equal(cache,model.ks3_prepared_geometry)
        assert pointers=={n:b.data_ptr() for n,b in model.named_buffers()}
        refresh_checks.append(dict(index=index,original_weights_bitwise=True,
            independent_torch_quantized_coefficients_bitwise=True,
            repeated_cache_bitwise=True,all_buffer_pointers_preserved=True))
    prepared.install();prepared.install();checks=[];e,n,d=8,1024,768
    for site,(tuned,tokens) in enumerate(((True,1),(True,257),(True,16384),(True,49152),(False,257))):
        up=model.ks3_up[site];down=model.ks3_down[site]
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        assignments=torch.zeros(tokens,device='cuda',dtype=torch.int64) if tokens==257 else torch.randint(e,(tokens,),device='cuda')
        _,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        for zero in (True,False):
            core.BF16_TUNED_KERNELS=True
            with torch.no_grad():
                g=model.ks3_geometry[site]
                g.copy_(torch.randn_like(g)*.015);g[...,6]=.75;g[...,7]=0.
                if zero:g[...,:6].zero_()
            prepared.prepare(model);cache=model.ks3_prepared_geometry.clone()
            # Fresh leaf after initialization; keep the same cache-bound storage.
            g=model.ks3_geometry[site].detach().requires_grad_()
            core.BF16_TUNED_KERNELS=tuned
            with original_kernels():
                old=core.forward(x,src,tiles,up,down,g)
                oldb=core.backward(grad,*old,gate,src,offs,tiles,up,down,g,bank,0);oldbank=bank.clone()
            new=core.forward(x,src,tiles,up,down,g)
            newb=core.backward(grad,*new,gate,src,offs,tiles,up,down,g,bank,0)
            names=('output','input_pack','coordinates','features','input_gradient','gate_gradient','geometry_gradient','weight_gradient')
            differences={k:dict(bitwise=torch.equal(a,b),relative_rms=error(a,b))
                for k,a,b in zip(names,(*new,*newb,bank),(*old,*oldb,oldbank),strict=True)}
            assert all(v['bitwise'] for v in differences.values()),(site,tokens,zero,differences)
            xx=x.clone().requires_grad_();router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
            post=torch.tensor(.8,device='cuda',requires_grad=True)
            def routed(original):
                context=original_kernels() if original else ExitStack()
                with context:
                    y,_,_=core.RoutedKS3.apply(xx,router,post,up,down,up,down,g,bank,0,True)
                    grads=torch.autograd.grad(y,(xx,router,post,g),grad)
                    return y.detach(),[v.detach().clone() for v in grads],bank.clone()
            a=routed(True);b=routed(False)
            assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2])
            assert all(torch.equal(v,w) for v,w in zip(a[1],b[1]))
            assert torch.equal(cache,model.ks3_prepared_geometry)
            assert pointers=={n:b.data_ptr() for n,b in model.named_buffers()}
            row=dict(tuned=tuned,tokens=tokens,site=site,zero_coefficients=zero,
                active_experts=int((counts>0).sum()),padded_rows=src.numel(),comparisons=differences,
                routed_output_input_router_post_geometry_weight_bitwise_equal=True,cache_readonly_during_consumers=True)
            checks.append(row);print(row,flush=True)
    return dict(passed=True,numerical_seed=2369,cache_refresh_checks=refresh_checks,checks=checks,
        binding_after_optimizer_style_storage_consolidation=True,
        all_original_outputs_and_gradients_bitwise_equal=True,original_GS32_and_tiles_retained=True,
        new_parameters=0,extra_buffer_bytes=prepared.CACHE_BYTES,allocation_consumes_no_RNG=True,
        original_untuned_evaluation_bitwise=True,install_idempotent=True,
        full_graph_training_tested=False,full_convergence_verified=False)
