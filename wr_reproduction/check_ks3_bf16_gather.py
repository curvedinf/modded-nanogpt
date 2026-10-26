"""Actual-width producer, full-block gradients and bounded paired timing gate."""
import statistics
import torch
import triton as tr
from wr_reproduction import ks3_moe as core,ks3_bf16_gather as fused


def oracle(grad,y,src,gate):
    valid=(src<grad.shape[0])[:,None];idx=src.clamp(0,grad.shape[0]-1).long()
    gathered=grad[idx].float()
    gg=torch.where(valid,gathered*gate[idx,None].to(torch.bfloat16).float(),0).to(torch.bfloat16)
    ds=torch.where(valid,gathered*y.float(),0).sum(1)
    size=torch.where(valid,(gathered*y.float()).abs(),0).sum(1)
    return gg,ds,size


@torch.no_grad()
def check():
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(869);torch.set_num_threads(1)
    core.BF16_TUNED_KERNELS=True;core.KS3_RANK=1
    e,n,d=8,1024,768
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.025).bfloat16()) for _ in range(e)])
    g=torch.randn(e,n,8,device='cuda')*.015;g[...,6]=.75
    rows=[]
    for tokens in (1,257,16384,49152):
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(x)*.02;gate=torch.randn(tokens,device='cuda')*.4
        expert=torch.randint(e,(tokens,),device='cuda')
        if tokens==257:expert.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
        y,xp,c,h=core.forward(x,src,tiles,up,down,g)
        expected,ds,size=oracle(grad,y,src,gate);actual,got=fused.gather(grad,y,src,gate)
        assert torch.equal(actual,expected),'BF16 gated gradient changed'
        assert torch.isfinite(got).all()
        assert torch.equal(got,ds),'Original gate reduction must remain exact'
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        before=core.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,bank,0)
        before_bank=bank.clone()
        after=fused.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,bank,0)
        exact={key:torch.equal(a,b) for key,a,b in [('input_gradient',after[0],before[0]),
               ('geometry_gradient',after[2],before[2]),('weight_gradient',bank,before_bank)]}
        assert all(exact.values()),(tokens,exact)
        # Inspect full routed autograd, including router and post gradients.
        router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
        post=torch.tensor(.8,device='cuda',requires_grad=True)
        xx=x.clone().requires_grad_();gg=g.clone().requires_grad_()
        def routed(back):
            saved=core.backward;core.backward=back
            try:
                out,probs,cts=core.RoutedKS3.apply(xx,router,post,up,down,up,down,gg,bank,0,True)
                gradients=torch.autograd.grad(out,(xx,router,post,gg),grad,retain_graph=False)
                return out.detach(),[v.detach().clone() for v in gradients],bank.clone()
            finally:core.backward=saved
        with torch.enable_grad():
            old=routed(core.backward);new=routed(fused.backward)
        assert torch.equal(old[0],new[0]) and torch.equal(old[2],new[2])
        routed_errors={}
        for name,a,b in zip(('input','router','post','geometry'),new[1],old[1]):
            assert torch.isfinite(a).all()
            err=((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
            assert torch.equal(a,b),(tokens,name,err)
            routed_errors[name]=err
        timings={key:[] for key in ('original','fused')}
        if tokens>=16384:
            functions={'original':lambda:core.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,bank,0),
                       'fused':lambda:fused.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,bank,0)}
            for order in (('original','fused'),('fused','original'),('original','fused')):
                for name in order:timings[name].append(tr.testing.do_bench_cudagraph(functions[name],rep=40))
        row=dict(tokens=tokens,active_experts=int((counts>0).sum()),padded_rows=src.numel(),
                 gated_gradient_bitwise_equal=True,gate_derivative_bitwise_equal=True,
                 routed_gradients_bitwise_equal=True,original_core_gradients_bitwise_equal=exact,
                 gate_gradient_max_abs_error=(got-ds).abs().max().item(),
                 gate_gradient_max_weighted_error=((got-ds).abs()/size.clamp_min(1e-12)).max().item(),
                 routed_relative_rms=routed_errors,backward_samples_ms=timings,
                 backward_median_ms={k:statistics.median(v) for k,v in timings.items() if v})
        rows.append(row);print(dict(event='bf16_gather_check',**row),flush=True)
    return dict(passed=True,numerical_seed=869,checks=rows,products='unchanged incumbent GS32/BF16',
                scope='KS3 routed backward producer only; original forward, STE, matrix products, weight packing and evaluation unchanged',
                full_graph_training_tested=False,full_convergence_verified=False,
                timing_limitation='Paired backward timing is not actual-model speed or convergence evidence')
