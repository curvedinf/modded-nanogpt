"""Linear-coordinate audit plus actual-width original-kernel differential tests."""
import torch
from wr_reproduction import ks3_geometry_basis as basis,ks3_moe as core


def cpu_check():
    torch.manual_seed(1069)
    theta=torch.randn(37,8,dtype=torch.float64,requires_grad=True)
    g=basis.reference_map(theta)
    t=torch.linspace(-4,4,129,dtype=torch.float64)[:,None]
    a0,a1,a2,a3=theta[:,:4].unbind(-1);c0,c1,c2,c3=g[:,:4].unbind(-1)
    hermite=((2*c0-2*c1+c2+c3)*t+(-3*c0+3*c1-2*c2-c3))*t*t+c2*t+c0
    u=t/4;cheb=a0+a1*u+a2*(2*u*u-1)+a3*(4*u*u*u-3*u)
    error=(hermite-cheb).abs().max().item();assert error<2e-13,error
    weight=torch.randn_like(hermite)
    hgrad=torch.autograd.grad((hermite*weight).sum(),theta,retain_graph=True)[0]
    cgrad=torch.autograd.grad((cheb*weight).sum(),theta)[0]
    gradient_error=(hgrad-cgrad).abs().max().item();assert gradient_error<3e-12,gradient_error
    matrix=basis.reference_map(torch.eye(8,dtype=torch.float64))
    assert torch.linalg.matrix_rank(matrix)==8 and torch.equal(g[:,4:],theta[:,4:])
    assert torch.equal(basis.reference_map(torch.zeros_like(theta)),torch.zeros_like(theta))
    old=torch.stack((2*t**3-3*t**2+1,-2*t**3+3*t**2,t**3-2*t**2+t,t**3-t**2),-1)
    new=torch.stack((torch.ones_like(u),u,2*u*u-1,4*u**3-3*u),-1)
    return dict(passed=True,independent_cubic_identity_max_abs=error,
        independent_cubic_gradient_max_abs=gradient_error,invertible=True,
        zero_initialization_and_warp_coordinates_exact=True,
        theoretical_basis_max_abs_on_t_minus4_to4=dict(hermite=old.abs().amax((0,1)).tolist(),chebyshev=new.abs().amax((0,1)).tolist()),
        limitation='Smooth polynomial identity; unchanged coefficient quantization acts after the map. This is an optimizer-coordinate change, not a new function class or variance rescaling.')


def relative(a,b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()


def check():
    result=cpu_check()
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(1069);torch.set_num_threads(1)
    e,n,d=8,1024,768;core.KS3_RANK=1
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.025).bfloat16()) for _ in range(e)])
    checks=[]
    for tuned,tokens in ((True,1),(True,257),(True,4096),(True,16384),(False,257)):
        core.BF16_TUNED_KERNELS=tuned
        xx=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16).requires_grad_()
        router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
        # Explicit empty experts / padded token coverage independent of router test.
        assignment=torch.zeros(tokens,device='cuda',dtype=torch.int64) if tokens==257 else torch.randint(e,(tokens,),device='cuda')
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignment,e)
        post=torch.tensor(.8,device='cuda',requires_grad=True)
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(xx)*.02;gate=torch.rand(tokens,device='cuda')
        for zero in (True,False):
            theta=(torch.zeros(e,n,8,device='cuda') if zero else torch.randn(e,n,8,device='cuda')*.025)
            theta[...,6]=.75;theta[...,7]=0;theta.requires_grad_()
            mapped=basis.reference_map(theta)
            assert torch.equal(basis.transform(theta),mapped)
            old=basis.ORIGINAL_FORWARD(xx,src,tiles,up,down,mapped)
            new=basis.forward(xx,src,tiles,up,down,theta)
            assert all(torch.equal(a,b) for a,b in zip(old,new)),(tuned,tokens,zero)
            original_backward=basis.gather.backward if tuned else basis.ORIGINAL_BACKWARD
            dx,ds,dg=original_backward(grad,*old,gate,src,offs,tiles,up,down,mapped,bank,0)
            before_bank=bank.clone()
            expected=torch.autograd.grad(mapped,theta,dg,retain_graph=True)[0]
            after=basis.backward(grad,*new,gate,src,offs,tiles,up,down,theta,bank,0)
            assert torch.equal(dx,after[0]) and torch.equal(ds,after[1]) and torch.equal(before_bank,bank)
            chain_error=relative(after[2],expected);assert chain_error<2e-6,chain_error
            def routed(mapped_input):
                saved=core.forward,core.backward
                core.forward=basis.ORIGINAL_FORWARD if mapped_input else basis.forward
                core.backward=original_backward if mapped_input else basis.backward
                try:
                    g=basis.reference_map(theta) if mapped_input else theta
                    out,probs,ct=core.RoutedKS3.apply(xx,router,post,up,down,up,down,g,bank,0,True)
                    gradients=torch.autograd.grad(out,(xx,router,post,theta),grad)
                    return out.detach(),[v.detach().clone() for v in gradients],bank.clone()
                finally:core.forward,core.backward=saved
            before=routed(True);after_r=routed(False)
            assert torch.equal(before[0],after_r[0]) and torch.equal(before[2],after_r[2])
            errors={name:relative(a,b) for name,a,b in zip(('input','router','post','theta'),after_r[1],before[1])}
            assert all(torch.equal(a,b) for a,b in zip(after_r[1][:3],before[1][:3]))
            assert errors['theta']<2e-6,errors
            if zero:
                # Same original initial forward and nongeometry gradients.
                initial=basis.ORIGINAL_FORWARD(xx,src,tiles,up,down,theta)
                assert all(torch.equal(a,b) for a,b in zip(initial,new))
            row=dict(tuned=tuned,tokens=tokens,zero_coefficients=zero,active_experts=int((counts>0).sum()),
                padded_rows=src.numel(),mapped_values_exact=True,original_mapped_forward_exact=True,
                original_input_router_post_and_weight_gradients_exact=True,geometry_chain_relative_rms=chain_error,
                routed_gradient_relative_rms=errors)
            checks.append(row);print(row,flush=True)
    result.update(checks=checks,numerical_seed=1069,passed=True,full_graph_training_tested=False,
                  full_convergence_verified=False,shared_kernels_unchanged=True)
    return result


if __name__=='__main__':
    import json
    print(json.dumps(cpu_check(),indent=2))
