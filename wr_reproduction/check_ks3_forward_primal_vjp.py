"""Independent forward-rounded multiplier / smooth-Jacobian STE checks."""
from unittest.mock import patch
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_forward_primal_vjp as candidate
from wr_reproduction.ks3_fp32_horner import reference_coefficients
from wr_reproduction.check_ks3_fp16_coefficients import matrix_reference, relative


def rounded(x):
    return x.half().float()


def primal(c, g):
    """Independent Torch copy of original FP16 stages, before squaring."""
    coeff=g[..., :6]
    ps=rounded((coeff.abs().amax(-1)/127).clamp(2**-24,65504))
    c0,c1,c2,c3,c4,c5=(coeff/ps[...,None]).round().clamp(-127,127).unbind(-1)
    x=c.clamp(-65504,65504)
    t=rounded(4*torch.tanh(rounded(x*.25)))
    scale=torch.nn.functional.softplus(g[...,6])+1e-4
    shift=.5*torch.tanh(g[...,7])
    z=rounded(torch.tanh(rounded(scale*(t+shift))))
    a=2*c0-2*c1+c2+c3; b=-3*c0+3*c1-2*c2-c3
    poly=rounded(rounded(a)*t+rounded(b))
    poly=rounded(poly*t+rounded(c2)); poly=rounded(poly*t+rounded(c0))
    warp=rounded(4*rounded(c5)*z+2*rounded(c4))
    warp=rounded(warp*z-3*rounded(c5)); warp=rounded(warp*z-rounded(c4))
    return rounded(x+(poly+warp)*ps).clamp_min(0.)


def smooth(c, packed):
    x=c.clamp(-65504,65504)
    c0,c1,c2,c3,c4,c5,rs,rh=packed.unbind(-1)
    t=4*torch.tanh(x*.25)
    z=torch.tanh((torch.nn.functional.softplus(rs)+1e-4)*(t+.5*torch.tanh(rh)))
    a=2*c0-2*c1+c2+c3; b=-3*c0+3*c1-2*c2-c3
    return x+((a*t+b)*t+c2)*t+c0+((4*c5*z+2*c4)*z-3*c5)*z-c4


def reference_vjp(c,g,v,src,token_count,offsets):
    cc=c.detach().clone().requires_grad_(); gg=g.detach().clone().requires_grad_()
    packed=reference_coefficients(gg,ste=True)
    terms=[];lo=0
    for e,hi in enumerate(offsets):
        if hi==lo:continue
        s=smooth(cc[lo:hi],packed[e])
        # The detached correction defines the intended hybrid STE exactly:
        # forward value is rounded ReLU, derivative is the original smooth map.
        r=primal(cc[lo:hi].detach(),gg[e].detach())
        proxy=s+(r-s).detach()
        cotangent=torch.where((src[lo:hi]<token_count)[:,None],v[lo:hi],0.)
        terms.append((proxy.square()*cotangent).sum());lo=hi
    return torch.autograd.grad(sum(terms),(cc,gg))


def cpu_check():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3369)
        g=torch.randn(2,32,8)*.02;g[...,6]=.75
        c=torch.randn(384,32)*1.3-.4
        src=torch.arange(384);src[150:]=150
        v=torch.randn_like(c)
        dc,dg=reference_vjp(c,g,v,src,150,[128,256])
        assert torch.count_nonzero(dc[150:])==0
        assert torch.isfinite(dc).all() and torch.isfinite(dg).all()
        # Saturated coordinate clamp must retain its original zero derivative.
        z=torch.zeros(1,32,8);z[...,6]=.75
        extreme=torch.full((128,32),70000.)
        dx,_=reference_vjp(extreme,z,torch.ones_like(extreme),torch.arange(128),128,[128])
        assert torch.count_nonzero(dx)==0
        z[..., :6]=0
        assert torch.equal(primal(c[:128],z[0]),rounded(c[:128]).clamp_min(0))
        # With zero geometry the expected new multiplier is2*ReLU(round16(x)).
        dx,_=reference_vjp(c[:128],z,torch.ones_like(c[:128]),torch.arange(128),128,[128])
        assert torch.allclose(dx,2*rounded(c[:128]).clamp_min(0),rtol=0,atol=1e-6)
    return dict(passed=True,explicit_hybrid_STE=True,padded_rows_zero_gradient=True,
        original_coordinate_clamp=True,zero_geometry_multiplier_checked=True,
        initial_function_unchanged=True,initial_gradient_expected_to_change=True,
        no_claim_of_derivative_of_discrete_FP16_map=True)


def check():
    cpu=cpu_check()
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(3369);torch.set_num_threads(1);core.KS3_RANK=1
    e,n,d=8,1024,768;checks=[]
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.025).bfloat16()) for _ in range(e)])
    for tokens in (1,257,4096,16384):
        core.BF16_TUNED_KERNELS=True
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        assignments=torch.zeros(tokens,device='cuda',dtype=torch.int64) if tokens==257 else torch.randint(e,(tokens,),device='cuda')
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignments,e);offsets=offs.cpu().tolist()
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        for zero in (True,False):
            g=torch.randn(e,n,8,device='cuda')*.015;g[...,4:6]*=2;g[...,6]=.75;g[...,7]=0.
            if zero:g[...,:6].zero_()
            original=core.forward(x,src,tiles,up,down,g)
            with patch.object(core,'_vjp',candidate.vjp):
                actual=core.forward(x,src,tiles,up,down,g)
                got=core.backward(grad,*actual,gate,src,offs,tiles,up,down,g,bank,0)
                got_bank=bank.clone()
            assert all(torch.equal(a,b) for a,b in zip(original,actual))
            y,xp,c,h=actual
            valid=(src<tokens)[:,None];idx=src.clamp(0,tokens-1).long()
            gg=torch.where(valid,grad[idx].float()*gate[idx,None].bfloat16().float(),0.).bfloat16()
            v=core.mm(gg,down,tiles)
            with torch.enable_grad():dc_ref,dg_ref=reference_vjp(c,g,v,src,tokens,offsets)
            dc=torch.empty_like(c);part=torch.empty(tr.cdiv(c.shape[0],64),n,8,device='cuda')
            candidate.vjp[(tr.cdiv(c.shape[0],64),tr.cdiv(n,32))](
                v,c,g,dc,part,tiles,src,*c.shape,tokens,num_warps=4)
            dg=torch.empty_like(g)
            core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),num_warps=4)
            dc_error=relative(dc,dc_ref);dg_error=relative(dg,dg_ref)
            components=[relative(dg[...,i],dg_ref[...,i]) for i in range(8)]
            # Same independent-gradient bounds as preceding numerical checks.
            assert max(dc_error,dg_error,*components)<3e-5,('explicit STE',tokens,zero,dc_error,dg_error,components)
            dx_ref,dw_ref=matrix_reference(dc_ref,h,xp,gg,up,offsets)
            dx_error=relative(got[0],dx_ref);dw_error=relative(got_bank,dw_ref)
            assert dx_error<.003 and dw_error<.003,('input/weight',tokens,zero,dx_error,dw_error)
            actual_geometry_error=relative(got[2],dg_ref)
            actual_components=[relative(got[2][...,i],dg_ref[...,i]) for i in range(8)]
            assert max(actual_geometry_error,*actual_components)<3e-5, ('actual BF16-dc backward geometry', tokens, zero, actual_geometry_error, actual_components, 'isolated FP32-dc geometry', dg_error, components, 'actual versus isolated', relative(got[2],dg))
            old=core.backward(grad,*original,gate,src,offs,tiles,up,down,g,bank,0)
            assert torch.equal(got[1],old[1]) and torch.equal(got_bank[e:],bank[e:])
            changed=dict(input=relative(got[0],old[0]),geometry=relative(got[2],old[2]),
                input_projection_weight=relative(got_bank[:e],bank[:e]))
            # Forward and evaluation remain original, including nonzero geometry.
            core.BF16_TUNED_KERNELS=False
            evaluation=core.forward(x,src,tiles,up,down,g)
            with patch.object(core,'_vjp',candidate.vjp):
                same=core.forward(x,src,tiles,up,down,g)
            assert all(torch.equal(a,b) for a,b in zip(evaluation,same))
            core.BF16_TUNED_KERNELS=True
            row=dict(tokens=tokens,zero_coefficients=zero,padded_rows=src.numel(),active_experts=int((counts>0).sum()),
                original_training_and_evaluation_forward_bitwise=True,original_gate_and_output_weight_gradient_bitwise=True,
                independent_coordinate_gradient_relative_rms=dc_error,independent_geometry_gradient_relative_rms=dg_error,
                independent_component_gradient_relative_rms=components,actual_BF16_dc_geometry_relative_rms=actual_geometry_error,actual_BF16_dc_geometry_components=actual_components,independent_input_gradient_relative_rms=dx_error,
                independent_weight_gradient_relative_rms=dw_error,change_from_original_gradient_relative_rms=changed)
            checks.append(row);print(row,flush=True)
    candidate.install();candidate.install()
    assert core.forward is candidate.ORIGINAL_FORWARD and core._geometry is candidate.ORIGINAL_GEOMETRY
    assert core._vjp is candidate.vjp
    return dict(passed=True,cpu_checks=cpu,numerical_seed=3369,checks=checks,
        new_parameters=0,original_forward_and_evaluation=True,original_coefficient_packing=True,
        original_smooth_polynomial_jacobian=True,only_squared_relu_multiplier_changed=True,
        initial_function_unchanged=True,initial_gradients_changed_as_specified=True,
        oracle='Independent Torch FP16 primal plus autograd of smooth+(rounded_relu-smooth).detach(), squared, with original joint-coefficient identity STE and BF16 matrix boundaries.',
        tolerance_scope='Independent coordinate and individual geometry gradients<3e-5 relativeRMS; input/weight<.003 including BF16 products. All forward/evaluation, gate gradient and output-weight gradients bitwise original.',
        actual_model_training_tested=False,full_convergence_verified=False)
