"""Independent compact-residual equations, derivatives and actual-width checks."""
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_local_residual as local
from wr_reproduction.check_ks3_pairwise import error, base_derivative, matrix


def cpu_check():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1569)
        c=torch.tensor([[-3.3,-1.7,-.3,.8,2.7,4.8]],dtype=torch.float64,requires_grad=True)
        p=(torch.randn(6,3,dtype=torch.float64)*.2).requires_grad_()
        assert torch.autograd.gradcheck(local.reference,(c,p))
        assert torch.autograd.gradgradcheck(local.reference,(c,p))
        zero=torch.zeros_like(p,requires_grad=True)
        value=local.reference(c,zero)
        assert torch.equal(value,torch.zeros_like(c))
        dc,dp=torch.autograd.grad(value.sum(),(c,zero))
        assert torch.equal(dc,torch.zeros_like(c)) and (dp[:5].abs().sum(-1)>0).all()
        assert torch.count_nonzero(dp[5])==0
        grid=torch.linspace(-20,20,4001,dtype=torch.float64)[:,None]
        extreme=torch.full((1,3),50.,dtype=torch.float64)
        bounded=local.reference(grid,extreme)
        assert bounded.abs().max()<=3. and (bounded[grid.abs()>=4]==0).all()
        boundary=torch.tensor([[-4.,4.]],dtype=torch.float64,requires_grad=True)
        out=local.reference(boundary,torch.ones(2,3,dtype=torch.float64))
        assert torch.count_nonzero(out)==0
        assert torch.count_nonzero(torch.autograd.grad(out.sum(),boundary)[0])==0
        jac=torch.autograd.functional.jacobian(lambda z:local.reference(z,p),c)
        flat=jac.reshape(6,6)
        assert torch.count_nonzero(flat-torch.diag(torch.diagonal(flat)))==0
    return dict(passed=True,gradcheck=True,gradgradcheck=True,zero_function_and_coordinate_gradient=True,
        zero_coefficients_have_local_learning_signal=True,compact_support_outside_minus4_plus4=True,
        conservative_absolute_bound=3.,boundary_value_and_derivative_zero=True,
        independent_coordinate_paths=True,new_parameters=5*8*1024*3)


def check(settings):
    local.configure(settings); cpu=cpu_check()
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(1569);torch.set_num_threads(1);core.KS3_RANK=1
    e,n,d=8,1024,768;bound=settings['bound'];checks=[]
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.02).bfloat16()) for _ in range(e)])
    for tuned,tokens in ((True,1),(True,257),(True,4096),(True,16384),(False,257)):
        core.BF16_TUNED_KERNELS=tuned
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        assignments=torch.randint(e,(tokens,),device='cuda')
        if tokens==257:assignments.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        g=torch.randn(e,n,8,device='cuda')*.01;g[...,6]=.75
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        old=core.forward(x,src,tiles,up,down,g)
        oldb=core.backward(grad,*old,gate,src,offs,tiles,up,down,g,bank,0);oldbank=bank.clone()
        zero=torch.zeros(e,n,3,device='cuda')
        new=local.forward(x,src,tiles,up,down,g,zero,bound)
        newb=local.backward(grad,*new,gate,src,offs,tiles,up,down,g,zero,bound,bank,0)
        names=('output','input_pack','coordinates','packed_features','input_gradient','gate_gradient','geometry_gradient','weight_gradient')
        exact={k:torch.equal(a,b) for k,a,b in zip(names,(*new,*newb[:3],bank),(*old,*oldb,oldbank))}
        initial_error={k:error(a,b) for k,a,b in zip(names,(*new,*newb[:3],bank),(*old,*oldb,oldbank))}
        assert all(exact.values()),(tuned,tokens,exact,initial_error)
        assert all(newb[3][...,k].abs().max()>0 for k in range(3))
        # Actual routed wrapper also verifies router/post and projection banks.
        xx=x.clone().requires_grad_();router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
        post=torch.tensor(.8,device='cuda',requires_grad=True);ggg=g.clone().requires_grad_()
        zr=zero.clone().requires_grad_()
        a,_,_=core.RoutedKS3.apply(xx,router,post,up,down,up,down,ggg,bank,0,True)
        ag=torch.autograd.grad(a,(xx,router,post,ggg),grad);ab=bank.clone()
        b,_,_=local.RoutedLocalized.apply(xx,router,post,up,down,up,down,ggg,zr,bank,0,bound,True)
        bg=torch.autograd.grad(b,(xx,router,post,ggg,zr),grad)
        routed_pairs=[('output',b,a),('input_gradient',bg[0],ag[0]),('router_gradient',bg[1],ag[1]),
            ('post_gradient',bg[2],ag[2]),('geometry_gradient',bg[3],ag[3]),
            ('up_weight_gradient',bank[:e],ab[:e]),('down_weight_gradient',bank[e:],ab[e:])]
        print(dict(event='routed_initial_parity',tuned=tuned,tokens=tokens,
            comparisons={key:dict(bitwise=torch.equal(left,right),relative_rms=error(left,right),
                max_abs=(left.float()-right.float()).abs().max().item(),changed_elements=(left!=right).sum().item())
                for key,left,right in routed_pairs}),flush=True)
        assert torch.equal(a,b) and torch.equal(ab,bank)
        assert all(torch.equal(v,w) for v,w in zip(ag,bg[:4]))
        routed_geometry_error=error(bg[3],ag[3]);assert routed_geometry_error==0.

        raw=torch.randn(e,n,3,device='cuda')*.35
        y,xp,c,h=local.forward(x,src,tiles,up,down,g,raw,bound)
        dx,ds,dg,dp=local.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,raw,bound,bank,0)
        m=c.shape[0];bm=64 if tuned else 16
        baseline=torch.empty_like(c,dtype=torch.bfloat16)
        core._geometry[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,baseline,tiles,m,n,BM=bm,RANK=1,GS32=True,num_warps=4)
        expert_rows=tiles.repeat_interleave(128)[:m].long()
        cr=c.clone().requires_grad_();pr=raw.clone().requires_grad_()
        with torch.enable_grad():term=local.reference(cr,pr[expert_rows],bound)
        hr=(baseline.float()+term.detach()).bfloat16();yr=matrix(hr,down.transpose(1,2),tiles,x.dtype)
        valid=(src<tokens)[:,None];index=src.clamp(0,tokens-1).long();gathered=grad[index].float()
        gg=torch.where(valid,gathered*gate[index,None].bfloat16().float(),0).bfloat16()
        vr=matrix(gg,down,tiles)
        dc0,dgr=base_derivative(vr,c,g,tiles,src,offs,tokens)
        with torch.enable_grad():dcp,dpr=torch.autograd.grad(term,(cr,pr),torch.where(valid,vr,0))
        # Check new smooth derivative before any BF16 storage or matrix reduction.
        smooth=torch.empty_like(c);part=torch.empty((tr.cdiv(m,64),n,3),device=c.device)
        local._add_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](vr,c,raw,torch.zeros_like(c),smooth,part,tiles,src,m,n,tokens,bound,num_warps=4)
        smooth_dp=torch.empty_like(raw)
        local._reduce_local[(e,tr.cdiv(n,32))](part,smooth_dp,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),num_warps=4)
        smooth_errors=dict(coordinate=error(smooth,dcp),coefficient=error(smooth_dp,dpr),
            each_coefficient=[error(smooth_dp[...,k],dpr[...,k]) for k in range(3)])
        assert max(smooth_errors['coordinate'],smooth_errors['coefficient'],*smooth_errors['each_coefficient'])<2e-5
        # The independent branch retains its original BF16 VJP storage before
        # the smooth new derivative is combined and stored back to BF16.
        dcr=(dc0.bfloat16().float()+dcp).bfloat16();dxr=matrix(dcr,up,tiles,x.dtype);br=torch.zeros_like(bank)
        for expert in range(e):
            lo=0 if expert==0 else int(offs[expert-1]);hi=int(offs[expert])
            br[expert]=(dcr[lo:hi].float().T@xp[lo:hi].float()).bfloat16()
            br[e+expert]=(hr[lo:hi].float().T@gg[lo:hi].float()).bfloat16()
        errors={k:error(a,b) for k,a,b in [('features',h,hr),('output',y,yr),('input_gradient',dx,dxr),
            ('geometry_gradient',dg,dgr),('local_gradient',dp,dpr),('weight_gradient',bank,br)]}
        assert errors['features']<.003 and errors['output']<.003
        assert errors['input_gradient']<.004 and errors['weight_gradient']<.004
        assert errors['geometry_gradient']<2e-5 and errors['local_gradient']<2e-5
        actual_gate=torch.where(valid,gathered*y.float(),0).sum(1)
        assert torch.equal(ds,actual_gate)
        independent_gate=torch.where(valid,gathered*yr.float(),0).sum(1)
        size=torch.where(valid,gathered.abs()*torch.maximum(y.float().abs(),yr.float().abs()),0).sum(1)
        gate_bound=torch.where(valid,gathered.abs()*(y.float()-yr.float()).abs(),0).sum(1)+2e-6*size
        assert ((ds-independent_gate).abs()<=gate_bound).all()
        check=dict(tuned=tuned,tokens=tokens,active_experts=int((counts>0).sum()),padded_rows=m,
            zero_initialization_bitwise=exact,zero_initialization_relative_rms=initial_error,
            routed_initial_output_input_router_post_weight_bitwise_equal=True,routed_initial_geometry_relative_rms=routed_geometry_error,
            nonzero_relative_rms=errors,smooth_added_derivative_relative_rms=smooth_errors,
            actual_gate_gradient_bitwise_equal=True,independent_gate_gradient_forward_error_bound_satisfied=True)
        checks.append(check);print(check,flush=True)
    return dict(passed=True,cpu_checks=cpu,checks=checks,numerical_seed=1569,
        original_independent_rank_one_paths_retained=True,additional_parameters=122880,
        initial_output_BF16_input_router_post_weight_gradients_bitwise_equal=True,
        original_geometry_gradient_bitwise_equal=True,
        smooth_new_gradient_relative_rms_tolerance=2e-5,
        oracle='Original independent geometry and BF16 coordinate-VJP boundary plus independent Torch compact-residual autograd and groupedFP32 matrix products withBF16 storage. Padding follows original dispatch tiles andSRC validity.',
        full_graph_training_tested=False,full_convergence_verified=False)
