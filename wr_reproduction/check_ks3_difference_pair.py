"""Compact coordinate-difference equations, cross derivatives and routed gradients."""
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_difference_pair as local
from wr_reproduction.check_ks3_pairwise import error, base_derivative, matrix


def cpu_check():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3569)
        c=torch.tensor([[-.7,-.9,1.3,-2.1,.2,-.3,2.,2.5]],dtype=torch.float64,requires_grad=True)
        h=torch.linspace(-2.,2.,8,dtype=torch.float64).reshape(1,8).requires_grad_()
        p=(torch.randn(8,dtype=torch.float64)*.2).requires_grad_()
        assert torch.autograd.gradcheck(local.reference,(c,h,p))
        assert torch.autograd.gradgradcheck(local.reference,(c,h,p))
        zero=torch.zeros_like(p,requires_grad=True)
        initial=local.reference(c,h,zero)
        assert torch.equal(initial,h)
        dc,dh,dp=torch.autograd.grad(initial.sum(),(c,h,zero))
        assert torch.equal(dc,torch.zeros_like(c)) and torch.equal(dh,torch.ones_like(h))
        assert (dp[[0,1,4,5,6,7]]>0).all() and torch.equal(dp[2:4],torch.zeros_like(dp[2:4]))
        raw=torch.full((8,),.3,dtype=torch.float64)
        coupled=local.reference(c,c.square(),raw)[0,0]
        first=torch.autograd.grad(coupled,c,create_graph=True)[0][0,0]
        cross=torch.autograd.grad(first,c)[0][0,1]
        difference=c[0,0]-c[0,1]
        expected=raw[0].tanh()*(1.-.75*difference.square())
        assert cross.abs()>.001 and torch.allclose(cross,expected,rtol=1e-12,atol=1e-12)
        grid_values=torch.cat((torch.linspace(-3,3,4001,dtype=torch.float64),torch.tensor([-100000.,100000.],dtype=torch.float64)))
        grid=torch.stack((grid_values,torch.zeros_like(grid_values)),dim=1)
        extreme=torch.tensor([50.,-50.],dtype=torch.float64)
        term=local.reference(grid,torch.zeros_like(grid),extreme)
        assert term.abs().max()==1.
        equal=torch.tensor([[-2.,-2.,0.,0.,2.,2.]],dtype=torch.float64)
        eqraw=torch.full_like(equal,.3)
        assert torch.equal(local.reference(equal,torch.zeros_like(equal),eqraw),eqraw.tanh())
        opposite=torch.tensor([[-.2,.2]],dtype=torch.float64)
        assert (local.reference(opposite,torch.zeros_like(opposite),torch.ones_like(opposite))>0).all()
        pairs=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=torch.float64)
        values=local.reference(pairs,torch.zeros_like(pairs),torch.ones_like(pairs))[:,0]
        determinant=values[0]*values[3]-values[1]*values[2]
        assert determinant.abs()>.1,'Basis must not factor into one function per coordinate'
        assert torch.equal(local.reference(pairs+2.,torch.zeros_like(pairs),torch.ones_like(pairs))[:,0],values)
        for edge in (-2.,2.):
            boundary=torch.tensor([[edge-1e-7,0.,edge,0.,edge+1e-7,0.]],dtype=torch.float64,requires_grad=True)
            value=local.reference(boundary,torch.zeros_like(boundary),torch.ones_like(boundary))
            derivative=torch.autograd.grad(value.sum(),boundary)[0]
            assert value.abs().max()<1e-12 and derivative.abs().max()<2e-6
            assert value[0,2]==0 and derivative[0,2]==0
        outside=torch.tensor([[3.,0.,-3.,0.]],dtype=torch.float64,requires_grad=True)
        outside_raw=torch.ones_like(outside,requires_grad=True)
        value=local.reference(outside,torch.zeros_like(outside),outside_raw)
        dc0,dp0=torch.autograd.grad(value.sum(),(outside,outside_raw))
        assert torch.count_nonzero(value)==torch.count_nonzero(dc0)==torch.count_nonzero(dp0)==0
        clamped=torch.tensor([[100000.,100001.,-100000.,-100001.]],dtype=torch.float64,requires_grad=True)
        value=local.reference(clamped,torch.zeros_like(clamped),torch.ones_like(clamped))
        clamp_gradient=torch.autograd.grad(value.sum(),clamped)[0]
        assert torch.count_nonzero(clamp_gradient)==0 and torch.isfinite(value).all()
    return dict(passed=True,gradcheck=True,gradgradcheck_away_from_support_boundary=True,
        initial_function_and_original_gradients_exact=True,active_pairs_have_zero_init_learning_signal=True,
        inactive_pairs_have_exact_zero_signal=True,cross_derivative=cross.item(),analytic_cross_derivative=expected.item(),
        absolute_residual_bound=1.,difference_width=2.,C1_support_boundaries_checked=True,
        exact_outside_support_zero_value_coordinate_and_coefficient_gradient=True,
        equal_coordinates_same_basis_across_sign_quadrants=True,opposite_sign_nearby_coordinates_supported=True,
        nonseparable_basis_determinant=determinant.item(),common_coordinate_translation_invariance=True,
        clamped_coordinates_zero_gradient_outside_clamp=True,can_activate_zero_independent_features=True,
        independent_rank_one_paths_retained=True,new_parameters=40960)


def boundary_gpu_check():
    """Exercise exact support edges, clamps, opposite signs and padding directly."""
    m,n,rows=128,32,125
    sample=torch.tensor([0.,0.,0.,2.,0.,-2.,100000.,100001.,-100000.,-100001.,
                         -.2,.2,-.7,-.9,3.,0.,0.,1.9999,0.,2.0001,0.,-1.9999,
                         0.,-2.0001,2.,2.,-2.,-2.,1.,0.,-1.,0.],device='cuda')
    c=sample.repeat(m,1).requires_grad_()
    raw=torch.linspace(-.4,.5,n,device='cuda').reshape(1,n).requires_grad_()
    base=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
    v=torch.randn(m,n,device='cuda');v[rows:]=0
    tiles=torch.zeros(1,device='cuda',dtype=torch.int32)
    src=torch.arange(m,device='cuda',dtype=torch.int32).clamp_max(rows)
    offs=torch.tensor([m],device='cuda',dtype=torch.int32)
    out=torch.empty_like(base)
    local._add_features[(2,1)](c,raw,base,out,tiles,m,n,1.,BM=64,num_warps=4)
    exact=local.reference(c,base.float(),raw)
    dcr,dpr=torch.autograd.grad(exact,(c,raw),v)
    dc=torch.empty_like(c);part=torch.empty((2,n),device='cuda')
    local._add_vjp[(2,1)](v,c,raw,torch.zeros_like(c),dc,part,tiles,src,m,n,rows,1.,num_warps=4)
    dp=torch.empty_like(raw)
    local.routed._reduce_pair[(1,1)](part,dp,offs,n,2,2,num_warps=4)
    errors=dict(output=error(out,exact.bfloat16()),coordinate=error(dc,dcr),coefficient=error(dp,dpr))
    assert errors['output']<.003 and max(errors['coordinate'],errors['coefficient'])<2e-5,errors
    assert torch.count_nonzero(dc[rows:])==0
    assert torch.count_nonzero(dc[:,[2,3,4,5,6,7,8,9,14,15,18,19,22,23]])==0
    return dict(passed=True,relative_rms=errors,exact_zero_gradients_at_support_edges_and_clamped_or_padded_rows=True)


def check(settings):
    local.configure(settings); cpu=cpu_check()
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    torch.manual_seed(3569);torch.set_num_threads(1);core.KS3_RANK=1
    boundary=boundary_gpu_check()
    e,n,d=8,1024,768;bound=settings['bound'];checks=[]
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.02).bfloat16()) for _ in range(e)])
    for tuned,tokens,zero_geometry in ((True,1,False),(True,257,True),(True,257,False),(True,4096,False),(True,16384,False),(False,257,False)):
        core.BF16_TUNED_KERNELS=tuned
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        assignments=torch.randint(e,(tokens,),device='cuda')
        if tokens==257:assignments.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(assignments,e)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        g=torch.randn(e,n,8,device='cuda')*.01
        if zero_geometry:g.zero_()
        g[...,6]=.75
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        old=core.forward(x,src,tiles,up,down,g)
        oldb=core.backward(grad,*old,gate,src,offs,tiles,up,down,g,bank,0);oldbank=bank.clone()
        zero=torch.zeros(e,n,device='cuda')
        new=local.forward(x,src,tiles,up,down,g,zero,bound)
        newb=local.backward(grad,*new,gate,src,offs,tiles,up,down,g,zero,bound,bank,0)
        names=('output','input_pack','coordinates','packed_features','input_gradient','gate_gradient','geometry_gradient','weight_gradient')
        exact={k:torch.equal(a,b) for k,a,b in zip(names,(*new,*newb[:3],bank),(*old,*oldb,oldbank))}
        initial_error={k:error(a,b) for k,a,b in zip(names,(*new,*newb[:3],bank),(*old,*oldb,oldbank))}
        assert all(exact.values()),(tuned,tokens,exact,initial_error)
        assert newb[3].abs().max()>0
        # Actual routed wrapper also verifies router/post and projection banks.
        xx=x.clone().requires_grad_();router=(torch.randn(e,d,device='cuda')*.02).bfloat16().requires_grad_()
        post=torch.tensor(.8,device='cuda',requires_grad=True);ggg=g.clone().requires_grad_()
        zr=zero.clone().requires_grad_()
        a,_,_=core.RoutedKS3.apply(xx,router,post,up,down,up,down,ggg,bank,0,True)
        ag=torch.autograd.grad(a,(xx,router,post,ggg),grad);ab=bank.clone()
        b,_,_=local.RoutedDifferencePair.apply(xx,router,post,up,down,up,down,ggg,zr,bank,0,bound,True)
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

        raw=torch.randn(e,n,device='cuda')*.35
        y,xp,c,h=local.forward(x,src,tiles,up,down,g,raw,bound)
        dx,ds,dg,dp=local.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,raw,bound,bank,0)
        m=c.shape[0];bm=64 if tuned else 16
        baseline=torch.empty_like(c,dtype=torch.bfloat16)
        core._geometry[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,baseline,tiles,m,n,BM=bm,RANK=1,GS32=True,num_warps=4)
        expert_rows=tiles.repeat_interleave(128)[:m].long()
        cr=c.clone().requires_grad_();pr=raw.clone().requires_grad_()
        bh=baseline.float().clone().requires_grad_()
        with torch.enable_grad():full=local.reference(cr,bh,pr[expert_rows],bound)
        hr=full.detach().bfloat16();yr=matrix(hr,down.transpose(1,2),tiles,x.dtype)
        valid=(src<tokens)[:,None];index=src.clamp(0,tokens-1).long();gathered=grad[index].float()
        gg=torch.where(valid,gathered*gate[index,None].bfloat16().float(),0).bfloat16()
        vr=matrix(gg,down,tiles)
        with torch.enable_grad():weighted,dcp,dpr=torch.autograd.grad(full,(bh,cr,pr),torch.where(valid,vr,0))
        dc0,dgr=base_derivative(weighted,c,g,tiles,src,offs,tokens)
        # Independent smooth autograd checks the new cross-coordinate VJP.
        smooth=torch.empty_like(c)
        part=torch.empty((tr.cdiv(m,64),n),device=c.device,dtype=torch.float32)
        local._add_vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](vr,c,raw,torch.zeros_like(c),smooth,part,
            tiles,src,m,n,tokens,bound,num_warps=4)
        smooth_dp=torch.empty_like(raw)
        local.routed._reduce_pair[(e,tr.cdiv(n,32))](part,smooth_dp,offs,n,part.shape[0],
            tr.next_power_of_2(part.shape[0]),num_warps=4)
        smooth_errors=dict(coordinate=error(smooth,dcp),coefficient=error(smooth_dp,dpr))
        assert max(smooth_errors.values())<2e-5,smooth_errors
        # Preserve original BF16 geometry-VJP storage before partner addition.
        dcr=(dc0.bfloat16().float()+dcp).bfloat16();dxr=matrix(dcr,up,tiles,x.dtype);br=torch.zeros_like(bank)
        for expert in range(e):
            lo=0 if expert==0 else int(offs[expert-1]);hi=int(offs[expert])
            br[expert]=(dcr[lo:hi].float().T@xp[lo:hi].float()).bfloat16()
            br[e+expert]=(hr[lo:hi].float().T@gg[lo:hi].float()).bfloat16()
        errors={k:error(a,b) for k,a,b in [('features',h,hr),('output',y,yr),('input_gradient',dx,dxr),
            ('geometry_gradient',dg,dgr),('difference_pair_gradient',dp,dpr),('weight_gradient',bank,br)]}
        assert errors['features']<.003 and errors['output']<.003,errors
        assert errors['input_gradient']<.004 and errors['weight_gradient']<.004,errors
        assert errors['geometry_gradient']<2e-5 and errors['difference_pair_gradient']<2e-5,errors
        actual_gate=torch.where(valid,gathered*y.float(),0).sum(1)
        assert torch.equal(ds,actual_gate)
        independent_gate=torch.where(valid,gathered*yr.float(),0).sum(1)
        size=torch.where(valid,gathered.abs()*torch.maximum(y.float().abs(),yr.float().abs()),0).sum(1)
        gate_bound=torch.where(valid,gathered.abs()*(y.float()-yr.float()).abs(),0).sum(1)+2e-6*size
        assert ((ds-independent_gate).abs()<=gate_bound).all()
        check=dict(tuned=tuned,tokens=tokens,zero_geometry=zero_geometry,active_experts=int((counts>0).sum()),padded_rows=m,
            zero_initialization_bitwise=exact,zero_initialization_relative_rms=initial_error,
            routed_initial_output_input_router_post_weight_bitwise_equal=True,routed_initial_geometry_relative_rms=routed_geometry_error,
            nonzero_relative_rms=errors,smooth_added_derivative_relative_rms=smooth_errors,
            actual_gate_gradient_bitwise_equal=True,independent_gate_gradient_forward_error_bound_satisfied=True)
        checks.append(check);print(check,flush=True)
    return dict(passed=True,cpu_checks=cpu,gpu_boundary_checks=boundary,checks=checks,numerical_seed=3569,
        original_independent_rank_one_paths_retained=True,additional_parameters=40960,
        initial_output_BF16_input_router_post_weight_gradients_bitwise_equal=True,
        original_geometry_gradient_bitwise_equal=True,
        smooth_new_gradient_relative_rms_tolerance=2e-5,
        oracle='Original independent geometry and BF16 coordinate-VJP boundary plus independent Torch coordinate-difference/cross-coordinate autograd and groupedFP32 matrix products withBF16 storage. Padding follows original dispatch tiles andSRC validity.',
        full_graph_training_tested=False,full_convergence_verified=False)
