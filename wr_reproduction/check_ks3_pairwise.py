"""Bounded GPU gates for additive pairwise KS3; never convergence evidence."""
import argparse
from pathlib import Path
import torch
import triton as tr
from wr_reproduction import ks3_moe as core, ks3_pairwise as pair
from wr_reproduction.research import read,atomic,append


def error(a,b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()


def gs32(x):
    shape=x.shape;v=x.reshape(*shape[:-1],-1,32)
    scale=(v.abs().amax(-1,keepdim=True)/127).clamp(2**-24,65504).half().float()
    return ((v/scale).round().clamp(-127,127)*scale).reshape(shape).bfloat16()


def base_derivative(v,c,g,tiles,src,offs,rows):
    m,n=c.shape;e=g.shape[0];dc=torch.empty_like(c)
    part=torch.empty((tr.cdiv(m,64),n,8),device=c.device)
    core._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](v,c,g,dc,part,tiles,src,m,n,rows,RANK=1,num_warps=4)
    dg=torch.empty_like(g)
    core._reduce[(e,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),RANK=1,num_warps=4)
    return dc,dg


def matrix(a,w,tiles,dtype=torch.float32):
    out=torch.empty((a.shape[0],w.shape[-1]),device=a.device,dtype=dtype)
    for start in range(0,a.shape[0],128):
        expert=int(tiles[start//128])
        out[start:start+128]=(a[start:start+128].float()@w[expert].float()).to(dtype)
    return out


def check(coupling):
    placement=pair.configure(coupling)
    from wr_reproduction.backend import install
    install('pr393')
    from track_1_short.perf.kernels import moe_mlp as moe
    from ks3_research.pairwise_reference import pairwise_term
    torch.manual_seed(602);torch.set_num_threads(1)
    e,n,d=8,1024,768;bound=coupling['bound']
    up=core.pack((torch.randn(e*n,d,device='cuda')*.025).bfloat16()).view(e,n,d)
    down=torch.stack([core.pack((torch.randn(d,n,device='cuda')*.02).bfloat16()) for _ in range(e)])
    g=torch.randn(e,n,8,device='cuda')*.01;g[...,6]=.75
    checks=[]
    for tuned,tokens in ((True,1),(True,257),(True,4096),(False,257)):
        core.BF16_TUNED_KERNELS=tuned
        x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)
        experts=torch.randint(e,(tokens,),device='cuda')
        if tokens==257:experts.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(experts,e)
        grad=torch.randn_like(x)*.02;gate=torch.rand(tokens,device='cuda')
        bank=torch.zeros(16,n,d,device='cuda',dtype=torch.bfloat16)
        old=core.forward(x,src,tiles,up,down,g)
        oldb=core.backward(grad,*old,gate,src,offs,tiles,up,down,g,bank,0)
        oldbank=bank.clone();zero=torch.zeros(e,n,device='cuda')
        new=pair.forward(x,src,tiles,up,down,g,zero,bound)
        newb=pair.backward(grad,*new,gate,src,offs,tiles,up,down,g,zero,bound,bank,0)
        exact={name:torch.equal(a,b) for name,a,b in zip(('output','input_pack','coordinates','packed_features','input_gradient','gate_gradient','geometry_gradient','weight_gradient'),(*new,*newb[:3],bank),(*old,*oldb,oldbank))}
        zero_errors={name:error(a,b) for name,a,b in zip(('output','input_pack','coordinates','packed_features','input_gradient','gate_gradient','geometry_gradient','weight_gradient'),(*new,*newb[:3],bank),(*old,*oldb,oldbank))}
        print(dict(event='zero_initialization_check',tuned=tuned,tokens=tokens,exact=exact,relative_rms=zero_errors),flush=True)
        # Changing the output pointer type of the reused STE kernel can alter
        # FP32 compiler reassociation in its coefficient reduction. Preserve
        # exact forward/BF16 products; separately bound FP32 geometry roundoff.
        assert all(value for name,value in exact.items() if name!='geometry_gradient'),(tuned,tokens,exact,zero_errors)
        assert zero_errors['geometry_gradient']<2e-6,(tuned,tokens,zero_errors)
        assert newb[3].abs().max()>0,'zero initialization must have learnable pair coefficients'

        raw=torch.randn(e,n,device='cuda')*.35
        y,xp,c,h=pair.forward(x,src,tiles,up,down,g,raw,bound)
        dx,ds,dg,dp=pair.backward(grad,y,xp,c,h,gate,src,offs,tiles,up,down,g,raw,bound,bank,0)
        m=c.shape[0];bm=64 if tuned else 16
        base=torch.empty_like(c,dtype=torch.bfloat16 if pair.AFTER_GS32 else torch.float32)
        core._geometry[(tr.cdiv(m,bm),tr.cdiv(n,32))](c,g,base,tiles,m,n,BM=bm,RANK=1,GS32=pair.AFTER_GS32,num_warps=4)
        expert_rows=tiles.repeat_interleave(128)[:m].long()
        cr=c.detach().clone().requires_grad_();pr=raw.detach().clone().requires_grad_()
        with torch.enable_grad():term=pairwise_term(cr.clamp(-65504,65504),pr[expert_rows],bound)
        hr=(base.float()+term.detach()).bfloat16() if pair.AFTER_GS32 else gs32(base+term.detach())
        yr=matrix(hr,down.transpose(1,2),tiles,x.dtype)
        valid=(src<tokens)[:,None];index=src.clamp(0,tokens-1).long()
        gathered=grad[index].float()
        gg=torch.where(valid,gathered*gate[index,None].bfloat16().float(),0).bfloat16()
        vr=matrix(gg,down,tiles)
        dc0,dgr=base_derivative(vr,c,g,tiles,src,offs,tokens)
        with torch.enable_grad():dcp,dpr=torch.autograd.grad(term,(cr,pr),torch.where(valid,vr,0))
        dcr=(dc0+dcp).bfloat16();dxr=matrix(dcr,up,tiles,x.dtype)
        dsr=torch.where(valid,gathered*yr.float(),0).sum(1)
        br=torch.zeros_like(bank)
        for expert in range(e):
            lo=0 if expert==0 else int(offs[expert-1]);hi=int(offs[expert])
            br[expert]=(dcr[lo:hi].float().T@xp[lo:hi].float()).bfloat16()
            br[e+expert]=(hr[lo:hi].float().T@gg[lo:hi].float()).bfloat16()
        errors={name:error(a,b) for name,a,b in (
            ('packed_features',h,hr),('output',y,yr),('input_gradient',dx,dxr),
            ('gate_gradient',ds,dsr),('geometry_gradient',dg,dgr),('pair_gradient',dp,dpr),('weight_gradient',bank,br))}
        # A one-row gate dot product can nearly cancel. Verify its derivative
        # exactly for the actual forward values, then bound its difference from
        # the independent projection by the measured forward discrepancy.
        actual_gate_oracle=torch.where(valid,gathered*y.float(),0).sum(1)
        assert torch.equal(ds,actual_gate_oracle)
        weighted_error=torch.where(valid,gathered.abs()*(y.float()-yr.float()).abs(),0).sum(1)
        weighted_size=torch.where(valid,gathered.abs()*torch.maximum(y.float().abs(),yr.float().abs()),0).sum(1)
        gate_bound=weighted_error+2e-6*weighted_size
        assert ((ds-dsr).abs()<=gate_bound).all(),(ds,dsr,gate_bound)
        gate_check=dict(actual_forward_derivative_bitwise_equal=True,
            independent_max_abs_error=(ds-dsr).abs().max().item(),
            measured_output_error_bound_max=gate_bound.max().item(),
            output_bound_satisfied=True,
            weighted_relative_error=((ds-dsr).abs()/weighted_size.clamp_min(1e-12)).max().item(),
            independent_relative_error=errors['gate_gradient'],
            reason='Gate sums may cancel; use exact actual-forward derivative and measured independent-output absolute error bound, not relative error of a near-zero sum.')
        assert max(v for k,v in errors.items() if k!='gate_gradient')<.015,(tuned,tokens,errors)
        quantization_effect=None
        if tuned and tokens==4096 and pair.AFTER_GS32:
            small=torch.full_like(raw,.02)
            after=pair.features(c,g,small,tiles,bound)
            # Bounded same-coordinate comparison only, before any graph exists.
            pair.configure(dict(placement='before-GS32'))
            try:before=pair.features(c,g,small,tiles,bound)
            finally:pair.configure(coupling)
            quantization_effect=dict(raw_pair=.02,independent_feature_rms=base.float().square().mean().sqrt().item(),
                before_changed_fraction=(before!=base).float().mean().item(),
                after_changed_fraction=(after!=base).float().mean().item(),
                before_delta_rms=(before.float()-base.float()).square().mean().sqrt().item(),
                after_delta_rms=(after.float()-base.float()).square().mean().sqrt().item(),
                limitation='Synthetic actual-width coordinates; measures surviving BF16 feature effect, not convergence.')
            assert quantization_effect['after_changed_fraction']>quantization_effect['before_changed_fraction']
        checks.append(dict(tuned=tuned,tokens=tokens,active_experts=int((counts>0).sum()),zero_initialization_bitwise=exact,zero_initialization_relative_rms=zero_errors,
                           zero_pair_gradient_max=newb[3].abs().max().item(),nonzero_pair_relative_rms=errors,gate_gradient_check=gate_check,quantization_effect=quantization_effect))
        print(checks[-1],flush=True)

    # GPU finite difference of the analytic interaction cross-derivative, before
    # the deliberately nondifferentiable GS32/BF16 boundary. CPU gradgrad oracle
    # separately covers the smooth equations; quantization is treated with STE.
    m,n=128,32;c=torch.randn(m,n,device='cuda')*.4;p=torch.full((1,n),.2,device='cuda')
    v=torch.zeros_like(c);v[0,0]=1.;z=torch.zeros_like(c)
    src=torch.arange(m,device='cuda',dtype=torch.int32);tiles=torch.zeros(1,device='cuda',dtype=torch.int32)
    def interaction_dc(q):
        dc=torch.empty_like(c);part=torch.empty((2,n),device='cuda')
        pair._add_vjp[(2,1)](v,q,p,z,dc,part,tiles,src,m,n,m,1.,num_warps=4)
        return dc[0,0].item()
    delta=.002;plus=c.clone();minus=c.clone();plus[0,1]+=delta;minus[0,1]-=delta
    observed=(interaction_dc(plus)-interaction_dc(minus))/(2*delta)
    u=torch.tanh(c[0,:2]/4)
    expected=(torch.tanh(p[0,0])*(1-u[0]**2)*(1-u[1]**2)/16).item()
    assert expected!=0 and abs(observed-expected)<max(1e-5,abs(expected)*.002),(observed,expected)
    return dict(passed=True,checks=checks,pair_placement=placement,cross_derivative=dict(observed=observed,expected=expected),
                oracle='Original independent low-precision KS3 plus independent Torch smooth pair VJP and per-expert FP32 matrix products',
                added_term_bound=bound,initial_function_bitwise_equal=True,
                old_gradient_equivalence='Exact BF16 input/weight/gate gradients; FP32 geometry relative RMS below2e-6, measured individually',
                independent_rank_one_paths_retained=True,full_graph_training_tested=False,full_convergence_verified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cfg=read(a.run/'run.json')
    if any(Path(x).exists() for x in cfg['stop_files']):raise RuntimeError('Explicit study STOP')
    from wr_reproduction.probe_controller import sources
    if sources()!=cfg['source_hashes']:raise RuntimeError('Source hashes changed before numerical gate')
    value=check(cfg['recipe']['pairwise']);atomic(a.run/'calibration.json',value)
    append(a.run/'events.jsonl',dict(event='calibration_complete',mode='pairwise-numerical',**value))

if __name__=='__main__':main()
