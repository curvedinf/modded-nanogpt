"""Independent upstream-forward and smooth-STE autograd checks for chart ranks."""
import torch
import torch.nn.functional as F
import triton as tr


def surrogate(coords,geometry,rank):
    """Pinned upstream FP32 STE equations, with explicit per-chart sharing."""
    n=coords.shape[-1]
    coeff=geometry[...,:6*rank].reshape(n,6)
    scale=(coeff.detach().abs().amax(-1,keepdim=True)/127).clamp(2**-24,65504).half().float()
    rounded=(coeff.detach()/scale).round().clamp(-127,127)*scale
    c=coeff+(rounded-coeff).detach()
    c0,c1,c2,c3,c4,c5=c.unbind(-1)
    raw_scale=geometry[...,6*rank].repeat_interleave(rank)
    raw_shift=geometry[...,6*rank+1].repeat_interleave(rank)
    warp_scale=F.softplus(raw_scale)+1e-4;shift=.5*raw_shift.tanh()
    x=coords.clamp(-65504,65504)
    mean=x.reshape(-1,n//rank,rank).mean(-1).repeat_interleave(rank,-1)
    t=4*(mean*.25).tanh();z=(warp_scale*(t+shift)).tanh()
    spine=(((2*c0-2*c1+c2+c3)*t+(-3*c0+3*c1-2*c2-c3))*t+c2)*t+c0
    warped=((4*c5*z+2*c4)*z-3*c5)*z-c4
    return (x+spine+warped).relu().square()


def check(rank=2,features=1024):
    from wr_reproduction import ks3_moe as bf
    from ks3_research.vendor.ks3_int8 import pack_spine,_geometry_quant
    torch.manual_seed(169)
    e,m,n=2,512,features;charts=n//rank
    coords=torch.randn(m,n,device='cuda')*2
    coords[-3:]=0
    value=torch.randn_like(coords)*.01
    g=torch.randn(e,charts,6*rank+2,device='cuda')*.02
    g[...,6*rank]=.75+torch.randn(e,charts,device='cuda')*.1
    tiles=torch.tensor([0,0,1,1],device='cuda',dtype=torch.int32)
    offs=torch.tensor([256,512],device='cuda',dtype=torch.int32)
    src=torch.arange(m,device='cuda',dtype=torch.int32)
    h=torch.empty_like(coords,dtype=torch.bfloat16)
    bf._geometry[(tr.cdiv(m,64),tr.cdiv(n,32))](coords,g,h,tiles,m,n,BM=64,RANK=rank,num_warps=4)
    expected=[]
    for expert in range(e):
        c=coords[expert*256:(expert+1)*256].contiguous()
        coeff=g[expert,:,:6*rank].reshape(n,6).contiguous()
        packed,scales=pack_spine(coeff)
        mean=c.clamp(-65504,65504).reshape(256,charts,rank).mean(-1)
        scale=F.softplus(g[expert,:,6*rank])+1e-4;shift=.5*g[expert,:,6*rank+1].tanh()
        codes=torch.empty_like(c,dtype=torch.int8)
        hs=torch.empty(256,tr.cdiv(n,32),device='cuda',dtype=torch.float16)
        _geometry_quant[(16,tr.cdiv(n,32))](c,mean,packed,scales,scale,shift,None,codes,hs,
                                          256,n,rank,1,BM=16,BN=32,num_warps=4)
        expected.append((codes.float()*hs.float().repeat_interleave(32,-1)).bfloat16())
    forward_reference=torch.cat(expected)
    dc=torch.empty_like(coords,dtype=torch.bfloat16)
    partial=torch.empty(tr.cdiv(m,64),n,8,device='cuda',dtype=torch.float32)
    dg=torch.empty_like(g)
    bf._vjp[(tr.cdiv(m,64),tr.cdiv(n,32))](value,coords,g,dc,partial,tiles,src,m,n,m-3,
                                        RANK=rank,num_warps=4)
    bf._reduce[(e,n)](partial,dg,offs,n,partial.shape[0],tr.next_power_of_2(partial.shape[0]),RANK=rank,num_warps=4)
    with torch.enable_grad():
        x=coords.detach().requires_grad_();geometry=g.detach().requires_grad_()
        outputs=torch.cat([surrogate(x[i*256:(i+1)*256],geometry[i],rank) for i in range(e)])
        v=value.clone();v[-3:]=0
        (outputs*v).sum().backward()
    errors={}
    for name,actual,reference,tolerance in (
            ('forward_vs_pinned_upstream',h,forward_reference,.01),
            ('coordinate_vjp_vs_autograd',dc,x.grad.bfloat16(),.002),
            ('geometry_vjp_vs_autograd',dg,geometry.grad,.0002)):
        assert actual.isfinite().all() and reference.isfinite().all(),name
        rms=((actual.float()-reference.float()).square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12)).item()
        assert rms<tolerance,(name,rms,tolerance)
        errors[name]=rms
    assert (dc[-3:]==0).all(),'Padded rows received coordinate gradients'
    return dict(rank=rank,features=features,charts=charts,relative_rms=errors,
                padding_gradient_zero=bool((dc[-3:]==0).all()),
                reference='Byte-identical pinned upstream _geometry_quant and independent FP32 STE autograd')
