"""Exercise the unchanged PR393 MLP autograd class through MI300 FP8 kernels."""
import json
import sys
import torch
from wr_reproduction.contracts import ROOT,audit
from wr_reproduction.mlp_kernels import install

def main(version=None,tokens=256):
    version=version or (sys.argv[1] if len(sys.argv)>1 else 'pr393')
    audit(version)
    sys.path.insert(0,str(ROOT/version/'upstream'))
    from track_1_short.perf.kernels import mlp
    install(mlp)
    torch.manual_seed(393)
    device='cuda';n,d,h=tokens,768,2816
    x=torch.randn(n,d,device=device,dtype=torch.bfloat16,requires_grad=True)
    up=(torch.randn(h,d,device=device)*.02).to(torch.bfloat16).requires_grad_()
    down=(torch.randn(h,d,device=device)*.02).to(torch.bfloat16).requires_grad_()
    def scalar(v):return torch.tensor(v,device=device,dtype=torch.float32)
    xs,gs=scalar(2**-4),scalar(2**-6)
    us,ds=up.detach().float().abs().amax()/448,down.detach().float().abs().amax()/448
    ps,dps=scalar(.01),scalar(.01);pa,dpa=scalar(0),scalar(0)
    def q(v,s,dt):return (v.detach().float()/s).clamp(-torch.finfo(dt).max,torch.finfo(dt).max).to(dt)
    u8,d8,x8=q(up,us,torch.float8_e4m3fn),q(down,ds,torch.float8_e4m3fn),q(x,xs,torch.float8_e4m3fn)
    out=mlp.FusedLinearReLUSquareFunction.apply(x,up,down,u8,xs*us,x8,d8.T.contiguous().T,ds,
        ps,pa,d8,ds*gs,x8.T.contiguous(),xs,u8.T.contiguous().T,us,gs,dps,dpa,None)
    grad=torch.randn_like(out)*.01
    actual=torch.autograd.grad(out,(x,up,down),grad)
    # Independent explicit oracle using decoded FP8 values and FP32 dot products;
    # retains the record's nonstandard quantized STE and gradient reconstruction.
    pre=((x8.float()@u8.float().T)*(xs*us)).to(torch.bfloat16)
    post=pre.clamp_min(0).square()
    p8=q(post,ps,torch.float8_e4m3fn)
    expected=((p8.float()@d8.float())*(ps*ds)).to(torch.bfloat16)
    g8=q(grad,gs,torch.float8_e5m2)
    dp=(2*((g8.float()@d8.float().T)*(ds*gs))*torch.sqrt(p8.float()*ps)).to(torch.bfloat16)
    dp8=q(dp,dps,torch.float8_e5m2)
    oracle=[((dp8.float()@u8.float())*(dps*us)).to(torch.bfloat16),
            ((dp8.float().T@x8.float())*(dps*xs)).to(torch.bfloat16),
            ((p8.float().T@g8.float())*(ps*gs)).to(torch.bfloat16)]
    evidence=[]
    for name,a,b in zip(['output','dx','dup','ddown'],[out,*actual],[expected,*oracle]):
        rms=((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
        if not a.isfinite().all() or rms>.005:raise AssertionError((name,rms))
        evidence.append(dict(tensor=name,relative_rms=rms))
    torch.testing.assert_close(pa,post.float().abs().amax(),rtol=0,atol=0)
    torch.testing.assert_close(dpa,dp.float().abs().amax(),rtol=.005,atol=1e-6)
    print(json.dumps(dict(passed=True,shape=[n,d,h],checks=evidence,
                         original_autograd=True,training_gemms='native MI300 FP8 only'),indent=2))

if __name__=='__main__':main()
