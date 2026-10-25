"""ROCm Triton port of the record's fused FP8 CE/CPLM equations.

Input logits retain E4M3FN rounding. Sigmoid is rounded to FP16 before both
loss and derivative, fixed-max LSE is retained, gradients are E5M2. In
particular the source's tail S_w includes all MTP weights even where the
future target is unavailable; do not replace this with ordinary autograd CE.
The original autograd/head/sampling machinery will call this kernel.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _ce(logits,targets,mtp,prefix_targets,prefix_weight,pc,asink,copy_col,
        losses,grad,dpc,das,N:tl.constexpr,V:tl.constexpr,P:tl.constexpr,
        A:tl.constexpr,B:tl.constexpr,C:tl.constexpr,GS:tl.constexpr,
        COPY:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK)
    raw=tl.load(logits+row*V+cols,cols<V,other=0.).to(tl.float32)
    sf=(.5+.5*libdevice.tanh((raw*(1./C)+B*(1./C))*.5)).to(tl.float16).to(tl.float32)
    z=A*sf
    es=tl.exp(z-A)
    lse=A+tl.log(tl.sum(tl.where(cols<V,es,0),0))
    pw=tl.load(prefix_weight);pt=tl.load(prefix_targets+row)
    loss=tl.full((),0,tl.float32);sw=tl.full((),0,tl.float32)
    correction=tl.full((BLOCK,),0,tl.float32)
    gt=tl.full((),-1.,tl.float32);gc=tl.full((),0.,tl.float32)
    if COPY:
        cc=tl.load(copy_col);t0=tl.load(targets+row);w0=tl.load(mtp)
        zt=tl.sum(tl.where(cols==t0,z,0),0);zc=tl.sum(tl.where(cols==cc,z,0),0)
        pl=tl.exp(zt-lse);alpha=tl.exp(zc-lse)
        pcv=tl.load(pc+row);av=tl.load(asink+row)
        reciprocal=1./tl.maximum(1.-alpha,1e-6)
        pmix=pl*(1.+alpha*av*reciprocal)+alpha*pcv
        inv=1./(pmix+1e-9)
        gt=-pl*(1.+alpha*av*reciprocal)*inv
        gc=-alpha*(pl*av*reciprocal*reciprocal+pcv)*inv
        nll=-tl.log(pmix+1e-9)
        sw+=w0*(-gt-gc-1.)
        correction+=tl.where(cols==cc,-w0*gc,0)
        tl.store(dpc+row,w0*(-alpha*inv))
        tl.store(das+row,w0*(-pl*alpha*reciprocal*inv))
    for k in tl.static_range(P):
        weight=tl.load(mtp+k);target=tl.load(targets+row+k,row+k<N,other=-1)
        sw+=weight
        zt=tl.sum(tl.where(cols==target,z,0),0)
        if COPY and k==0:
            loss+=tl.where((target>=0)&(target<V),weight*nll,0)
            correction+=tl.where(cols==target,-weight*gt,0)
        else:
            loss+=tl.where((target>=0)&(target<V),weight*(lse-zt),0)
            correction+=tl.where(cols==target,weight,0)
    sw+=tl.where(pt>=0,pw,0)
    zpt=tl.sum(tl.where(cols==pt,z,0),0)
    loss+=tl.where((pt>=0)&(pt<V),pw*(lse-zpt),0)
    correction+=tl.where(cols==pt,pw,0)
    g=((1./C*A)*(1./GS))*(sw*tl.exp(z-lse)-correction)*sf*(1.-sf)
    g=tl.maximum(tl.minimum(g,57344.),-57344.)
    tl.store(losses+row,loss)
    tl.store(grad+row*V+cols,g.to(tl.float8e5),cols<V)

def run(logits,targets,mtp,prefix_targets,prefix_weight,grad_scale,*,p_copy=None,a_sink=None,copy_col=None):
    if logits.dtype!=torch.float8_e4m3fn:raise TypeError('Record CE requires E4M3FN logits')
    n,v=logits.shape
    loss=torch.empty(n,device=logits.device,dtype=torch.float32)
    grad=torch.empty_like(logits,dtype=torch.float8_e5m2)
    dpc,das=torch.empty_like(loss),torch.empty_like(loss)
    copy=p_copy is not None
    if copy and (a_sink is None or copy_col is None):raise ValueError('Missing CPLM inputs')
    _ce[(n,)](logits,targets,mtp,prefix_targets,prefix_weight,
        p_copy if copy else loss,a_sink if copy else loss,copy_col if copy else targets,
        loss,grad,dpc,das,n,v,len(mtp),23.,5.,7.5,float(grad_scale),copy,
        triton.next_power_of_2(v),num_warps=16 if v>16384 else 8,enable_fp_fusion=False)
    return (loss,grad,dpc,das) if copy else (loss,grad)
