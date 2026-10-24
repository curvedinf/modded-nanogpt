"""Numerical routed KS3 checks, invoked only inside registered GPU diagnostics."""
import torch
import statistics
import time
from torch import nn
import torch.nn.functional as F
from control393.ks3_model import MoEKS3GPT
from ks3_research.vendor.ks3_int8 import ks3_int8
from ks3_research.backward import ks3_blas_fused_unpack


def check_partition(device, tokens, dim, charts, dtype, all_one=False,dynamic_rows=False,bf16_products=False,expert_unbind=False):
    torch.manual_seed(393)
    # Exercise the actual model method with small independent expert tensors.
    model = MoEKS3GPT.__new__(MoEKS3GPT)
    nn.Module.__init__(model)
    model.ks3_dynamic_rows=dynamic_rows
    model.ks3_bf16_products=bf16_products
    model.ks3_expert_unbind=expert_unbind
    model.moe_bank = nn.Parameter(torch.randn(5,2,8,charts,dim,device=device)*.04)
    model.moe_router = nn.Parameter(torch.zeros(5,8,dim,device=device))
    with torch.no_grad():
        model.moe_router[:, :, :8].copy_(torch.eye(8,device=device))
    model.ks3_coefficients = nn.Parameter(torch.randn(5,8,charts,1,6,device=device)*.01)
    model.ks3_scale = nn.Parameter(torch.full((5,8,charts),.75,device=device))
    model.ks3_shift = nn.Parameter(torch.zeros(5,8,charts,device=device))
    model.ks3_tanh_lut = (torch.tanh(torch.arange(-128,128,device=device).float()*(3.2/127))*127).round().clamp(-127,127).to(torch.int8)
    model._expert_views = [(*w.unbind(0),r) for w,r in zip(model.moe_bank,model.moe_router)]
    model._moe_aux = []
    x = torch.randn(tokens,dim,device=device,dtype=dtype)*.15
    choices = torch.zeros(tokens,device=device,dtype=torch.long) if all_one else torch.arange(tokens,device=device)%8
    x[:, :8] = 0
    x[torch.arange(tokens,device=device), choices] = 1
    x.requires_grad_()
    actual = model._mlp(x,None,None,slot=1)
    probs = F.softmax(F.linear(x,model.moe_router[0].to(dtype)).float(),-1)
    gate, selected = probs.max(-1)
    torch.testing.assert_close(selected,choices)
    expected = torch.zeros_like(x)
    primitive = ks3_blas_fused_unpack if dynamic_rows else ks3_int8
    for e in range(8):
        indices = (selected==e).nonzero().flatten()
        y = primitive(x.index_select(0,indices),model.moe_bank[0,0,e],model.moe_bank[0,1,e].T,
                    model.ks3_coefficients[0,e],model.ks3_scale[0,e],model.ks3_shift[0,e],
                    subspace_rank=1,nonlinear='fp16',tanh_lut=model.ks3_tanh_lut)
        expected = expected.index_copy(0,indices,y)
    expected = expected*gate[:,None].to(dtype)
    aux = 8*(F.one_hot(selected,8).float().mean(0)*probs.mean(0)).sum()
    torch.testing.assert_close(model._moe_aux[0],aux)
    upstream = torch.randn_like(x)/x.numel()**.5
    parameters = [x,model.moe_bank,model.moe_router,model.ks3_coefficients,model.ks3_scale,model.ks3_shift]
    actual_grads = torch.autograd.grad((actual*upstream).sum()+.01*model._moe_aux[0],parameters)
    expected_grads = torch.autograd.grad((expected*upstream).sum()+.01*aux,parameters)
    errors=[]
    for name,a,b in zip(['forward','input','bank','router','coefficients','scale','shift'],
                        [actual,*actual_grads],[expected,*expected_grads]):
        a,b=a.detach(),b.detach()
        if dynamic_rows and not bf16_products:
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert torch.isfinite(a).all(),name
        relative=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        absolute=float((a.float()-b.float()).abs().max())
        assert relative < (.02 if bf16_products else .006) or absolute < 2e-6,(name,relative,absolute)
        errors.append(dict(name=name,relative_rms=relative,max_absolute=absolute))
    counts=torch.bincount(selected,minlength=8)
    empty=(counts==0).nonzero().flatten()
    if len(empty):
        for grad in [actual_grads[1][0,:,empty],actual_grads[3][0,empty],
                     actual_grads[4][0,empty],actual_grads[5][0,empty]]:
            assert torch.count_nonzero(grad)==0
    timings=None
    if expert_unbind:
        model.ks3_expert_unbind=False
        model._expert_views=[(*w.unbind(0),r) for w,r in zip(model.moe_bank,model.moe_router)]
        model._moe_aux=[]
        baseline=model._mlp(x,None,None,slot=1)
        baseline_grads=torch.autograd.grad((baseline*upstream).sum()+.01*model._moe_aux[0],parameters)
        for a,b in zip([actual,*actual_grads],[baseline,*baseline_grads]):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        if tokens==4097:
            samples={False:[],True:[]}
            for repeat in range(9):
                for enabled in ((False,True) if repeat%2==0 else (True,False)):
                    model.ks3_expert_unbind=enabled
                    model._expert_views=[(*w.unbind(0),r) for w,r in zip(model.moe_bank,model.moe_router)]
                    model._moe_aux=[]
                    torch.cuda.synchronize(device)
                    begun=time.perf_counter()
                    y=model._mlp(x,None,None,slot=1)
                    grads=torch.autograd.grad((y*upstream).sum()+.01*model._moe_aux[0],parameters)
                    torch.cuda.synchronize(device)
                    duration=time.perf_counter()-begun
                    del y,grads
                    if repeat>=3:samples[enabled].append(duration)
            timings={str(k):dict(median_seconds=statistics.median(v),samples=v) for k,v in samples.items()}
    return dict(tokens=tokens,dim=dim,charts=charts,dtype=str(dtype),counts=counts.tolist(),errors=errors,
                unbind_bitwise_parity=expert_unbind,unbind_forward_backward=timings)


def run_device_checks(device,dynamic_rows=False,bf16_products=False,charts=1024,expert_unbind=False):
    checks=[check_partition(device,n,d,c,dtype,one,dynamic_rows,bf16_products,expert_unbind) for n,d,c,dtype,one in (
        (1,65,33,torch.float32,True), (7,65,33,torch.bfloat16,True),
        (17,65,33,torch.float32,False), (4097,768,charts,torch.bfloat16,False))]
    return dict(passed=True,checks=checks,dynamic_rows=dynamic_rows,bf16_products=bf16_products,
                bf16_relative_rms_limit=.02 if bf16_products else None)
