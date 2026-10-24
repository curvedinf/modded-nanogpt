"""CPLM slab compiler fusion: objective, all input gradients, and paired timing."""
import statistics
import time
import unittest
import torch
from control393.training_loss import slab_loss,compiled_slab_loss


def oracle(logits,ys,pys,p,a,mtp,prefix,cc):
    logp=torch.log_softmax(23*torch.sigmoid((logits.float()+5)/7.5),-1)
    pl=logp.gather(1,ys[:,0,None])[:,0].exp();alpha=logp[:,cc].exp()
    mixed=pl*(1+alpha*a/(1-alpha).clamp_min(1e-6))+alpha*p
    loss=-mtp[0]*torch.log(mixed+1e-9)
    for i in range(1,ys.shape[1]):
        y=ys[:,i];loss=loss-mtp[i]*logp.gather(1,y.clamp_min(0)[:,None])[:,0]*(y>=0)
    return loss-prefix[0]*logp.gather(1,pys.clamp_min(0)[:,None])[:,0]*(pys>=0)


class TrainingLossTests(unittest.TestCase):
    def test_sink_only_reduces_to_noncopy_language_model(self):
        torch.manual_seed(393)
        z=torch.randn(7,16,requires_grad=True);cc=9
        ys=torch.tensor([[0],[1],[2],[3],[4],[5],[6]])
        out=slab_loss(z,ys,torch.full((7,),-1),torch.zeros(7),torch.ones(7),
                      torch.ones(1),torch.zeros(1),torch.tensor(cc))
        scores=23*torch.sigmoid((z+5)/7.5)
        mask=torch.arange(16)==cc
        expected=-scores.masked_fill(mask,float('-inf')).log_softmax(-1).gather(1,ys)[:,0]
        torch.testing.assert_close(out,expected,rtol=1e-6,atol=1e-6)
        a=torch.autograd.grad(out.sum(),z)[0];b=torch.autograd.grad(expected.sum(),z)[0]
        torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)


def run_device_checks(device,rows=4096):
    torch.manual_seed(393129);checks=[]
    # Use production slab shapes so diagnostics do not consume graph variants
    # on artificial shapes. Test changing copy positions as device data.
    for vocab,depth in ((10240,3),(16384,2),(18432,1),(24576,1),(50304,1)):
        n=rows;cc=vocab-7
        z=(torch.randn(n,vocab,device=device,dtype=torch.bfloat16)*3).requires_grad_()
        p=torch.rand(n,device=device,requires_grad=True)
        a=torch.rand(n,device=device,requires_grad=True)
        with torch.no_grad():
            z[0].fill_(-1000.);z[0,cc]=1000.
            p[0]=0.;a[0]=1.
        ys=torch.randint(vocab,(n,depth),device=device);ys[0,0]=0
        if depth>1:
            ys[1,1]=ys[1,0];ys[-1,1:]=-1
        pys=torch.randint(vocab,(n,),device=device);pys[1]=ys[1,0];pys[-2:]=-1
        mtp=torch.tensor([1.,.5,.25][:depth],device=device)
        prefix=torch.tensor([.15 if depth>1 else 0.],device=device)
        column=torch.tensor(cc,device=device);upstream=torch.randn(n,device=device)/n
        for copy_index in (cc,cc-3):
            column.fill_(copy_index)
            expected=oracle(z,ys,pys,p,a,mtp,prefix,copy_index)
            actual=compiled_slab_loss(z,ys,pys,p,a,mtp,prefix,column)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-5)
            old=torch.autograd.grad(expected,(z,p,a),upstream)
            new=torch.autograd.grad(actual,(z,p,a),upstream)
            errors=[]
            for got,want in zip(new,old):
                difference=(got.float()-want.float())
                relative=float(difference.norm()/want.float().norm().clamp_min(1e-12))
                absolute=float(difference.abs().max())
                limit=.002 if got.dtype==torch.bfloat16 else 5e-5
                assert relative<=limit,(vocab,copy_index,got.dtype,relative,absolute)
                errors.append(dict(dtype=str(got.dtype),relative_rms=relative,max_absolute=absolute))
            checks.append(dict(vocab=vocab,mtp=depth,copy_index=copy_index,
                               forward_max_absolute=float((actual-expected).abs().max()),gradients=errors))
            del old,new,actual,expected
        if vocab==18432:
            # Include forward, backward, and launch overhead on the same tensors.
            benchmark_copy=int(column)
            funcs=dict(eager=lambda:oracle(z,ys,pys,p,a,mtp,prefix,benchmark_copy),
                       compiled=lambda:compiled_slab_loss(z,ys,pys,p,a,mtp,prefix,column))
            samples={name:[] for name in funcs}
            for repetition in range(7):
                for name in (('eager','compiled') if repetition%2==0 else ('compiled','eager')):
                    torch.cuda.synchronize();start=time.perf_counter()
                    value=funcs[name]();grads=torch.autograd.grad(value,(z,p,a),torch.ones_like(value))
                    torch.cuda.synchronize();duration=time.perf_counter()-start
                    if repetition>=2:samples[name].append(duration)
                    del value,grads
            medians={name:statistics.median(values) for name,values in samples.items()}
            benchmark=dict(rows=n,vocab=vocab,samples_seconds=samples,median_seconds=medians,
                           speedup=medians['eager']/medians['compiled'])
        del z,p,a,ys,pys,mtp,prefix,column,upstream
    return dict(passed=True,checks=checks,benchmark=benchmark)


if __name__=='__main__':unittest.main()
