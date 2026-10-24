"""CPLM slab partition invariance, sampled head derivatives, and paired timing."""
import statistics
import time
import unittest
import torch
from torch.utils.checkpoint import checkpoint
from control393.loss import Head,TiledHead,sampled_rows
from control393.training_loss import compiled_slab_loss
from test_control393_training_loss import oracle


class SlabTests(unittest.TestCase):
    def test_sampled_rows_are_identical_across_slab_boundaries(self):
        for counter in (0,8,216):
            for offset in (0,8192,114688):
                whole=sampled_rows(8192,counter,offset,'cpu')+offset
                split=torch.cat([sampled_rows(4096,counter,offset+lo,'cpu')+offset+lo
                                 for lo in (0,4096)])
                torch.testing.assert_close(whole,split,rtol=0,atol=0)


def partitioned_loss(x,w,ys,pys,p,a,mtp,prefix,cc,sample,slab,compiled=False,tiled_head=False):
    column=torch.tensor(cc,device=x.device)
    head=TiledHead if tiled_head and slab>4096 else Head
    def one(h,weight,labels,prefixes,pc,asin,offset):
        logits=head.apply(h,weight,labels,prefixes,cc,sample,216,offset)
        if compiled:return compiled_slab_loss(logits,labels,prefixes,pc,asin,mtp,prefix,column)
        return oracle(logits,labels,prefixes,pc,asin,mtp,prefix,cc)
    return torch.cat([checkpoint(one,x[lo:lo+slab],w,ys[lo:lo+slab],pys[lo:lo+slab],
                                 p[lo:lo+slab],a[lo:lo+slab],8192+lo,use_reentrant=False)
                      for lo in range(0,x.shape[0],slab)])


def run_device_checks(device,slab=8192,tiled_head=False):
    assert slab==8192,'This ablation verifies4096 versus8192 slabs'
    torch.manual_seed(393159);checks=[];benchmark=None
    for vocab,depth,sample in ((10240,3,True),(18432,1,True),(50304,1,False)):
        n,dim=8192,768;cc=vocab-7
        x=torch.randn(n,dim,device=device,dtype=torch.bfloat16,requires_grad=True)
        w=(torch.randn(dim,vocab,device=device,dtype=torch.bfloat16)*.03).requires_grad_()
        p=torch.rand(n,device=device,requires_grad=True);a=torch.rand(n,device=device,requires_grad=True)
        ys=torch.randint(vocab,(n,depth),device=device)
        pys=torch.randint(vocab,(n,),device=device);pys[::7]=-1;pys[::11]=ys[::11,0]
        ys[0,0]=cc
        if depth>1:ys[1,1]=ys[1,0];ys[-1,1:]=-1
        mtp=torch.tensor([1.,.5,.25][:depth],device=device)
        prefix=torch.tensor([.15 if depth>1 else 0.],device=device)
        upstream=torch.rand(n,device=device)/n
        args=(x,w,ys,pys,p,a,mtp,prefix,cc,sample)
        original=partitioned_loss(*args,4096)
        candidate=partitioned_loss(*args,8192,tiled_head=tiled_head)
        torch.testing.assert_close(candidate,original,rtol=2e-6,atol=2e-5)
        old=torch.autograd.grad(original,(x,w,p,a),upstream)
        new=torch.autograd.grad(candidate,(x,w,p,a),upstream)
        errors=[]
        for label,got,want in zip(('hidden','head_weight','copy_probability','sink_probability'),new,old):
            difference=got.float()-want.float()
            relative=float(difference.norm()/want.float().norm().clamp_min(1e-12))
            absolute=float(difference.abs().max())
            limit=.01 if got.dtype==torch.bfloat16 else 5e-5
            assert relative<=limit,(label,vocab,relative,absolute)
            errors.append(dict(name=label,relative_rms=relative,max_absolute=absolute,limit=limit))
        checks.append(dict(vocab=vocab,mtp=depth,sampled_head=sample,
                           forward_max_absolute=float((candidate-original).abs().max()),gradients=errors))
        del old,new,original,candidate
        if vocab==18432:
            samples={4096:[],8192:[]}
            for repetition in range(7):
                for size in ((4096,8192) if repetition%2==0 else (8192,4096)):
                    torch.cuda.synchronize();start=time.perf_counter()
                    loss=partitioned_loss(*args,size,compiled=True,tiled_head=tiled_head)
                    grads=torch.autograd.grad(loss,(x,w,p,a),upstream)
                    torch.cuda.synchronize();duration=time.perf_counter()-start
                    if repetition>=2:samples[size].append(duration)
                    del loss,grads
            medians={size:statistics.median(values) for size,values in samples.items()}
            benchmark=dict(tokens=n,dim=dim,vocab=vocab,samples_seconds=samples,
                           median_seconds=medians,speedup=medians[4096]/medians[8192])
        del args,x,w,p,a,ys,pys,mtp,prefix,upstream
    return dict(passed=True,checks=checks,benchmark=benchmark)


if __name__=='__main__':unittest.main()
