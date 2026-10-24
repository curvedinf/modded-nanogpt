"""Equal-shaped attention batching must retain canonical evaluation arithmetic."""
import unittest
import torch
from control393.eval_attention import bucketed_attention
from mi300.portable import varlen_attention


class EvaluationTests(unittest.TestCase):
    def test_matches_independent_causal_document_window(self):
        torch.manual_seed(393)
        n,h,qd,vd=273,2,8,16
        q=torch.randn(n,h,qd,dtype=torch.float64)
        k=torch.randn_like(q);v=torch.randn(n,h,vd,dtype=torch.float64)
        boundaries=torch.tensor([0,13,n,n],dtype=torch.int32)
        out=bucketed_attention(q,k,v,boundaries,37,.13,group_size=2)
        pos=torch.arange(n);docs=pos>=13
        mask=(pos[:,None]>=pos[None,:])&(pos[:,None]-pos[None,:]<=37)&(docs[:,None]==docs[None,:])
        logits=torch.einsum('thd,shd->hts',q,k)*.13
        want=torch.einsum('hts,shd->thd',logits.masked_fill(~mask,float('-inf')).softmax(-1),v)
        torch.testing.assert_close(out,want,atol=1e-12,rtol=1e-12)


@torch.no_grad()
def run_device_checks(device):
    checks=[]
    torch.manual_seed(394)
    for dtype in (torch.float32,torch.bfloat16):
        for n,qd,vd,window in ((529,64,128,0),(529,128,64,384),
                               (32545,64,128,768),(32545,128,64,2560)):
            q=torch.randn(3,n,qd,device=device,dtype=dtype).transpose(0,1)
            k=torch.randn_like(q)
            v=torch.randn(3,n,vd,device=device,dtype=dtype).transpose(0,1)
            cu=torch.tensor([0,17,289,n,n],device=device,dtype=torch.int32)
            expected=varlen_attention(q,k,v,cu,window,.13)
            actual=bucketed_attention(q,k,v,cu,window,.13)
            if dtype==torch.bfloat16:
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            else:
                torch.testing.assert_close(actual,expected,atol=4e-6,rtol=4e-5)
            checks.append(dict(dtype=str(dtype),tokens=n,qdim=qd,vdim=vd,window=window,
                               bitwise_equal=bool(torch.equal(actual,expected)),
                               max_absolute=float((actual-expected).abs().max())))
    return dict(passed=True,checks=checks)


if __name__=='__main__':unittest.main()
