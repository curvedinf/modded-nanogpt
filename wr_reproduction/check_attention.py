import json,torch
from wr_reproduction.attention_kernel import flash_attn_varlen_func

def main():
    torch.manual_seed(393);torch.set_num_threads(8);results=[]
    for dq,dv,window in [(64,128,64),(128,64,192),(128,128,0)]:
        n,h=257,3;cu=torch.tensor([0,7,45,178,n,n],device='cuda',dtype=torch.int32)
        q,k=[torch.randn(n,h,dq,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(2)]
        v=torch.randn(n,h,dv,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        doc=torch.bucketize(torch.arange(n,device='cuda'),cu[1:],right=True)
        pos=torch.arange(n,device='cuda');mask=(pos[:,None]>=pos[None,:])&(pos[:,None]-pos[None,:]<=window)&(doc[:,None]==doc[None,:])
        scale=.12
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1),k.transpose(0,1),v.transpose(0,1),attn_mask=mask,scale=scale).transpose(0,1)
        got=flash_attn_varlen_func(q,k,v,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=133,max_seqlen_k=133,causal=True,softmax_scale=scale,window_size=(window,0))
        dy=torch.randn_like(got)
        actual=torch.autograd.grad(got,(q,k,v),dy);expected=torch.autograd.grad(ref,(q,k,v),dy)
        rms=[]
        for a,b in zip((got,*actual),(ref,*expected)):
            err=((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-6)).item()
            # Self-only attention has exactly zero Q/K gradient; enforce absolute error there.
            if window==0:assert (a.float()-b.float()).abs().max()<.02
            else:assert err<.02,err
            assert a.isfinite().all();rms.append(err)
        results.append(dict(qk=dq,v=dv,window=window,relative_rms=rms))
    print(json.dumps(dict(passed=True,cases=results),indent=2))
if __name__=='__main__':main()
