import torch
from wr_reproduction.moe_kernels import grouped_bf16
from wr_reproduction.optimizer_kernel import gemm_xb_add

def main():
    torch.manual_seed(393)
    a=torch.randn(257,768,device='cuda',dtype=torch.bfloat16)
    b=torch.randn(8,1024,768,device='cuda',dtype=torch.bfloat16).transpose(1,2)
    sizes=[0,1,31,64,0,100,60,1];offs=torch.tensor(sizes,device='cuda',dtype=torch.int32).cumsum(0).int()
    out=grouped_bf16(a,b,offs);ref=[];lo=0
    for e,n in enumerate(sizes):ref.append(a[lo:lo+n]@b[e]);lo+=n
    torch.testing.assert_close(out,torch.cat(ref),rtol=.008,atol=.125)
    x=torch.randn(2,1024,768,device='cuda',dtype=torch.bfloat16)
    gram=torch.randn(2,768,768,device='cuda',dtype=torch.bfloat16);gram=(gram+gram.transpose(1,2)).bfloat16()
    actual=torch.empty_like(x);gemm_xb_add(x,gram,actual,2.5)
    expected=(torch.bmm(x,gram).float()+x.float()*2.5).bfloat16()
    rms=((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()).item()
    assert rms<.005,rms
    print('PASS: validation grouped BF16 including empty experts; original rounded optimizer epilogue',rms)
if __name__=='__main__':main()
