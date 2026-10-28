"""Bitwise producer oracles and paired timings, including padded source rows."""
import datetime,json,os,statistics
from pathlib import Path
import torch,triton as tr,triton.testing
from experiments.h100_ks3 import fp8_kernels as old,input_layouts as new


def main():
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(421)
    cases=[]
    for rows in (16384,32768,49152):
        m=rows+896
        src=torch.cat((torch.randperm(rows,device='cuda',dtype=torch.int32),
                       torch.full((m-rows,),rows,device='cuda',dtype=torch.int32)))
        x=torch.randn(rows,768,device='cuda',dtype=torch.bfloat16)
        grad=torch.randn_like(x)*.01;y=torch.randn(m,768,device='cuda',dtype=x.dtype)*.1
        gate=torch.rand(rows,device='cuda')*.8+.2
        xs=torch.tensor([1/128],device='cuda');gs=torch.tensor([1/64],device='cuda')
        first=torch.empty(m,768,device='cuda',dtype=torch.float8_e4m3fn)
        second=torch.empty(768,m,device='cuda',dtype=first.dtype)
        output=torch.empty_like(first);dual=torch.empty_like(second)
        old_pack=lambda:old.pack_input[(m//128,24)](x,first,second,src,xs,m,rows,num_warps=4)
        new_pack=lambda:new.pack(x,output,dual,src,xs)
        old_pack();new_pack()
        assert torch.equal(first.view(torch.uint8),output.view(torch.uint8))
        assert torch.equal(second.view(torch.uint8),dual.view(torch.uint8))
        assert torch.count_nonzero(output[-896:].float())==0
        gg=torch.empty(m,768,device='cuda',dtype=torch.float8_e5m2);gt=torch.empty(768,m,device='cuda',dtype=gg.dtype)
        ng=torch.empty_like(gg);nt=torch.empty_like(gt)
        ds=torch.empty(m,device='cuda');nd=torch.empty_like(ds)
        old_grad=lambda:old.gather_gate_tiled[(m//64,)](grad,y,src,gate,gg,gt,ds,gs,rows,m,num_warps=4,enable_fp_fusion=False)
        new_grad=lambda:new.gather(grad,y,src,gate,ng,nt,nd,gs,rows,m)
        old_grad();new_grad()
        assert torch.equal(gg.view(torch.uint8),ng.view(torch.uint8))
        assert torch.equal(gt.view(torch.uint8),nt.view(torch.uint8))
        print(json.dumps(dict(rows=rows,gate_gradient_bitwise=torch.equal(ds,nd),
                              gate_gradient_max_abs=float((ds-nd).abs().max()),
                              gate_gradient_relative_rms=float((ds-nd).square().mean().sqrt()/ds.square().mean().sqrt()),
                              finite=bool(torch.isfinite(nd).all()))),flush=True)
        assert torch.equal(ds,nd)
        timing={}
        for name,fn in [('pointer_input',old_pack),('tma_input',new_pack),('pointer_gradient',old_grad),('tma_gradient',new_grad)]:
            samples=[tr.testing.do_bench_cudagraph(fn,rep=30) for _ in range(3)]
            timing[name]=dict(samples_ms=samples,median_ms=statistics.median(samples))
        result=dict(rows=rows,padded_rows=m,bitwise_equal=True,padding_zero=True,timing=timing)
        cases.append(result);print(json.dumps(result),flush=True)
    path=Path(os.environ.get('KS3_INPUT_CHECK_OUTPUT','research_results/wr_reproduction/h100-research/fp8-input-layouts.json'))
    path.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases),indent=2)+'\n')


if __name__=='__main__':main()
