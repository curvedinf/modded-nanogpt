"""Eight-rank bitwise and fixed-count timing checks for one-collective FP8 refresh."""
import datetime,json,os,statistics
from pathlib import Path
import torch
import torch.distributed as dist
from experiments.h100_ks3.fp8_kernels import publish,layouts


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl');assert dist.get_world_size()==8
    torch.manual_seed(421+rank)
    n,d=1024,768;lo,hi=rank*10,(rank+1)*10
    bank=(torch.randn(80,n,d,device='cuda')*.03).bfloat16()
    scale=bank.float().abs().amax((1,2))/448
    partial=torch.empty(80,n*d//4096,device='cuda')
    q=torch.empty(80,n*d,device='cuda',dtype=torch.float8_e4m3fn)
    packed=torch.empty(80,n*d+64,device='cuda',dtype=q.dtype)
    out_scale=torch.empty_like(scale)
    def outputs():return [torch.empty(5,8,a,b,device='cuda',dtype=q.dtype) for a,b in [(n,d),(d,n),(d,n),(n,d)]]
    baseline_outputs=outputs();packed_outputs=outputs()
    def original():
        publish[(n*d//4096,10)](bank,q,scale,partial,lo,n,num_warps=4)
        work=dist.all_gather_into_tensor(q.view(torch.uint8),q[lo:hi].view(torch.uint8),async_op=True)
        sw=dist.all_gather_into_tensor(scale,scale[lo:hi],async_op=True)
        work.wait();sw.wait()
        layouts[(n*d//1024,80)](q,*baseline_outputs,n,num_warps=4)
    def combined():
        publish[(n*d//4096,10)](bank,packed,scale,partial,lo,n,PACK_SCALE=True,num_warps=4)
        dist.all_gather_into_tensor(packed.view(torch.uint8),packed[lo:hi].view(torch.uint8))
        layouts[(n*d//1024,80)](packed,*packed_outputs,n,PACK_SCALE=True,SCALE_OUT=out_scale,num_warps=4)
    original();combined()
    assert torch.equal(q.view(torch.uint8),packed[:,:n*d].view(torch.uint8))
    assert torch.equal(scale,out_scale)
    assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(baseline_outputs,packed_outputs))
    def measure(fn):
        for _ in range(3):fn()
        dist.barrier();torch.cuda.synchronize()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):fn()
        end.record();end.synchronize()
        value=torch.tensor(start.elapsed_time(end)/30,device='cuda');dist.all_reduce(value,op=dist.ReduceOp.MAX)
        return float(value)
    pairs=[[measure(original),measure(combined)] for _ in range(3)]
    if rank==0:
        result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ranks=8,bitwise_weights_scales_and_all_dual_layouts=True,
                    payload_ratio=(n*d+64)/(n*d),paired_ms=pairs,baseline_median_ms=statistics.median(v[0] for v in pairs),packed_median_ms=statistics.median(v[1] for v in pairs))
        Path('research_results/wr_reproduction/h100-research/fp8-packed-transport.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
