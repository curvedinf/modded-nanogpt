import os,sys,json,statistics
from pathlib import Path
import torch,triton as tr,triton.testing,torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.port import publish
from experiments.h100_ks3.transport import encode,decode

def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1);dist.init_process_group('nccl');torch.manual_seed(819)
    bank=(torch.randn(80,1024,768,device='cuda')*.03).bfloat16();p=torch.empty_like(bank)
    q=torch.empty(80,1024*768,device='cuda',dtype=torch.int8);s=torch.empty(80,1024*768//32,device='cuda',dtype=torch.float16)
    up=torch.empty(5,8,1024,768,device='cuda',dtype=torch.bfloat16);down=torch.empty(5,8,768,1024,device='cuda',dtype=torch.bfloat16)
    lo,hi=rank*10,(rank+1)*10
    def original():
        publish(bank,p,lo,hi);dist.all_gather_into_tensor(p,p[lo:hi]);b=p.view(5,2,8,-1);up.copy_(b[:,0].view_as(up));down.copy_(b[:,1].view_as(down))
    original();expected=(up.clone(),down.clone());rows=[]
    def compressed(br):
        encode[(tr.cdiv(1024,br),10,32)](bank,q,s,lo,BR=br,num_warps=4)
        dist.all_gather_into_tensor(q,q[lo:hi]);dist.all_gather_into_tensor(s,s[lo:hi]);decode[(768,80)](q,s,up,down,num_warps=4)
    # Fixed-count collective timing: identical rank order, no adaptive benchmark loops.
    def measure(fn):
        for _ in range(3):fn()
        dist.barrier();torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):fn()
        end.record();end.synchronize()
        value=torch.tensor(start.elapsed_time(end)/30,device='cuda');dist.all_reduce(value,op=dist.ReduceOp.MAX)
        return value.item()
    for br in (8,16,32,64,128):
        compressed(br);assert torch.equal(up,expected[0]) and torch.equal(down,expected[1])
        pairs=[]
        for _ in range(3):pairs.append([measure(original),measure(lambda:compressed(br))])
        rows.append({'br':br,'all_values_bitwise':True,'paired_ms':pairs})
    if rank==0:
        out={'transport_bytes_ratio':17/32,'ranks':8,'cases':rows};Path('research_results/wr_reproduction/h100-ks3/micro/transport.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
    dist.destroy_process_group()
if __name__=='__main__':main()
