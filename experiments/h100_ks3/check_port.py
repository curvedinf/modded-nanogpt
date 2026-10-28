"""Distributed shard publication and independent KS3 core checks."""
import sys,os,json
from pathlib import Path
import torch,torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.port import publish,install
from wr_reproduction.ks3_moe import pack

def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
    torch.manual_seed(123)
    bank=(torch.randn(80,1024,768,device='cuda')*.03).bfloat16()
    expected=torch.empty_like(bank)
    for slot in range(80):
        w=bank[slot] if slot%16<8 else bank[slot].T
        expected[slot].view(-1).copy_(pack(w).flatten())
    p=torch.full_like(bank,float('nan'));lo=rank*10;hi=lo+10
    publish(bank,p,lo,hi)
    assert torch.equal(p[lo:hi],expected[lo:hi])
    dist.all_gather_into_tensor(p,p[lo:hi]);assert torch.equal(p,expected)
    # Simulate updated owner-only weights; all consumers must see fresh values.
    reference=bank.clone().add_(.013)
    bank[lo:hi].add_(.013);publish(bank,p,lo,hi)
    dist.all_gather_into_tensor(p,p[lo:hi])
    bank=reference
    for slot in range(80):
        w=bank[slot] if slot%16<8 else bank[slot].T
        assert torch.equal(p[slot].flatten(),pack(w).flatten())
    if rank==0:print(json.dumps({'distributed_publication':'passed','ranks':8,'all80_slots_bitwise':True,'owner_update_refresh':'passed'}),flush=True)
    dist.destroy_process_group()
if __name__=='__main__':main()
