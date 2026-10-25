"""Independent owner-order reference for serialized PR393 sparse updates."""
from types import SimpleNamespace
import torch
from wr_reproduction.backend import install
from wr_reproduction.virtual_tables import VirtualTables,Event

def main():
    install('pr393')
    from track_1_short import ngram_table as source
    torch.manual_seed(393);dev='cuda';rows,d=16,768
    t=VirtualTables.__new__(VirtualTables);t.source=source;t.version='pr393';t.rows_per_owner=2;t.event=0
    t.table=(torch.randn(rows,d,device=dev)*.01).bfloat16();initial=t.table.clone()
    t.moment=torch.zeros(rows,device=dev);t.last=torch.zeros(rows,device=dev,dtype=torch.int32)
    t.history=torch.zeros(8,device=dev);t.pending=[[] for _ in range(8)]
    t.model=SimpleNamespace(value_embeds=torch.nn.Parameter(torch.zeros(rows,d,device=dev,dtype=torch.bfloat16)))
    t.value_grads=[(torch.randint(-8,9,(rows,d),device=dev).float()/32).half() for _ in range(8)]
    saved_values=[g.clone() for g in t.value_grads];observed=[]
    t.optimizer=SimpleNamespace(adam_update_shard=lambda p,g:observed.append(g.clone()))
    for rank in range(8):
        for step in range(2):
            ids=torch.tensor([0,0,1,2,4,5,rank*2,rank*2+1],device=dev,dtype=torch.int32)
            g=(torch.randint(-8,9,(8,d),device=dev).float()/32).bfloat16()
            t.hold(rank,ids,g)
    event=Event(t,1,.008,.005,1e-10);event.launch_ngram()
    reference=torch.zeros(rows,d,device=dev,dtype=torch.bfloat16)
    for owner in range(8):
        for rank in [owner,*[r for r in range(8) if r!=owner]]:
            ids,vals=event.payload[rank]
            for i,row in enumerate(ids.tolist()):
                if owner*2<=row<(owner+1)*2:reference[row]=(reference[row]+vals[i].bfloat16()).bfloat16()
    beta,wd_mul=source.adam_beta2_and_wd_mul(1);lr=.008*source.NGRAM_LR_MUL
    wanted_moment=reference.float().square().mean(1)*(1-beta)/64
    update=(reference.float()/8)/(wanted_moment.sqrt()[:,None]+1e-10)*(lr*(1-beta)**.5)
    update+=torch.where(update*initial.float()>0,initial.float()*(lr*lr*.005*wd_mul),0.)
    wanted=(initial.float()-update).bfloat16()
    expected_values=torch.empty_like(saved_values[0])
    for owner in range(8):
        lo,hi=2*owner,2*owner+2
        expected_values[lo:hi]=saved_values[owner][lo:hi]
        for rank in range(8):
            if rank!=owner:expected_values[lo:hi]+=saved_values[rank][lo:hi]
    expected_values*=1/8
    event.update()
    torch.testing.assert_close(t.table,wanted,rtol=.008,atol=1e-5)
    torch.testing.assert_close(t.moment,wanted_moment,rtol=2e-6,atol=1e-9)
    torch.testing.assert_close(observed[0],expected_values,rtol=0,atol=0)
    assert t.event==1 and t.last.eq(1).all() and all(not entries for entries in t.pending)
    print('PASS: original owner-first BF16 merge, FP16 rank cycle accumulation, /8 average and sparse Adam event')
if __name__=='__main__':main()
