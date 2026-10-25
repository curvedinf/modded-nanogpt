"""Original sparse table updates over eight serialized logical ranks.

The complete table remains allocated. The only changed execution detail is that
rank contributions arrive serially, before one original optimizer event. Each
rank's FP16 cycle accumulator and owner-first merge order remain separate.
"""
import torch

class VirtualTables:
    def __init__(self,model,optimizer,total_steps,version):
        from track_1_short import ngram_table as source
        self.source=source;self.model=model;self.optimizer=optimizer;self.version=version
        v,d=source.NGRAM_VOCAB_SIZE,source.NGRAM_DIM;dev=model.ngram_cache.device
        self.rows_per_owner=v//8;self.event=0;self.event_policy=source.is_update_step
        self.table=torch.zeros(v,d,device=dev,dtype=torch.bfloat16)
        self.moment=torch.zeros(v,device=dev,dtype=torch.float32)
        self.last=torch.zeros(v,device=dev,dtype=torch.int32)
        self.history=torch.zeros(total_steps+1,device=dev,dtype=torch.float32)
        self.entry=torch.empty(v,device=dev,dtype=torch.int32) if version=='record360' else None
        self.pending=[[] for _ in range(8)]
        self.value_grads=[torch.zeros_like(model.value_embeds,dtype=torch.float16) for _ in range(8)]

    @torch.no_grad()
    def reset(self):
        self.table.zero_();self.moment.zero_();self.last.zero_();self.history.zero_();self.event=0
        for pending in self.pending:pending.clear()
        for gradient in self.value_grads:gradient.zero_()

    @torch.no_grad()
    def load(self,ids):
        unique,slots=torch.unique(ids,sorted=True,return_inverse=True)
        if unique.numel()>self.model.ngram_cache.shape[0]:raise RuntimeError('Original cache capacity exceeded')
        torch.index_select(self.table,0,unique.long(),out=self.model.ngram_cache[:unique.numel()])
        return slots.int()

    def hold(self,rank,ids,gradient):
        self.pending[rank].append((ids,gradient.detach()))

    def optimizer_event(self,step,lr,weight_decay,eps):
        if not self.event_policy(step):return None
        return Event(self,step,lr,weight_decay,eps)

class Event:
    def __init__(self,tables,step,lr,wd,eps):
        self.t=tables;self.step=step;self.lr=lr;self.wd=wd;self.eps=eps

    @torch.no_grad()
    def launch_ngram(self):
        from track_1_short.perf.kernels.row_scatter import scatter_add_rows
        self.payload=[]
        for entries in self.t.pending:
            if not entries:raise RuntimeError('Every logical rank must contribute before the sparse event')
            ids=torch.unique(torch.cat([i for i,g in entries]),sorted=True)
            values=torch.zeros(ids.numel(),self.t.source.NGRAM_DIM,device=ids.device,dtype=torch.float16)
            for rows,grad in entries:scatter_add_rows(grad,torch.searchsorted(ids,rows).int(),values)
            self.payload.append((ids,values))
            entries.clear()

    def launch(self):self.launch_ngram()  # record360 combined communication hook

    def launch_value_embeds(self):pass

    @torch.no_grad()
    def update(self):
        from track_1_short.perf.kernels import ngram_adam as kernel
        from track_1_short.perf.kernels.row_scatter import scatter_add_rows
        t=self.t;t.event+=1;event=t.event
        beta,wd_mul=t.source.adam_beta2_and_wd_mul(self.step)
        t.history[event].fill_(beta)
        lr=self.lr*t.source.NGRAM_LR_MUL
        kwargs=dict(beta2=beta,eps=self.eps,step_size=lr*(1-beta**event)**.5,
            decay=lr*lr*self.wd*wd_mul,grad_mul=1/8,sq_mul=(1-beta)/64)
        for owner in range(8):
            lo=owner*t.rows_per_owner;hi=lo+t.rows_per_owner
            pieces=[]
            for rank in [owner,*[r for r in range(8) if r!=owner]]:
                ids,values=self.payload[rank]
                mask=(ids>=lo)&(ids<hi)
                pieces.append(((ids[mask]-lo).int().contiguous(),values[mask].contiguous()))
            own_rows,own=pieces[0]
            recv_rows=torch.cat([p[0] for p in pieces[1:]])
            recv=torch.cat([p[1] for p in pieces[1:]])
            state=(t.table[lo:hi],t.moment[lo:hi],t.last[lo:hi],t.history)
            if t.version=='pr393':
                kernel.merge_adam_rows_(*state,own,own_rows,recv,recv_rows,event,**kwargs)
            else:
                entries=torch.cat([own_rows,recv_rows]);mapping=t.entry[lo:hi]
                kernel.claim_rows(entries,mapping)
                merged=torch.zeros(entries.numel(),t.source.NGRAM_DIM,device=t.table.device,dtype=torch.bfloat16)
                scatter_add_rows(own,own_rows,merged,row_map=mapping)
                scatter_add_rows(recv,recv_rows,merged,row_map=mapping)
                kernel.adam_rows_(*state,merged,entries,mapping,event,**kwargs)
        # The original owner accumulates value-embedding payloads in FP16, then /8.
        param=t.model.value_embeds
        rows=param.shape[0]//8
        combined=torch.empty_like(param,dtype=torch.float16)
        for owner in range(8):
            lo,hi=owner*rows,(owner+1)*rows
            target=combined[lo:hi];target.copy_(t.value_grads[owner][lo:hi])
            for rank in range(8):
                if rank!=owner:target.add_(t.value_grads[rank][lo:hi])
        combined.mul_(1/8)
        t.optimizer.adam_update_shard(param,combined)
        for g in t.value_grads:g.zero_()
        self.payload=None

    def finish(self):pass
