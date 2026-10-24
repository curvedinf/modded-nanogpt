"""Replay unchanged ANVIL bank updates on persistent optimizer buffers.

Capture and warmup are timed construction overhead. Their mutations are undone
in place before training; every real replay reads the current gradient/scalars.
"""
import time
import torch


FIELDS=('grad','velocity','lane_energy','mantissa','p_slice','momentum',
        'eff_wd','fast_beta','fast_weight','eff_lr')


def fingerprint(bank):
    return (tuple((name,getattr(bank,name).data_ptr(),tuple(getattr(bank,name).shape),
                   getattr(bank,name).stride(),getattr(bank,name).dtype,getattr(bank,name).device)
                  for name in FIELDS),bank.beta2,bank.split_baddbmm,bank.red_dim)


def check_addresses(bank,expected,live):
    if fingerprint(bank)!=expected:
        raise RuntimeError(f'Captured optimizer buffers changed: {bank.label}')
    for name,value in live.items():
        original=getattr(bank,name)
        if (value.data_ptr(),value.shape,value.stride(),value.dtype)!=(
                original.data_ptr(),original.shape,original.stride(),original.dtype):
            raise RuntimeError(f'Captured optimizer live state changed: {bank.label}.{name}')


@torch.no_grad()
def restore(state,snapshot):
    for name,tensor in state.items():
        tensor.copy_(snapshot[name])


class BankGraphs:
    @torch.no_grad()
    def __init__(self,optimizer,update):
        begun=time.perf_counter()
        self.optimizer=optimizer
        self.graphs={}
        self.addresses={}
        self.replays=0
        stream=torch.cuda.Stream()
        pool=torch.cuda.graph_pool_handle()
        torch.cuda.synchronize()
        # Reuse graph workspace in the same order as real optimizer execution.
        for label in optimizer.work_order:
            if label not in optimizer.banks:
                continue
            bank=optimizer.banks[label]
            state=bank.mutated()
            snapshot={name:value.clone() for name,value in state.items()}
            self.addresses[label]=fingerprint(bank)
            stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        update(bank)
                stream.synchronize()
                restore(state,snapshot)
                stream.wait_stream(torch.cuda.current_stream())
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=pool,stream=stream):
                    update(bank)
                stream.synchronize()
                self.graphs[label]=graph
            finally:
                restore(state,snapshot)
                torch.cuda.synchronize()
            check_addresses(bank,self.addresses[label],optimizer.live_bank_state(label))
        self.capture_seconds=time.perf_counter()-begun

    def update(self,bank):
        check_addresses(bank,self.addresses[bank.label],self.optimizer.live_bank_state(bank.label))
        self.graphs[bank.label].replay()
        self.replays+=1

    def stats(self):
        return dict(banks=list(self.graphs),capture_seconds=self.capture_seconds,replays=self.replays,
                    fallbacks=0,state_restored_after_capture=True)
