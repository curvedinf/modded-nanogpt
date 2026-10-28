"""Off-clock inference-window diagnostics; never official training results."""
from dataclasses import replace
import json
import torch
import torch.distributed as dist


@torch.no_grad()
def run(model,manager,ngram_table,batches,pulls,windows,print0):
    """Hold weights, token stream, mask and copy band fixed; vary causal context."""
    original=manager.get_forward_args()
    saved_diag=model.cplm_diag.clone()
    for window in dict.fromkeys([original.ws_long//128,*windows]):
        cfg=replace(original,ws_long=window*128)
        model.cplm_diag.zero_()
        loss=torch.zeros((),device=model.device)
        for batch,pull in zip(batches,pulls):
            ngram_table.land(pull)
            loss+=model(batch.inputs,batch.targets,batch.cum_seqlens,
                        ngram_table.slots(pull,batch.ngram_ids),cfg).mean()
        loss/=len(batches)
        dist.all_reduce(loss,op=dist.ReduceOp.AVG)
        print0('[eval-window-probe] '+json.dumps(dict(long_window_blocks=window,
               loss=float(loss),tokens=10485760,official_timing_eligible=False,
               interpretation='Same frozen weights; exploratory inference architecture only')),console=True)
    model.cplm_diag.copy_(saved_diag)
