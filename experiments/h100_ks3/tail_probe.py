"""Full-validation averaging diagnostics from one training trajectory."""
import json
import torch
import torch.distributed as dist

LABELS=('vo_bank','mlp_bank','moe_bank')


def install():
    from track_1_short.tail_average import TailAverages
    original=TailAverages.ship
    def ship(self):
        self.ks3_probe_norms={label:self._param(label).data.float().norm() for label in LABELS}
        original(self)
    TailAverages.ship=ship


@torch.no_grad()
def reconstruct(tail,richardson):
    from track_1_short.tail_average import BANK_BLEND
    by_tag={a.tag:a for a in tail.accumulators}
    for label in LABELS:
        shard=tail._own_shard(label)
        short=by_tag['tail-avg'].bufs[label].to(shard.dtype).float()
        long=by_tag['bank-blend'].bufs[label]
        shard.copy_(torch.lerp(short,long,BANK_BLEND-richardson).to(shard.dtype))
    for label in LABELS:
        p=tail._param(label);cfg=tail.optimizer.param_cfgs[p]
        dist.all_gather_into_tensor(p.data.view(cfg.reshape),tail._own_shard(label))
        after=p.data.float().norm()
        ratio=torch.where(after>0,tail.ks3_probe_norms[label]/after,torch.ones_like(after)).item()
        p.data.mul_(ratio)
    geometry=tail.ks3_geometry_tail
    geometry.parameter.copy_(torch.lerp(geometry.avg,geometry.bank,BANK_BLEND-richardson))


@torch.no_grad()
def run(model,raw,manager,tail,ngram_table,batches,pulls,print0):
    from track_1_short.tail_average import RICHARDSON_W
    saved={label:tail._param(label).detach().clone() for label in LABELS}
    geometry=raw.ks3_geometry.detach().clone();diag=raw.cplm_diag.clone()
    reconstruct(tail,RICHARDSON_W)
    # Prove that the diagnostic reconstructs the actual final weights before
    # changing the averaging coefficient. No approximate shipping shortcut.
    checks={label:torch.equal(tail._param(label),saved[label]) for label in LABELS}
    checks['geometry']=torch.equal(raw.ks3_geometry,geometry)
    ok=torch.tensor(int(all(checks.values())),device=raw.device)
    dist.all_reduce(ok,op=dist.ReduceOp.MIN)
    assert bool(ok),checks
    print0('[tail-probe-reconstruction] '+json.dumps(checks),console=True)
    try:
        for richardson in dict.fromkeys((RICHARDSON_W,0.,.4,.55,.7)):
            reconstruct(tail,richardson)
            raw.prepare_ks3_eval();raw.cplm_diag.zero_()
            loss=torch.zeros((),device=raw.device)
            for batch,pull in zip(batches,pulls):
                ngram_table.land(pull)
                loss+=model(batch.inputs,batch.targets,batch.cum_seqlens,
                            ngram_table.slots(pull,batch.ngram_ids),manager.get_forward_args()).mean()
            loss/=len(batches);dist.all_reduce(loss,op=dist.ReduceOp.AVG)
            print0('[tail-average-probe] '+json.dumps(dict(richardson=richardson,loss=float(loss),
                tokens=10485760,official_timing_eligible=False,
                interpretation='Averaging diagnostic on one trained trajectory; requires a fresh frozen-recipe run')),console=True)
    finally:
        for label in LABELS:tail._param(label).data.copy_(saved[label])
        raw.ks3_geometry.copy_(geometry);raw.prepare_ks3_eval();raw.cplm_diag.copy_(diag)
