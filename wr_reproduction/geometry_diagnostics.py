"""Terminal-only geometry evidence, outside all training/evaluation timers."""
import time
import torch


@torch.no_grad()
def collect(model, recipe, directory):
    from wr_reproduction.research import sha
    started=time.perf_counter()
    # A single readback after the terminal condition; never called per update.
    value=model.ks3_geometry.detach().float().cpu()
    path=directory/'terminal-geometry.pt'
    with path.open('xb') as out:
        torch.save(value,out)
    rank=recipe.get('rank',1)
    coeff=value[...,:6*rank].reshape(-1,rank,6)
    columns=[coeff[...,i].reshape(-1) for i in range(6)]
    columns.extend(value[...,6*rank+i].reshape(-1) for i in range(2))
    initial=torch.tensor(recipe['geometry']['coefficients']+[recipe['geometry']['raw_scale'],recipe['geometry']['raw_shift']],dtype=torch.float32)
    components={}
    for i,name in enumerate(['c0','c1','c2','c3','c4','c5','raw_scale','raw_shift']):
        column=columns[i];finite=torch.isfinite(column);x=column[finite].double()
        stats=dict(finite_count=int(finite.sum()),total_count=column.numel())
        if x.numel():
            stats.update(mean=x.mean().item(),rms=x.square().mean().sqrt().item(),
                         minimum=x.min().item(),maximum=x.max().item(),
                         abs_p95=x.abs().quantile(.95).item(),
                         delta_rms=(x-initial[i]).square().mean().sqrt().item())
        components[name]=stats
    dense={}
    if hasattr(model,'ks3_dense_geometry'):
        d=model.ks3_dense_geometry.detach().float().cpu()
        dp=directory/'terminal-dense-geometry.pt'
        with dp.open('xb') as out:torch.save(d,out)
        dense=dict(shape=list(d.shape),all_finite=bool(torch.isfinite(d).all()),
                   coefficient_rms=d[...,:6*rank].square().mean().sqrt().item(),
                   raw_scale_delta_rms=(d[...,6*rank]-initial[6]).square().mean().sqrt().item(),
                   artifact=str(dp),artifact_sha256=sha(dp))
    return dict(event='terminal_geometry',shape=list(value.shape),dense_feedforward=dense,
                all_finite=bool(torch.isfinite(value).all()) and dense.get('all_finite',True),components=components,
                artifact=str(path),artifact_sha256=sha(path),
                diagnostic_seconds=time.perf_counter()-started,
                outside_timed_training=True,used_for_screening=False)
