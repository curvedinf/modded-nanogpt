"""Compare reconstruction with the untouched upstream shipment on CPU."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from track_1_short.tail_average import TailAverages,TailAccumulator,DECONTRACTION_LABELS,BANK_BLEND
from experiments.h100_ks3.geometry_tail import GeometryTail
from experiments.h100_ks3.tail_probe import LABELS,reconstruct

torch.manual_seed(467)
for richardson in (0.,.2007,.4,.7):
    params={l:torch.nn.Parameter(torch.randn(4,16,dtype=torch.bfloat16)) for l in (*DECONTRACTION_LABELS,'value_embeds')}
    tail=TailAverages.__new__(TailAverages);tail.rank=0
    tail.optimizer=SimpleNamespace(_param_by_label=params,param_cfgs={p:SimpleNamespace(optim='anvil',reshape=(4,16),chunk_size=4) for p in params.values()})
    tail.accumulators=[]
    groups=[('tail-ema',('lm_head','embed'),.65),('tail-avg',LABELS,None),
            ('value-embed-avg',('value_embeds',),None),('bank-blend',('qk_bank',*LABELS),BANK_BLEND-richardson)]
    for tag,labels,blend in groups:
        acc=TailAccumulator(tag,labels,250,lambda *_:.1,{l:(BANK_BLEND if l=='qk_bank' else blend) for l in labels})
        acc.seeded=set(labels);acc.bufs={l:torch.randn_like(params[l],dtype=torch.float32) for l in labels}
        tail.accumulators.append(acc)
    g=torch.nn.Parameter(torch.zeros(2,8))
    geometry=GeometryTail(g,1237,blend=BANK_BLEND-richardson)
    geometry.avg.normal_();geometry.bank.normal_();geometry.avg_seeded=geometry.bank_seeded=True
    tail.ks3_geometry_tail=geometry
    tail.ks3_probe_norms={l:params[l].detach().float().norm() for l in LABELS}
    # One-rank shipment is identity communication, retaining the original
    # upstream arithmetic and dtype conversions as the independent oracle.
    with patch('torch.distributed.all_gather_into_tensor',lambda out,own:None):
        TailAverages.ship(tail);geometry.ship()
        expected={l:params[l].detach().clone() for l in LABELS};eg=g.detach().clone()
        reconstruct(tail,richardson)
    assert all(torch.equal(params[l],expected[l]) for l in LABELS)
    assert torch.equal(g,eg)
print('Four averaging coefficients reconstructed bitwise against untouched upstream shipment')
