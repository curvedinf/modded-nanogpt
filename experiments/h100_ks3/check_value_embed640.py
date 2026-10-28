"""Check original value-gradient atomics with 128-channel segments."""
import datetime,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from track_1_short.perf.kernels import value_embed as ve


torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(443)
rows=[]
for d in (512,640,768):
    for repeated in (False,True):
        t=257;v=1024;planes=4
        ids=torch.arange(t,device='cuda',dtype=torch.int32)
        if repeated:ids=ids%23
        grads=[torch.randn(t,d,device='cuda',dtype=torch.bfloat16)*.001 for _ in range(planes)]
        out=torch.zeros(planes*v,d,device='cuda',dtype=torch.float16)
        ve.SEGMENT=128
        for _ in range(2):ve.value_embed_grad_accumulate(ids,grads,out)
        oracle=torch.zeros(planes,v,d,device='cuda')
        for p in range(planes):oracle[p].index_add_(0,ids.long(),grads[p].float()*2)
        delta=(out.float()-oracle.view_as(out)).square().mean().sqrt()
        relative=(delta/oracle.square().mean().sqrt()).item()
        assert relative<.003,(d,repeated,relative)
        row=dict(dim=d,repeated=repeated,relative_rms=relative)
        if d%256==0:
            ref=torch.zeros_like(out);ve.SEGMENT=256
            for _ in range(2):ve.value_embed_grad_accumulate(ids,grads,ref)
            error=((out.float()-ref.float()).square().mean()/ref.float().square().mean()).sqrt().item()
            assert error<.004,(d,repeated,error)
            row['relative_rms_vs_original']=error
        rows.append(row)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,cases=rows)
print(json.dumps(result,indent=2))
Path('research_results/wr_reproduction/h100-research/value-embed640-check.json').write_text(json.dumps(result,indent=2)+'\n')
