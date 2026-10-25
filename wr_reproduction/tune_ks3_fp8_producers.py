"""Check FP8 producer tiling without changing GS32 or geometry arithmetic."""
import json,torch,triton as tr
from pathlib import Path
from wr_reproduction.ks3_moe import _pack,_geometry

def main():
    torch.manual_seed(169);torch.set_num_threads(1);results=[]
    for m in (16384,49152):
        device='cuda';d,n,e=768,1024,8;dt=torch.float8_e4m3fn
        x=torch.randn(m,d,device=device,dtype=torch.bfloat16);src=torch.arange(m,device=device,dtype=torch.int32)
        tiles=torch.arange(e,device=device,dtype=torch.int32).repeat_interleave(m//e//128)
        c=torch.randn(m,n,device=device);g=torch.randn(e,n,8,device=device)*.01;g[...,6]=.75
        scale=torch.tensor([.01],device=device);amax=torch.zeros(1,device=device)
        for kind,cols in [('pack',d),('geometry',n)]:
            q=torch.empty(m,cols,device=device,dtype=dt);qt=torch.empty(cols,m,device=device,dtype=dt)
            def launch(rows,warps,partial):
                if kind=='pack':
                    _pack[(tr.cdiv(m,rows),tr.cdiv(cols,32))](x,q,m,cols,*x.stride(),SCALE=scale,OT=qt,SRC=src,INPUT_ROWS=m,FP8=True,ROWS_PER_TILE=rows,num_warps=warps)
                else:
                    _geometry[(tr.cdiv(m,rows),tr.cdiv(cols,32))](c,g,q,tiles,m,cols,HT=qt,SCALE=scale,AMAX=partial,FP8=True,PARTIAL_AMAX=True,BM=rows,num_warps=warps)
                    torch.maximum(amax,partial.amax(),out=amax)
            default=8 if kind=='pack' else 16
            part=torch.empty(tr.cdiv(m,default)*tr.cdiv(cols,32),device=device)
            launch(default,4,part);ref=q.clone();ref_t=qt.clone();records=[]
            for rows in (8,16,32,64,128):
                for warps in (4,8):
                    part=torch.empty(tr.cdiv(m,rows)*tr.cdiv(cols,32),device=device)
                    try:
                        launch(rows,warps,part)
                        assert torch.equal(q.view(torch.uint8),ref.view(torch.uint8))
                        assert torch.equal(qt.view(torch.uint8),ref_t.view(torch.uint8))
                        ms=tr.testing.do_bench_cudagraph(lambda:launch(rows,warps,part),rep=60)
                        records.append(dict(rows=rows,warps=warps,ms=ms))
                    except Exception as ex:records.append(dict(rows=rows,warps=warps,error=str(ex)[:160]))
            best=min((r for r in records if 'ms' in r),key=lambda x:x['ms'])
            result=dict(kind=kind,m=m,best=best,variants=records);results.append(result)
            print(json.dumps(result),flush=True)
    Path('research_results/wr_reproduction/ks3_fp8_producer_sweep.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
