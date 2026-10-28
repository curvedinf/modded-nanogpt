import sys,json,statistics
from pathlib import Path
import torch,triton as tr,triton.testing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from track_1_short.perf.kernels import moe_mlp as moe
from wr_reproduction import ks3_moe as c
from wr_reproduction.ks3_prepared_geometry import _prepare,_geometry_prepared,_vjp_prepared

def error(a,b):return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
def main():
    torch.cuda.set_device(1);torch.set_num_threads(1);torch.manual_seed(213);cases=[]
    for tokens in [16384,49152]:
        dest,src,offs,tiles,_=moe.moe_dispatch(torch.randint(8,(tokens,),device='cuda'),8);m=src.numel();n=1024
        x=torch.randn(m,n,device='cuda');v=torch.randn_like(x);g=torch.randn(8,n,8,device='cuda')*.01;g[...,6]=.75
        prepared=torch.empty(8,n,9,device='cuda');_prepare[(32,)](g,prepared,8192)
        for kind in ['forward','backward']:
            def run(bm,warps,pre):
                out=torch.empty_like(x,dtype=torch.bfloat16)
                if kind=='forward':
                    kernel=_geometry_prepared if pre else c._geometry
                    kw={'PREPARED':prepared} if pre else {}
                    kernel[(tr.cdiv(m,bm),32)](x,g,out,tiles,m,n,BM=bm,RANK=1,num_warps=warps,**kw)
                    return (out,)
                p=torch.empty(tr.cdiv(m,bm),n,8,device='cuda');dg=torch.empty_like(g)
                kernel=_vjp_prepared if pre else c._vjp;kw={'PREPARED':prepared} if pre else {}
                kernel[(tr.cdiv(m,bm),32)](v,x,g,out,p,tiles,src,m,n,tokens,BM=bm,RANK=1,num_warps=warps,**kw)
                c._reduce[(8,n)](p,dg,offs,n,p.shape[0],tr.next_power_of_2(p.shape[0]),BM=bm,RANK=1,num_warps=4)
                return out,dg
            ref=run(64,4,False);rows=[]
            for pre in [False,True]:
                for bm in [16,32,64,128]:
                    for warps in [4,8]:
                        try:
                            vals=run(bm,warps,pre);errors=[error(a,b) for a,b in zip(vals,ref)];assert max(errors)<.0005
                            ms=statistics.median(tr.testing.do_bench_cudagraph(lambda:run(bm,warps,pre),rep=30) for _ in range(3))
                            rows.append({'bm':bm,'warps':warps,'prepared':pre,'ms':ms,'relative_rms':errors})
                        except Exception as e:rows.append({'bm':bm,'warps':warps,'prepared':pre,'error':str(e)[:300]})
            cases.append({'tokens':tokens,'kind':kind,'rows':rows});Path('research_results/wr_reproduction/h100-ks3/micro/geometry.json').write_text(json.dumps(cases,indent=2)+'\n');print(tokens,kind,sorted([r for r in rows if 'ms'in r],key=lambda r:r['ms'])[:3],flush=True)
if __name__=='__main__':main()
