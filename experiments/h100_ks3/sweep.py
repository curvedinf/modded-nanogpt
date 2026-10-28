"""Independent GPU micro ablations, with candidate-level numerical gates."""
import argparse,sys,json,time,statistics,datetime
from pathlib import Path
import torch,triton as tr
import triton.testing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from track_1_short.perf.kernels import moe_mlp as moe
from wr_reproduction import ks3_moe as core
from wr_reproduction.bf16_tiles import select_projection_tiles
from experiments.h100_ks3.h100_kernels import gemm,dw_split,dw_reduce

def error(a,b):
    if not bool(a.isfinite().all()):return 1e9
    return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()

def bench(fn):
    return statistics.median(tr.testing.do_bench_cudagraph(fn,rep=15) for _ in range(3))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--prepared',action='store_true');a=ap.parse_args()
    torch.cuda.set_device(a.gpu);torch.set_num_threads(1);torch.manual_seed(320)
    out=Path('research_results/wr_reproduction/h100-ks3/micro')/(a.mode+('-prepared' if a.prepared else '')+'.json')
    configs=[(bm,bn,bk,w,s) for bm,bn,w in [(32,64,4),(32,128,4),(64,64,4),(64,128,4),(64,256,8),(128,64,4),(128,128,8)] for bk,s in [(32,3),(64,3),(64,4),(128,3)]]
    if a.prepared:
        configs=[(bm,bn,64,nw,st) for bm,bn in [(32,128),(64,64),(64,128),(64,256),(128,64),(128,128),(128,256)] for nw in [4,8] for st in [3,4]]
    records=[]
    for tokens in (16384,49152):
        expert=torch.randint(8,(tokens,),device='cuda');dest,src,offs,tiles,_=moe.moe_dispatch(expert,8);m=src.numel()
        n,k=(1024,768) if a.mode in ('up','v','fusedup','fusedvjp') else (768,1024)
        weight=torch.randn(8,n,k,device='cuda',dtype=torch.bfloat16)*.02
        w=weight.transpose(1,2) if a.mode in ('up','down','fusedup','fusedout') else torch.randn(8,k,n,device='cuda',dtype=torch.bfloat16)*.02
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        g=torch.randn(8,1024,8,device='cuda')*.01;g[...,6]=.75
        prepared=torch.empty(8,1024,9,device='cuda')
        coords=torch.randn(m,1024,device='cuda');gate=torch.rand(tokens,device='cuda')
        base=select_projection_tiles(m,n,k,w.stride(2)==1)[:5]
        def plain(cfg):
            bm,bn,bk,nw,st=cfg;y=torch.empty(m,n,device='cuda',dtype=torch.bfloat16 if a.mode in ('down','dx','fusedout') else torch.float32)
            core._mm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](x,w,y,tiles,m,n,k,*w.stride(),BM=bm,BN=bn,BK=bk,num_warps=nw,num_stages=st)
            return y
        def finish_vjp(v,bm):
            dc=torch.empty_like(coords,dtype=torch.bfloat16);p=torch.empty(tr.cdiv(m,bm),1024,8,device='cuda');dg=torch.empty_like(g)
            core._vjp[(tr.cdiv(m,bm),32)](v,coords,g,dc,p,tiles,src,m,1024,tokens,BM=bm,RANK=1,num_warps=4)
            core._reduce[(8,1024)](p,dg,offs,1024,p.shape[0],tr.next_power_of_2(p.shape[0]),BM=bm,RANK=1,num_warps=4)
            return dc,dg
        def baseline():
            if a.mode=='dw':
                z=torch.empty(8,1024,768,device='cuda',dtype=torch.bfloat16)
                core._dw[(8,6,8)](da,db,z,offs,1024,768,False,BM=128,BN=128,BK=64,num_warps=8,num_stages=1)
                return (z,)
            y=plain(base)
            if a.mode=='fusedup':
                h=torch.empty_like(y,dtype=torch.bfloat16)
                core._geometry[(tr.cdiv(m,64),32)](y,g,h,tiles,m,1024,BM=64,RANK=1,num_warps=4)
                return y,h
            if a.mode=='fusedout':return y,y[dest.long()]*gate[:,None].to(torch.bfloat16)
            if a.mode=='fusedvjp':return finish_vjp(y,64)
            return (y,)
        da=torch.randn(m,1024,device='cuda',dtype=torch.bfloat16);db=torch.randn(m,768,device='cuda',dtype=torch.bfloat16)
        reference=baseline();base_ms=bench(baseline)
        candidates=configs if a.mode!='dw' else [(bm,bn,bk,nw,st,split) for bm,bn,nw in [(64,128,4),(128,128,8),(64,64,4)] for bk,st in [(32,3),(64,3),(128,3)] for split in [1,2,4]]
        table=[]
        for cfg in candidates:
            bm,bn,bk,nw,st=cfg[:5]
            def candidate():
                if a.mode=='dw':
                    split=cfg[5];p=torch.empty(8,split,1024,768,device='cuda');z=torch.empty(8,1024,768,device='cuda',dtype=torch.bfloat16)
                    dw_split[(tr.cdiv(1024,bm),tr.cdiv(768,bn),8*split)](da,db,p,offs,1024,768,split,bm,bn,bk,num_warps=nw,num_stages=st)
                    dw_reduce[(768,8)](p,z,split,1024*768);return (z,)
                if a.prepared:
                    from wr_reproduction.ks3_prepared_geometry import _prepare
                    _prepare[(32,)](g,prepared,8192)
                mode={'fusedup':1,'fusedout':2,'fusedvjp':3}.get(a.mode,0)
                y=torch.empty(m,n,device='cuda',dtype=torch.bfloat16 if a.mode in ('down','dx','fusedout','fusedvjp') else torch.float32)
                h=torch.empty(m,n,device='cuda',dtype=torch.bfloat16);sc=torch.empty(tokens,n,device='cuda',dtype=torch.bfloat16)
                part=torch.empty(tr.cdiv(m,bm),n,8,device='cuda') if mode==3 else None
                gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](x,w,y,tiles,m,n,k,*w.stride(),bm,bn,bk,mode,G=g,H=h,SRC=src,GATE=gate,SCATTER=sc,ROWS=tokens,C=coords,PART=part,PREPARED=prepared if a.prepared else None,num_warps=nw,num_stages=st)
                if mode==3:
                    dg=torch.empty_like(g);core._reduce[(8,n)](part,dg,offs,n,part.shape[0],tr.next_power_of_2(part.shape[0]),BM=bm,RANK=1,num_warps=4);return y,dg
                return (y,h) if mode==1 else ((y,sc) if mode==2 else (y,))
            try:
                actual=candidate();errs=[error(v,r) for v,r in zip(actual,reference)]
                if max(errs)>.003:table.append({'config':cfg,'error':errs,'passed':False});continue
                ms=bench(candidate);table.append({'config':cfg,'relative_rms':errs,'ms':ms,'passed':True})
            except Exception as exc:table.append({'config':cfg,'passed':False,'exception':str(exc)[:400]})
        best=min((r for r in table if r['passed']),key=lambda r:r['ms'],default=None)
        record={'tokens':tokens,'padded_rows':m,'baseline_config':base,'baseline_ms':base_ms,'best':best,'candidates':table};records.append(record)
        out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':a.mode,'gpu':a.gpu,'cases':records},indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='candidates'}),flush=True)
if __name__=='__main__':main()
