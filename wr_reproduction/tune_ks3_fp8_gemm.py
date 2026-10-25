"""Offline KS3-only tile sweep, validated against the existing FP8 kernels."""
import json,time,torch,triton as tr
from wr_reproduction.fused_fp8 import _gemm,_dw,run,grouped_dw

def mm(a,b,sa,sb,tiles,n,config):
    bm,bn,bk,warps,stages,mfma=config;m,k=a.shape
    out=torch.empty((m,n),device=a.device,dtype=torch.float32)
    _gemm[(tr.cdiv(m,bm),tr.cdiv(n,bn))](a.view(torch.uint8),b.view(torch.uint8),out,out,out,out,sa,
        sa,sb,sa,sa,sa,tiles,m,n,k,*a.stride(),*b.stride(),a.dtype==torch.float8_e5m2,
        b.dtype==torch.float8_e5m2,0,True,False,False,False,BM=bm,BN=bn,BK=bk,num_warps=warps,
        num_stages=stages,enable_fp_fusion=False,matrix_instr_nonkdim=mfma)
    return out

def dw(a,b,offs,sa,sb,e,out,flag,config):
    bm,bn,bk,warps,stages,mfma=config;h,n=a.shape[0],b.shape[0]
    _dw[(tr.cdiv(h,bm),tr.cdiv(n,bn),e)](a.view(torch.uint8),b.view(torch.uint8),out,offs,sa,sb,flag,
        h,n,*a.stride(),*b.stride(),a.dtype==torch.float8_e5m2,b.dtype==torch.float8_e5m2,
        BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False,matrix_instr_nonkdim=mfma)

def main():
    torch.manual_seed(169);torch.set_num_threads(1);f4=torch.float8_e4m3fn;f5=torch.float8_e5m2
    configs=[(32,64,64,4,1,16),(64,64,64,4,1,16),(64,128,64,4,1,16),
      (64,128,128,4,1,16),(64,128,128,4,2,16),(128,128,64,8,1,32),
      (64,128,128,4,1,32),(64,128,128,4,2,32),(128,128,128,8,1,32),
      (128,128,128,8,2,32),(64,64,128,4,1,32),(32,128,128,4,1,32)]
    results=[];dev='cuda';e=8
    scalar=torch.ones(1,device=dev);scales=torch.ones(e,device=dev);flag=torch.zeros(1,device=dev,dtype=torch.int32)
    for m in (16384,49152):
        tiles=torch.arange(e,device=dev,dtype=torch.int32).repeat_interleave(m//e//128)
        offs=torch.arange(1,e+1,device=dev,dtype=torch.int32)*(m//e)
        for k,n in ((768,1024),(1024,768)):
            for dtype in (f4,f5):
                a=torch.randn(m,k,device=dev).to(dtype);b=torch.randn(e*n,k,device=dev).to(f4).T
                ref=run(a,b,scales,scalar,tiles=tiles,n=n)[0];rows=[]
                base=tr.testing.do_bench_cudagraph(lambda:run(a,b,scales,scalar,tiles=tiles,n=n),rep=40)
                for config in configs:
                    try:
                        out=mm(a,b,scales,scalar,tiles,n,config)
                        torch.testing.assert_close(out,ref,atol=.005,rtol=.0001)
                        ms=tr.testing.do_bench_cudagraph(lambda:mm(a,b,scales,scalar,tiles,n,config),rep=40)
                        rows.append(dict(config=config,ms=ms))
                    except Exception as ex:rows.append(dict(config=config,error=str(ex)[:140]))
                best=min((r for r in rows if 'ms' in r),key=lambda r:r['ms'])
                result=dict(kind='mm',m=m,k=k,n=n,dtype=str(dtype),baseline_ms=base,best=best,variants=rows)
                results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='variants'}),flush=True)
        for ad,bd in ((f4,f5),(f5,f4)):
            a=torch.randn(1024,m,device=dev).to(ad);b=torch.randn(768,m,device=dev).to(bd)
            out=torch.empty(e,1024,768,device=dev,dtype=torch.bfloat16)
            grouped_dw(a,b,offs,scalar,scalar,e,out,flag);ref=out.clone();rows=[]
            base=tr.testing.do_bench_cudagraph(lambda:grouped_dw(a,b,offs,scalar,scalar,e,out,flag),rep=40)
            for config in configs:
                try:
                    dw(a,b,offs,scalar,scalar,e,out,flag,config)
                    torch.testing.assert_close(out,ref,atol=.01,rtol=.008)
                    ms=tr.testing.do_bench_cudagraph(lambda:dw(a,b,offs,scalar,scalar,e,out,flag,config),rep=40)
                    rows.append(dict(config=config,ms=ms))
                except Exception as ex:rows.append(dict(config=config,error=str(ex)[:140]))
            best=min((r for r in rows if 'ms' in r),key=lambda r:r['ms'])
            result=dict(kind='dw',m=m,k=1024,n=768,dtype=str(ad),baseline_ms=base,best=best,variants=rows)
            results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='variants'}),flush=True)
    from pathlib import Path
    Path('research_results/wr_reproduction/ks3_fp8_tile_sweep.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
