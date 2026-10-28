"""Short independent GEMM/packing oracles and paired complete-block timing."""
import os,sys,json,statistics,datetime,hashlib
from pathlib import Path
import torch,triton as tr,triton.testing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.port import install
install()
from experiments.h100_ks3 import optimized,fp8
from wr_reproduction import ks3_moe as core

def err(a,b):return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
def scalar(x):return torch.tensor([x],device='cuda')
def producer_checks(rows,router,grad,g,xs,gs,ps,ds):
    from track_1_short.perf.kernels.moe_mlp import moe_dispatch
    probs=torch.softmax((rows@router.T).float(),-1);gate,expert=probs.max(-1)
    _,src,offs,table,_=moe_dispatch(expert,8);m=src.numel();n=g.shape[1]
    x=torch.empty(m,768,device='cuda',dtype=torch.float8_e4m3fn);xt=torch.empty(768,m,device='cuda',dtype=x.dtype)
    fp8.pack_input[(tr.cdiv(m,128),24)](rows,x,xt,src,xs,m,rows.shape[0],num_warps=4)
    pairs={'input':(x,xt)}
    c=torch.randn(m,n,device='cuda')*.5;prepared=torch.empty(8,n,9,device='cuda')
    fp8._prepare[(tr.cdiv(8*n,256),)](g,prepared,8*n)
    h=torch.empty(m,n,device='cuda',dtype=x.dtype);ht=torch.empty(n,m,device='cuda',dtype=x.dtype)
    maxima=torch.empty(tr.cdiv(m,64)*tr.cdiv(n,32),device='cuda')
    fp8._geometry_prepared[(tr.cdiv(m,64),tr.cdiv(n,32))](c,g,h,table,m,n,HT=ht,SCALE=ps,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=64,PREPARED=prepared,num_warps=4)
    pairs['geometry']=(h,ht)
    dc=torch.empty(m,n,device='cuda',dtype=torch.float8_e5m2);dct=torch.empty(n,m,device='cuda',dtype=dc.dtype)
    part=torch.empty(tr.cdiv(m,128),n,8,device='cuda');v=torch.randn_like(c)*.01
    fp8._vjp_prepared[(tr.cdiv(m,128),tr.cdiv(n,32))](v,c,g,dc,part,table,src,m,n,rows.shape[0],DCT=dct,SCALE=ds,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=128,PREPARED=prepared,num_warps=4)
    pairs['geometry_gradient']=(dc,dct)
    gg=torch.empty(m,768,device='cuda',dtype=dc.dtype);gt=torch.empty(768,m,device='cuda',dtype=dc.dtype)
    dg=torch.empty(m,device='cuda');y=torch.randn(m,768,device='cuda',dtype=torch.bfloat16)
    fp8.gather_gate[(tr.cdiv(m,4),)](grad,y,src,gate,gg,gt,dg,gs,rows.shape[0],m,num_warps=4,enable_fp_fusion=False)
    pairs['gathered_gradient']=(gg,gt)
    result={name:dict(bitwise_equal=torch.equal(a.view(torch.uint8),b.T.view(torch.uint8)),finite=bool(torch.isfinite(a.float()).all()&torch.isfinite(b.float()).all())) for name,(a,b) in pairs.items()}
    print('PRODUCER_CHECK',m,json.dumps(result),flush=True)
    assert all(v['bitwise_equal'] and v['finite'] for v in result.values()),result
    return result
def packing_check():
    n=1024;d=768
    bank=torch.randn(80,n,d,device='cuda',dtype=torch.bfloat16)*.03
    scale=bank.float().abs().amax((1,2))/448
    q=torch.empty(80,n*d,device='cuda',dtype=torch.float8_e4m3fn)
    partial=torch.empty(80,n*d//4096,device='cuda')
    fp8.publish[(n*d//4096,80)](bank,q,scale,partial,0,n,num_warps=4)
    up=torch.empty(5,8,n,d,device='cuda',dtype=q.dtype);upt=torch.empty(5,8,d,n,device='cuda',dtype=q.dtype)
    down=torch.empty_like(upt);downt=torch.empty_like(up)
    fp8.layouts[(tr.cdiv(n*d,1024),80)](q,up,upt,down,downt,n,num_warps=4)
    differences=0;elements=0
    for slot in range(80):
        weight=bank[slot] if slot%16<8 else bank[slot].T
        a=weight.float().reshape(weight.shape[0],-1,32)
        s=(a.abs().amax(-1)/127).clamp(2**-24,65504).half().float()
        values=(a/s[:,:,None]).round().clamp(-127,127)*s[:,:,None]
        expected=(values/scale[slot]).clamp(-448,448).to(q.dtype).reshape(weight.shape)
        differences+=int((q[slot].view(torch.uint8)!=expected.flatten().view(torch.uint8)).sum());elements+=expected.numel()
        layer=slot//16;expert=slot%8
        primary,dual=(up,upt) if slot%16<8 else (down,downt)
        assert torch.equal(primary[layer,expert].flatten().view(torch.uint8),q[slot].view(torch.uint8))
        assert torch.equal(dual[layer,expert].T.contiguous().view(torch.uint8),primary[layer,expert].view(torch.uint8))
    assert differences/elements<1e-5,(differences,elements)
    return dict(dual_layouts_bitwise=True,torch_rounding_different_elements=differences,elements=elements)
def main():
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(419)
    torch.backends.cuda.matmul.allow_tf32=False
    optimized.install('separate-dx-prepared-128-gate-tanh');core.BF16_TUNED_KERNELS=True
    packing=packing_check();cases=[];oracle=[]
    for t in (16384,49152):
        n=1024;rows=torch.randn(t,768,device='cuda',dtype=torch.bfloat16).requires_grad_()
        router=(torch.randn(8,768,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_();post=scalar(.8).requires_grad_()
        w1=torch.randn(8,n,768,device='cuda',dtype=torch.bfloat16)*.02;w2=torch.randn_like(w1)*.02
        up=core.pack(w1.reshape(-1,768)).reshape_as(w1);down=torch.stack([core.pack(w.T) for w in w2])
        g=torch.randn(8,n,8,device='cuda')*.01;g[...,6]=.75;g.requires_grad_()
        bank=torch.empty(16,n,768,device='cuda',dtype=torch.bfloat16);flag=torch.zeros((),device='cuda',dtype=torch.int32)
        grad=torch.randn_like(rows)*.02;gp=torch.randn(t,8,device='cuda')*.002
        us=up.float().abs().amax((1,2)).clamp_min(1e-12)/448;ws=down.float().abs().amax((1,2)).clamp_min(1e-12)/448
        uf=(up.float()/us[:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        wf=(down.float()/ws[:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        caches=(uf,uf.transpose(1,2).contiguous(),wf,wf.transpose(1,2).contiguous())
        xs=scalar(8/448);gs=scalar(2**-6);ps=scalar(1.);pa=scalar(0.);ds=scalar(2**-6);da=scalar(0.)
        scales=(us,ws,xs,gs,ps,pa,ds,da)
        producers=producer_checks(rows,router,grad,g,xs,gs,ps,ds)
        table=torch.arange(8,device='cuda',dtype=torch.int32).repeat_interleave(t//8//128)
        af=(rows.detach().float()/xs).clamp(-448,448).to(torch.float8_e4m3fn)
        actual=fp8.mm(af,uf.transpose(1,2),table,xs,us)
        expected=torch.cat([(af[i*t//8:(i+1)*t//8].float()*xs)@(uf[i].float()*us[i]).T for i in range(8)])
        error=err(actual,expected);assert error<.003,error
        oracle_row=dict(tokens=t,projection_relative_rms=error)
        if t==16384:
            hf=(torch.randn(n,t,device='cuda')*.02/gs).to(torch.float8_e5m2)
            off=torch.arange(1,9,device='cuda',dtype=torch.int32)*(t//8)
            output=torch.empty(8,n,768,device='cuda',dtype=torch.bfloat16)
            fp8.dw(hf,af.T.contiguous(),output,off,gs,xs,flag)
            expected_dw=torch.stack([(hf[:,i*t//8:(i+1)*t//8].float()*gs)@(af[i*t//8:(i+1)*t//8].float()*xs) for i in range(8)])
            dw_error=err(output,expected_dw);assert dw_error<.004,dw_error
            flag.fill_(1);fp8.dw(hf,af.T.contiguous(),output,off,gs,xs,flag);flag.zero_()
            acc_error=err(output,expected_dw*2);assert acc_error<.004,acc_error
            oracle_row.update(weight_gradient_relative_rms=dw_error,accumulated_gradient_relative_rms=acc_error)
        oracle.append(oracle_row)
        def run(kind):
            if kind=='bf16':y,p,c=core.RoutedKS3.apply(rows,router,post,w1,w2,up,down,g,bank,0,True)
            else:y,p,c=fp8.Routed.apply(rows,router,post,w1,w2,g,caches,scales,bank,flag,0)
            grads=torch.autograd.grad((y,p),(rows,router,post,g),(grad,gp))
            return (y.detach(),*grads,bank)
        ref=[a.detach().clone() for a in run('bf16')]
        for warm in range(3):
            values=run('fp8')
            finite={name:dict(finite=int(torch.isfinite(value).sum()),elements=value.numel(),maximum=float(value.float().abs().nan_to_num().max())) for name,value in zip(('output','dx','router','post','geometry','weights'),values)}
            print(json.dumps(dict(tokens=t,warm=warm,finite=finite,ps=ps.item(),ds=ds.item(),pa=pa.item(),da=da.item())),flush=True)
            ps.copy_(pa*1.03/448);ds.copy_(da*1.25/57344)
        actual=run('fp8');errors={key:err(a,b) for key,a,b in zip(('output','dx','router','post','geometry','weights'),actual,ref)}
        assert all(torch.isfinite(a).all() for a in actual)
        assert max(errors.values())<.15,errors
        timing={}
        for kind in ('bf16','fp8'):
            samples=[tr.testing.do_bench_cudagraph(lambda:run(kind),rep=40) for _ in range(3)]
            timing[kind]=dict(samples_ms=samples,median_ms=statistics.median(samples))
        result=dict(tokens=t,producers=producers,errors_versus_bf16=errors,timing=timing,scales=dict(post=ps.item(),dpre=ds.item()),passed=True)
        cases.append(result);print(json.dumps(result),flush=True)
    from triton.backends.nvidia.compiler import get_ptxas
    assembler=get_ptxas(90)
    output=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),packing=packing,oracle=oracle,cases=cases,
                ptxas=dict(path=assembler.path,version=assembler.version,sha256=hashlib.sha256(Path(assembler.path).read_bytes()).hexdigest()))
    Path('research_results/wr_reproduction/h100-research/fp8-micro.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
