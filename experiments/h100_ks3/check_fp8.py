"""Short independent GEMM/packing oracles and paired complete-block timing."""
import os,sys,json,statistics,datetime,hashlib
from pathlib import Path
import torch,triton as tr,triton.testing
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wr_reproduction/pr393/upstream'))
from experiments.h100_ks3.port import install
MODEL_CLASS=install()
from experiments.h100_ks3 import optimized,fp8
from wr_reproduction import ks3_moe as core

def err(a,b):return ((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()
def scalar(x):return torch.tensor([x],device='cuda')

@torch.no_grad()
def post_jacobian_check(rows,router,post,up,down,g,caches,scales,grad,actual_post):
    """Check the scalar derivative without dividing by a cancelling projection.

    The post parameter's Jacobian is the ungated output times its router gate.
    Compare that complete vector with BF16, then independently contract the
    FP8 Jacobian with the test gradient in FP64. Keep the old scalar relative
    error in the report; it is not a conditioning-independent error measure.
    """
    from track_1_short.perf.kernels.moe_mlp import moe_dispatch
    probs=torch.softmax(torch.nn.functional.linear(rows,router).float(),-1)
    gate,expert=probs.max(-1);_,src,_,table,_=moe_dispatch(expert,8)
    scale=gate*post.float().reshape(())
    by=optimized.forward(rows,src,table,up,down,g,scale)[1]
    if not fp8.GS32:
        # Exercise the final-validation core independently of the optimized
        # BF16 training oracle, including unquantized input/hidden packing.
        previous_mode=core.BF16_TUNED_KERNELS
        core.BF16_TUNED_KERNELS=False
        try:eval_y,eval_x,_,_=core.forward(rows,src,table,up,down,g,False)
        finally:core.BF16_TUNED_KERNELS=previous_mode
        valid_src=src<rows.shape[0]
        assert torch.equal(eval_x[valid_src],rows[src[valid_src].long()])
        assert torch.count_nonzero(eval_x[~valid_src])==0
        assert err(by,eval_y)<.006,('native BF16 validation mismatch',err(by,eval_y))
    us,ws,xs,_,ps,pa,_,_=scales
    fy=fp8.forward(rows,src,table,caches[0],caches[2],us,ws,xs,ps,pa,g,scale)[1]
    valid=src<rows.shape[0];source=src[valid].long()
    bj=by[valid].float()*gate[source,None]
    fj=fy[valid].float()*gate[source,None]
    vector_error=err(fj,bj)
    terms=fy[valid].double()*gate[source,None].double()*grad[source].double()
    oracle=terms.sum();scale_norm=terms.square().sum().sqrt().clamp_min(1e-12)
    normalized_error=((actual_post.double().reshape(())-oracle).abs()/scale_norm).item()
    result=dict(jacobian_relative_rms=vector_error,fp64_contraction=float(oracle),
                actual_post=float(actual_post),contraction_error_over_l2=normalized_error,
                cancellation_ratio=float(oracle.abs()/scale_norm))
    assert vector_error<.15 and normalized_error<5e-5,result
    return result
def producer_checks(rows,router,grad,g,xs,gs,ps,ds):
    from track_1_short.perf.kernels.moe_mlp import moe_dispatch
    probs=torch.softmax((rows@router.T).float(),-1);gate,expert=probs.max(-1)
    _,src,offs,table,_=moe_dispatch(expert,8);m=src.numel();rank=(g.shape[-1]-2)//6;n=g.shape[1]*rank
    x=torch.empty(m,768,device='cuda',dtype=torch.float8_e4m3fn);xt=torch.empty(768,m,device='cuda',dtype=x.dtype)
    fp8.pack_input[(tr.cdiv(m,128),24)](rows,x,xt,src,xs,m,rows.shape[0],GS32=fp8.GS32,num_warps=4)
    pairs={'input':(x,xt)}
    c=torch.randn(m,n,device='cuda')*.5;prepared=fp8.prepare_geometry(g,n)
    h=torch.empty(m,n,device='cuda',dtype=x.dtype);ht=torch.empty(n,m,device='cuda',dtype=x.dtype)
    maxima=torch.empty(tr.cdiv(m,64)*tr.cdiv(n,32),device='cuda')
    hd,htd=fp8.store_descriptors(h,ht,64)
    fp8._geometry_prepared[(tr.cdiv(m,64),tr.cdiv(n,32))](c,g,h,table,m,n,HT=ht,SCALE=ps,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=64,RANK=rank,GS32=fp8.GS32,PREPARED=prepared,H_DESC=hd,HT_DESC=htd,CUBIC_ONLY=fp8.CUBIC_ONLY,num_warps=4)
    pairs['geometry']=(h,ht)
    dc=torch.empty(m,n,device='cuda',dtype=torch.float8_e5m2);dct=torch.empty(n,m,device='cuda',dtype=dc.dtype)
    part=torch.empty(tr.cdiv(m,fp8.VJP_ROWS),n,4 if fp8.COMPACT_PART else 8,device='cuda');v=torch.randn_like(c)*.01
    maxima=torch.empty(tr.cdiv(m,fp8.VJP_ROWS)*tr.cdiv(n,32),device='cuda')
    dcd,dctd=fp8.store_descriptors(dc,dct,fp8.VJP_ROWS)
    fp8._vjp_prepared[(tr.cdiv(m,fp8.VJP_ROWS),tr.cdiv(n,32))](v,c,g,dc,part,table,src,m,n,rows.shape[0],DCT=dct,SCALE=ds,AMAX=maxima,FP8=True,PARTIAL_AMAX=True,BM=fp8.VJP_ROWS,RANK=rank,PREPARED=prepared,DC_DESC=dcd,DCT_DESC=dctd,CUBIC_ONLY=fp8.CUBIC_ONLY,COMPACT_PART=fp8.COMPACT_PART,num_warps=fp8.VJP_WARPS)
    pairs['geometry_gradient']=(dc,dct)
    gg=torch.empty(m,768,device='cuda',dtype=dc.dtype);gt=torch.empty(768,m,device='cuda',dtype=dc.dtype)
    dg=torch.empty(m,device='cuda');y=torch.randn(m,768,device='cuda',dtype=torch.bfloat16)
    fp8.gather(grad,y,src,gate,gg,gt,dg,gs,rows.shape[0],m)
    pairs['gathered_gradient']=(gg,gt)
    result={name:dict(bitwise_equal=torch.equal(a.view(torch.uint8),b.T.view(torch.uint8)),finite=bool(torch.isfinite(a.float()).all()&torch.isfinite(b.float()).all())) for name,(a,b) in pairs.items()}
    print('PRODUCER_CHECK',m,json.dumps(result),flush=True)
    assert all(v['bitwise_equal'] and v['finite'] for v in result.values()),result
    return result
def packing_check():
    n=int(os.environ.get('KS3_H100_WIDTH','1024'));d=768
    bank=torch.randn(80,n,d,device='cuda',dtype=torch.bfloat16)*.03
    scale=bank.float().abs().amax((1,2))/448
    if not fp8.GS32:
        from types import SimpleNamespace
        dummy=SimpleNamespace(moe_bank=bank,ks3_up=torch.empty(5,8,n,d,device='cuda',dtype=bank.dtype),ks3_down=torch.empty(5,8,d,n,device='cuda',dtype=bank.dtype))
        MODEL_CLASS.prepare_ks3_eval(dummy)
        original=bank.view(5,2,8,n,d)
        assert torch.equal(dummy.ks3_up,original[:,0])
        assert torch.equal(dummy.ks3_down,original[:,1].transpose(-1,-2))
    q=torch.empty(80,n*d,device='cuda',dtype=torch.float8_e4m3fn)
    partial=torch.empty(80,n*d//4096,device='cuda')
    fp8.publish[(n*d//4096,80)](bank,q,scale,partial,0,n,GS32=fp8.GS32,num_warps=4)
    up=torch.empty(5,8,n,d,device='cuda',dtype=q.dtype);upt=torch.empty(5,8,d,n,device='cuda',dtype=q.dtype)
    down=torch.empty_like(upt);downt=torch.empty_like(up)
    fp8.layouts[(tr.cdiv(n*d,1024),80)](q,up,upt,down,downt,n,num_warps=4)
    differences=0;elements=0
    for slot in range(80):
        weight=bank[slot] if slot%16<8 else bank[slot].T
        a=weight.float().reshape(weight.shape[0],-1,32)
        s=(a.abs().amax(-1)/127).clamp(2**-24,65504).half().float()
        values=(a/s[:,:,None]).round().clamp(-127,127)*s[:,:,None] if fp8.GS32 else a
        expected=(values/scale[slot]).clamp(-448,448).to(q.dtype).reshape(weight.shape)
        differences+=int((q[slot].view(torch.uint8)!=expected.flatten().view(torch.uint8)).sum());elements+=expected.numel()
        layer=slot//16;expert=slot%8
        primary,dual=(up,upt) if slot%16<8 else (down,downt)
        assert torch.equal(primary[layer,expert].flatten().view(torch.uint8),q[slot].view(torch.uint8))
        assert torch.equal(dual[layer,expert].T.contiguous().view(torch.uint8),primary[layer,expert].view(torch.uint8))
    assert differences/elements<1e-5,(differences,elements)
    return dict(native_eval_weight_copy_bitwise=True if not fp8.GS32 else None,dual_layouts_bitwise=True,torch_rounding_different_elements=differences,elements=elements)
def main():
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(419)
    torch.backends.cuda.matmul.allow_tf32=False
    reference_variant=os.environ.get('KS3_FP8_REFERENCE_VARIANT','separate-dx-prepared-128-gate-tanh')
    rank=int(os.environ.get('KS3_H100_RANK','1'))
    optimized.install(reference_variant);core.BF16_TUNED_KERNELS=True
    assert fp8.CUBIC_ONLY==optimized.CUBIC_ONLY
    packing=packing_check();cases=[];oracle=[]
    for t in map(int,os.environ.get('KS3_FP8_CHECK_TOKENS','16384,49152').split(',')):
        n=int(os.environ.get('KS3_H100_WIDTH','1024'));rows=torch.randn(t,768,device='cuda',dtype=torch.bfloat16).requires_grad_()
        router=(torch.randn(8,768,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_();post=scalar(.8).requires_grad_()
        w1=torch.randn(8,n,768,device='cuda',dtype=torch.bfloat16)*.02;w2=torch.randn_like(w1)*.02
        up=core.pack(w1.reshape(-1,768)).reshape_as(w1) if fp8.GS32 else w1
        down=torch.stack([core.pack(w.T) for w in w2]) if fp8.GS32 else w2.transpose(1,2).contiguous()
        g=torch.randn(8,n//rank,6*rank+2,device='cuda')*.01;g[...,6*rank]=.75
        if fp8.CUBIC_ONLY:
            assert rank==1
            g[...,4:6]=0.
        g.requires_grad_()
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
        if fp8.CUBIC_ONLY:assert torch.count_nonzero(actual[4][...,4:])==0
        assert all(torch.isfinite(a).all() for a in actual)
        post_check=post_jacobian_check(rows,router,post,up,down,g,caches,scales,grad,actual[3])
        assert max(v for k,v in errors.items() if k!='post')<.15,errors
        cache_errors=None
        system_variant=fp8.C_DTYPE!=torch.float32 or fp8.V_DTYPE!=torch.float32 or fp8.FUSE_GEOMETRY or fp8.MM_IMPL!='pointer' or fp8.TMA_OUTPUTS or fp8.COMPACT_PART or fp8.FUSE_CUBIC_VJP
        if system_variant:
            candidate=[a.detach().clone() for a in actual]
            selected_c_dtype,selected_dtype,selected_fusion,selected_mm,selected_outputs,selected_compact,selected_cubic_fusion=fp8.C_DTYPE,fp8.V_DTYPE,fp8.FUSE_GEOMETRY,fp8.MM_IMPL,fp8.TMA_OUTPUTS,fp8.COMPACT_PART,fp8.FUSE_CUBIC_VJP
            fp8.C_DTYPE=torch.float32;fp8.V_DTYPE=torch.float32;fp8.FUSE_GEOMETRY=False;fp8.MM_IMPL='pointer';fp8.TMA_OUTPUTS=False;fp8.COMPACT_PART=False;fp8.FUSE_CUBIC_VJP=False
            full_cache=[a.detach().clone() for a in run('fp8')]
            fp8.C_DTYPE=selected_c_dtype;fp8.V_DTYPE=selected_dtype;fp8.FUSE_GEOMETRY=selected_fusion;fp8.MM_IMPL=selected_mm;fp8.TMA_OUTPUTS=selected_outputs;fp8.COMPACT_PART=selected_compact;fp8.FUSE_CUBIC_VJP=selected_cubic_fusion
            cache_errors={key:err(a,b) for key,a,b in zip(('output','dx','router','post','geometry','weights'),candidate,full_cache)}
            assert max(v for k,v in cache_errors.items() if k!='post')<.006 and cache_errors['post']<.02,cache_errors
        timing={}
        for kind in ('bf16','fp8'):
            samples=[tr.testing.do_bench_cudagraph(lambda:run(kind),rep=40) for _ in range(3)]
            timing[kind]=dict(samples_ms=samples,median_ms=statistics.median(samples))
        if system_variant:
            selected_c_dtype,selected_dtype,selected_fusion,selected_mm,selected_outputs,selected_compact,selected_cubic_fusion=fp8.C_DTYPE,fp8.V_DTYPE,fp8.FUSE_GEOMETRY,fp8.MM_IMPL,fp8.TMA_OUTPUTS,fp8.COMPACT_PART,fp8.FUSE_CUBIC_VJP
            fp8.C_DTYPE=torch.float32;fp8.V_DTYPE=torch.float32;fp8.FUSE_GEOMETRY=False;fp8.MM_IMPL='pointer';fp8.TMA_OUTPUTS=False;fp8.COMPACT_PART=False;fp8.FUSE_CUBIC_VJP=False
            samples=[tr.testing.do_bench_cudagraph(lambda:run('fp8'),rep=40) for _ in range(3)]
            fp8.C_DTYPE=selected_c_dtype;fp8.V_DTYPE=selected_dtype;fp8.FUSE_GEOMETRY=selected_fusion;fp8.MM_IMPL=selected_mm;fp8.TMA_OUTPUTS=selected_outputs;fp8.COMPACT_PART=selected_compact;fp8.FUSE_CUBIC_VJP=selected_cubic_fusion
            timing['fp8_full_cache']=dict(samples_ms=samples,median_ms=statistics.median(samples))
        result=dict(tokens=t,producers=producers,post_jacobian=post_check,errors_versus_bf16=errors,errors_versus_full_cache=cache_errors,timing=timing,scales=dict(post=ps.item(),dpre=ds.item()),passed=True)
        cases.append(result);print(json.dumps(result),flush=True)
    from triton.backends.nvidia.compiler import get_ptxas
    assembler=get_ptxas(90)
    output=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),packing=packing,oracle=oracle,cases=cases,
                implementation=dict(width=int(os.environ.get('KS3_H100_WIDTH','1024')),gs32=fp8.GS32,rank=rank,cubic_only=fp8.CUBIC_ONLY,compact_part=fp8.COMPACT_PART,fuse_cubic_vjp=fp8.FUSE_CUBIC_VJP,cubic_vjp_tile=os.environ.get('KS3_FP8_CUBIC_VJP_TILE'),cubic_vjp_warps=os.environ.get('KS3_FP8_CUBIC_VJP_WARPS'),cubic_vjp_stages=os.environ.get('KS3_FP8_CUBIC_VJP_STAGES','3'),cubic_vjp_flatten=os.environ.get('KS3_FP8_CUBIC_VJP_FLATTEN','1'),cubic_vjp_moments=os.environ.get('KS3_FP8_CUBIC_VJP_MOMENTS','0'),fused_up_flatten=os.environ.get('KS3_FP8_FUSED_UP_FLATTEN','1'),tma_outputs=fp8.TMA_OUTPUTS,dw=fp8.DW_IMPL,vjp_rows=fp8.VJP_ROWS,vjp_warps=fp8.VJP_WARPS,gather=fp8.GATHER_IMPL,tma_stores=fp8.TMA_STORES,c_dtype=str(fp8.C_DTYPE),v_dtype=str(fp8.V_DTYPE),fuse_geometry=fp8.FUSE_GEOMETRY,fused_up_tile=os.environ.get('KS3_FP8_FUSED_UP_TILE'),mm=fp8.MM_IMPL,mm_tile=os.environ.get('KS3_FP8_MM_TILE'),reference_variant=reference_variant),
                ptxas=dict(path=assembler.path,version=assembler.version,sha256=hashlib.sha256(Path(assembler.path).read_bytes()).hexdigest()))
    Path(os.environ.get('KS3_FP8_CHECK_OUTPUT','research_results/wr_reproduction/h100-research/fp8-micro.json')).write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
