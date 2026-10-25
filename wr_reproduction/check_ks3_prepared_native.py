"""Independent format oracle, refresh replay, and paired prepared-weight checks."""
import statistics
import torch
import triton as tr


def native_copy(a):
    """Independent byte oracle: original E4M3FN becomes half-value FNUZ."""
    bits=a.contiguous().view(torch.uint8)
    is_e5=a.dtype==torch.float8_e5m2
    invalid=(bits&127)>=124 if is_e5 else (bits&127)==127
    bits=torch.where(invalid,128,torch.where(bits==128,0,bits)).to(torch.uint8)
    return bits.view(torch.float8_e5m2fnuz if is_e5 else torch.float8_e4m3fnuz)


@torch.no_grad()
def check(activations=False,dual=False,native_gather=False,gs32=True,charts=1024):
    from wr_reproduction import ks3_moe_fp8 as fp,ks3_fp8_gemm as gemm
    tile_policy=gemm.TILE_MODE
    from wr_reproduction.ks3_moe_fp8 import init_buffers,refresh_weights,forward,backward
    from wr_reproduction.ks3_fp8_gemm import _run_tiled,_dw_tiled
    from track_1_short.perf.kernels import moe_mlp as moe
    device='cuda';e,n,d=8,charts,768
    scalar=lambda x:torch.tensor([x],device=device,dtype=torch.float32)
    one=scalar(1.);byte_checks=[]
    # Cover every stored byte in both activation formats and every weight byte,
    # with both matrix layouts. Nonfinite source values stay invalid.
    for dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
        a=torch.arange(256,device=device,dtype=torch.int32).byte()[:,None].expand(256,128).contiguous().view(dtype)
        base=torch.arange(256,device=device,dtype=torch.int32).byte()[None,:].expand(128,256).contiguous().view(torch.float8_e4m3fn)
        for transpose in (False,True):
            b=base.T.contiguous().T if transpose else base
            nb=native_copy(b.T).T if transpose else native_copy(b)
            tiles=torch.zeros(2,device=device,dtype=torch.int32)
            values=[]
            for weights in (b,nb):values.append(_run_tiled(a,weights,one,one,tiles,256,(64,128,128,4,1,16)))
            if activations:values.append(_run_tiled(native_copy(a),nb,one,one,tiles,256,(64,128,128,4,1,16)))
            af,bf=a.float(),b.float();af[~af.isfinite()]=float('nan');bf[~bf.isfinite()]=float('nan')
            oracle=af@bf
            for actual in values:torch.testing.assert_close(actual,oracle,rtol=1e-6,atol=0,equal_nan=True)
            torch.testing.assert_close(values[0],values[1],rtol=0,atol=0,equal_nan=True)
            byte_checks.append(dict(activation_format=str(dtype),transposed_weights=transpose,all256_bytes=True,decoded_fp32_oracle=True,prepared_activations=activations))
    dw_checks=[]
    if dual:
        for adtype in (torch.float8_e4m3fn,torch.float8_e5m2):
            for bdtype in (torch.float8_e4m3fn,torch.float8_e5m2):
                raw=torch.arange(256,device=device,dtype=torch.int32).byte()[:,None].expand(256,128).contiguous()
                a=raw.view(adtype);b=raw.view(bdtype)
                af,bf=a.float(),b.float();af[~af.isfinite()]=float('nan');bf[~bf.isfinite()]=float('nan')
                product=af@bf.T;offs=torch.tensor([0,128,128],device=device,dtype=torch.int32)
                for native_a,native_b in ((True,False),(False,True),(True,True)):
                    aa=native_copy(a) if native_a else a;bb=native_copy(b) if native_b else b
                    for accumulate in (False,True):
                        out=torch.full((3,256,256),2.,device=device)
                        flag=torch.tensor([int(accumulate)],device=device,dtype=torch.int32)
                        _dw_tiled(aa,bb,offs,one,one,3,out,flag,(64,128,64,4,1,16))
                        oracle=torch.zeros_like(out);oracle[1]=product
                        if accumulate:oracle+=2.
                        torch.testing.assert_close(out,oracle,rtol=1e-6,atol=0,equal_nan=True)
                    dw_checks.append(dict(a_format=str(adtype),b_format=str(bdtype),native_a=native_a,native_b=native_b,
                                          all256_bytes=True,empty_experts=True,accumulation_checked=True))
    gather_checks=[]
    if native_gather:
        from wr_reproduction.ks3_native_gather import gather
        for tokens in (1,257,16384,49152):
            for nonfinite in ((False,True) if tokens==257 else (False,)):
                grad=torch.randn(tokens,768,device=device,dtype=torch.bfloat16)*.02
                if nonfinite:grad[0,0]=float('nan');grad[1,1]=float('inf')
                expert=torch.randint(8,(tokens,),device=device)
                if tokens==257:expert.zero_()
                _,src,_,_,_=moe.moe_dispatch(expert,8)
                y=torch.randn(src.numel(),768,device=device,dtype=torch.bfloat16)
                gate=torch.rand(tokens,device=device).bfloat16().float();scale=scalar(.002)
                original=moe.grad_gather_quant(grad,y,src,gate,scale)
                packed=gather(grad,y,src,gate,scale)
                for old,new in zip(original[:2],packed[:2]):assert torch.equal(old.view(torch.uint8),new.view(torch.uint8))
                torch.testing.assert_close(original[2],packed[2],rtol=0,atol=0,equal_nan=True)
                for old,new in zip(original[:2],packed[3:]):assert torch.equal(native_copy(old).view(torch.uint8),new.view(torch.uint8))
                gather_checks.append(dict(tokens=tokens,nonfinite=nonfinite,original_bytes_and_gate_gradient_exact=True,native_layouts_exact=True))
    m=torch.nn.Module();m.device=torch.device(device);m.ks3_batched_weight_pack=True
    m.ks3_training_gs32=gs32;m.ks3_prepared_native_weights=True
    m.ks3_geometry=torch.zeros((5,e,n,8),device=device)
    m.moe_bank=(torch.randn(80,n,d,device=device)*.025).bfloat16();init_buffers(m)
    original_names=['ks3_f8_'+suffix for suffix in ('up','up_t','down','down_t')]
    names=original_names+['ks3_weight_scales','ks3_weight_amax']
    def compare():
        for name in names:assert torch.equal(getattr(m,name).view(torch.uint8),expected[name].view(torch.uint8)),name
        for name in original_names:
            got=getattr(m,name.replace('ks3_f8_','ks3_native_'))
            assert torch.equal(got.view(torch.uint8),native_copy(getattr(m,name)).view(torch.uint8)),name
    refresh_checks=[]
    for index,exact in enumerate((True,False,False,True)):
        if index:m.moe_bank.add_(torch.randn_like(m.moe_bank)*.001)
        previous=m.ks3_weight_amax.clone();m.ks3_prepared_native_weights=False;refresh_weights(m,exact)
        expected={name:getattr(m,name).clone() for name in names}
        m.ks3_weight_amax.copy_(previous);m.ks3_prepared_native_weights=True;refresh_weights(m,exact);compare()
        refresh_checks.append(dict(exact=exact,original_caches_and_scales_bitwise=True,native_byte_mapping_exact=True))
    graph=torch.cuda.CUDAGraph();torch.cuda.synchronize()
    with torch.cuda.graph(graph):refresh_weights(m,False)
    for index in range(2):
        m.moe_bank.add_(torch.randn_like(m.moe_bank)*.001)
        previous=m.ks3_weight_amax.clone();m.ks3_prepared_native_weights=False;refresh_weights(m,False)
        expected={name:getattr(m,name).clone() for name in names}
        m.ks3_weight_amax.copy_(previous);m.ks3_prepared_native_weights=True;graph.replay();compare()
        refresh_checks.append(dict(captured_replay=True,original_caches_and_scales_bitwise=True,native_byte_mapping_exact=True))
    refresh_samples={key:[] for key in ('original','prepared')}
    for order in (('original','prepared'),('prepared','original'),('original','prepared')):
        for key in order:
            m.ks3_prepared_native_weights=key=='prepared'
            refresh_samples[key].append(tr.testing.do_bench_cudagraph(lambda:refresh_weights(m,False),rep=100))
    m.ks3_prepared_native_weights=True;refresh_weights(m,True)
    g=torch.randn(e,n,8,device=device)*.01;g[...,6]=.75
    us,ws=m.ks3_weight_scales[0,0],m.ks3_weight_scales[0,1]
    xs,ps,gs,ds=map(scalar,(2**-4,.01,2**-6,.0001));pa,da=scalar(0),scalar(0)
    flag=torch.zeros(1,device=device,dtype=torch.int32);block_checks=[]
    for tokens in (1,257,16384,49152):
        x=torch.randn(tokens,d,device=device,dtype=torch.bfloat16);grad=torch.randn_like(x)*.02
        expert=torch.randint(e,(tokens,),device=device)
        if tokens==257:expert.zero_()
        dest,src,offs,tiles,counts=moe.moe_dispatch(expert,e)
        gate=torch.ones(tokens,device=device);bank=torch.zeros(16,n,d,device=device,dtype=torch.bfloat16)
        def execute():
            up,upt,down,downt=caches
            y,c,xt,ht=forward(x,src,tiles,up,down,us,ws,xs,ps,pa,g,gs32)
            dx,dgate,dg=backward(grad,y,c,xt,ht,gate,src,offs,tiles,upt,downt,us,ws,xs,gs,ps,ds,da,bank,flag,g,0)
            return y,dx,dgate,dg,bank,xt,ht
        caches=tuple(getattr(m,name)[0] for name in original_names)
        fp.PREPARED_ACTIVATIONS=False;fp.PREPARED_DUAL=False;fp.PREPARED_GATHER=False
        expected=[a.clone() for a in execute()]
        caches=tuple(getattr(m,name.replace('ks3_f8_','ks3_native_'))[0] for name in original_names)
        fp.PREPARED_ACTIVATIONS=activations;fp.PREPARED_DUAL=dual;fp.PREPARED_GATHER=native_gather
        got=execute()
        for name,a,b in zip(('output','dx','dgate','geometry','weight_gradient'),got,expected):
            assert a.isfinite().all(),(tokens,name)
            torch.testing.assert_close(a,b,rtol=0,atol=0,msg=lambda msg:f'{tokens} {name}: {msg}')
        if dual:
            for actual,original in zip(got[-2:],expected[-2:]):
                assert torch.equal(actual.view(torch.uint8),native_copy(original).view(torch.uint8))
        labels=('original','weights_only','rows_only','dual_only','prepared') if native_gather else (('original','weights_only','rows_only','prepared') if dual else (('original','weights_only','prepared') if activations else ('original','prepared')))
        if tile_policy!='legacy':labels=('native_v1','prepared')
        samples={key:[] for key in labels}
        if tokens>=16384:
            for order in (labels,labels[::-1],labels):
                for key in order:
                    gemm.TILE_MODE='legacy' if key=='native_v1' else tile_policy
                    fp.PREPARED_ACTIVATIONS=activations and key in ('rows_only','dual_only','prepared','native_v1')
                    fp.PREPARED_DUAL=dual and key in ('dual_only','prepared','native_v1')
                    fp.PREPARED_GATHER=native_gather and key in ('prepared','native_v1')
                    prefix='ks3_native_' if key!='original' else 'ks3_f8_'
                    caches=tuple(getattr(m,prefix+suffix)[0] for suffix in ('up','up_t','down','down_t'))
                    samples[key].append(tr.testing.do_bench_cudagraph(execute,rep=100))
        row=dict(tokens=tokens,full_forward_backward_bitwise_equal=True,core_samples_ms=samples,
                 median_ms={k:statistics.median(v) for k,v in samples.items() if v})
        block_checks.append(row);print(row,flush=True)
    fp.PREPARED_ACTIVATIONS=activations;fp.PREPARED_DUAL=dual;fp.PREPARED_GATHER=native_gather
    gemm.TILE_MODE=tile_policy
    return dict(native_tile_policy=tile_policy,charts=charts,training_gs32=gs32,prepared_gather=native_gather,gather_checks=gather_checks,prepared_dual=dual,dw_checks=dw_checks,prepared_activations=activations,byte_checks=byte_checks,refresh_checks=refresh_checks,refresh_samples_ms=refresh_samples,
                refresh_median_ms={k:statistics.median(v) for k,v in refresh_samples.items()},block_checks=block_checks,
                original_formats_retained=True,extra_native_cache_bytes=4*5*8*n*d,
                scope='Correctness and isolated core/producer timing; whole-model speed and convergence not established')
