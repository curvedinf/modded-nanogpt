"""FP32 BLAS VJP verification; device checks only run inside registered diagnostics."""
import unittest
import torch
from ks3_research.backward import ks3_blas,unpack_fp32
from ks3_research.vendor.ks3_int8 import ks3_int8,KS3Int8
from ks3_research.vendor.ks3_reference import KS3
from test_ks3 import reference


def relative_error(a,b):
    a,b=a.detach().float(),b.detach().float()
    return dict(max_absolute=float((a-b).abs().max()),
                relative_rms=float((a-b).norm()/b.norm().clamp_min(1e-12)))


def check(device,dtype,n,inp,out,charts,rank,all_products=False,output_blas=False,input_blas=False,
          fused_unpack=False):
    torch.manual_seed(103)
    model=KS3Int8(KS3(inp,out,charts,rank).to(device),nonlinear='fp16',trainable=True)
    with torch.no_grad():
        model.master_coefficients.normal_(0,.01)
    x=(torch.randn(1,n,inp,device=device,dtype=dtype)*.3).requires_grad_()
    params=(x,model.master_down,model.master_up,model.master_coefficients,
            model.master_raw_scale,model.master_raw_shift)
    options=dict(subspace_rank=rank,nonlinear='fp16',tanh_lut=model.tanh_lut)
    old=ks3_int8(*params,**options)
    new=ks3_blas(*params,**options,all_products=all_products,output_blas=output_blas,input_blas=input_blas,
                 fused_unpack=fused_unpack)
    if input_blas and rank==1:
        saved=new.grad_fn.saved_tensors
        assert saved[8].data_ptr()==saved[-1].data_ptr(), 'Rank-one coordinate storage must be shared'
    forward_error=relative_error(new,old)
    if input_blas:
        # Input-product rounding can cross a later FP16/INT8 boundary. Set the
        # numerical-variant bounds before GPU measurement, separately from the
        # tighter output-only/backward-only equivalence requirements.
        assert forward_error['relative_rms']<.002,forward_error
        assert forward_error['max_absolute']<max(2e-6,.01*float(old.detach().abs().max())),forward_error
    elif output_blas:
        if dtype==torch.float32:
            assert forward_error['relative_rms']<1e-5,forward_error
            torch.testing.assert_close(new,old,atol=2e-6,rtol=2e-5)
        else:
            assert forward_error['relative_rms']<.001,forward_error
            assert forward_error['max_absolute']<max(2e-6,.005*float(old.detach().abs().max())),forward_error
    else:
        torch.testing.assert_close(new,old,atol=0,rtol=0)
    upstream=torch.randn_like(old)/old.numel()**.5
    expected=torch.autograd.grad(old,params,upstream)
    actual=torch.autograd.grad(new,params,upstream)
    if fused_unpack:
        baseline=ks3_blas(*params,**options,all_products=True,output_blas=True,input_blas=True)
        torch.testing.assert_close(new,baseline,atol=0,rtol=0)
        baseline_grads=torch.autograd.grad(baseline,params,upstream)
        for a,b in zip(actual,baseline_grads):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    errors=[]
    for i,(a,b) in enumerate(zip(actual,expected)):
        e=relative_error(a,b)
        if input_blas:
            assert e['relative_rms']<.003,e
            assert e['max_absolute']<max(2e-6,.01*float(b.detach().abs().max())),e
        else:
            assert e['relative_rms']<(.003 if i==0 and dtype==torch.bfloat16 else .0005),e
            torch.testing.assert_close(a,b,atol=2e-6,rtol=.005 if dtype==torch.bfloat16 else .0005)
        errors.append(e)
    if output_blas:
        # Inference under an ambient autocast must still compute the FP32
        # projection, then cast the result as the original API specifies.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            inference=ks3_blas(*params,**options,all_products=True,output_blas=True,input_blas=input_blas,
                               fused_unpack=fused_unpack)
        torch.testing.assert_close(inference,new.detach().bfloat16(),atol=0,rtol=0)
    independent=False
    if dtype==torch.float32 and n<100:
        dense=reference(model,x)
        dense_grads=torch.autograd.grad(dense,params,upstream)
        for a,b in zip(actual,dense_grads):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=5e-4)
        independent=True
    return dict(dtype=str(dtype),tokens=n,input_dim=inp,output_dim=out,charts=charts,rank=rank,
                forward_bitwise_equal=bool(torch.equal(new,old)),forward_error=forward_error,
                autocast_inference_checked=output_blas,
                rank1_coordinate_storage_shared=input_blas and rank==1,
                fused_unpack_bitwise_forward_and_gradients=fused_unpack,
                gradients=errors,independent_reference=independent)


def run_device_checks(device,backend='fp32_blas',forward_backend='original'):
    checks=[check(device,dtype,*shape,all_products=backend=='fp32_blas_all',
                  output_blas=forward_backend in ('blas_output','blas_both','blas_fused_unpack'),
                  input_blas=forward_backend in ('blas_both','blas_fused_unpack'),
                  fused_unpack=forward_backend=='blas_fused_unpack') for dtype in (torch.float32,torch.bfloat16)
            for shape in ((37,65,37,33,1),(37,65,37,33,2),(37,65,37,33,3),
                          (4097,768,768,1536,1),(4097,768,768,768,2),
                          (32769,65,37,33,1))]
    unpack_checks=[]
    if forward_backend=='blas_fused_unpack':
        from ks3_research.unpack import unpack_gs32
        for rows,columns in ((0,65),(71,65),(1025,768),(131073,1536)):
            for dtype in (torch.float16,torch.float32):
                codes=torch.randint(-128,128,(rows*2,columns*2),device=device,dtype=torch.int8)[::2,::2]
                scales=torch.rand(rows*2,((columns+31)//32)*2,device=device,dtype=dtype)[::2,::2]
                if rows:
                    scales[0]=0
                    scales[-1]=65504
                got=unpack_gs32(codes,scales)
                want=unpack_fp32(codes,scales)
                torch.testing.assert_close(got,want,atol=0,rtol=0)
                unpack_checks.append(dict(rows=rows,columns=columns,scale_dtype=str(dtype),
                                          strided=True,bitwise_equal=True))
    return dict(checks=checks,passed=True,backend=backend,
                unpack_checks=unpack_checks,
                forward_backend=forward_backend,
                float32_matmul_precision=torch.get_float32_matmul_precision())


class BLASBackwardCPUTests(unittest.TestCase):
    def test_fused_unpack_requires_a_new_pilot(self):
        from ks3_research.planning import load_config,plan,signature
        config=load_config('configs/ks3/ablation_attention64_pilot.json')
        old=signature(config)
        config['ks3_forward']='blas_fused_unpack'
        self.assertNotEqual(old,signature(config))
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(config,dict(signature=old))

    def test_fused_unpack_cpu_fallback_and_shape_validation(self):
        from ks3_research.unpack import unpack_gs32
        codes=torch.randint(-128,128,(142,130),dtype=torch.int8)[::2,::2]
        scales=torch.rand(142,6).half()[::2,::2]
        torch.testing.assert_close(unpack_gs32(codes,scales),unpack_fp32(codes,scales),atol=0,rtol=0)
        torch.testing.assert_close(unpack_gs32(codes,scales,dtype=torch.bfloat16),
                                   unpack_fp32(codes,scales).bfloat16(),atol=0,rtol=0)
        with self.assertRaises(ValueError):
            unpack_gs32(codes,scales[:,:2])

    def test_input_projection_change_requires_a_new_pilot(self):
        from ks3_research.planning import load_config,plan,signature
        config=load_config('configs/ks3/ablation_b64k_fast_trial.json')
        old=signature(config)
        config['ks3_forward']='blas_both'
        self.assertNotEqual(old,signature(config))
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(config,dict(signature=old))

    def test_output_projection_change_requires_a_new_pilot(self):
        from ks3_research.planning import load_config,plan,signature
        config=load_config('configs/ks3/ablation_all_blas_trial.json')
        old=signature(config)
        config['ks3_forward']='blas_output'
        self.assertNotEqual(old,signature(config))
        with self.assertRaisesRegex(ValueError,'Run a pilot'):
            plan(config,dict(signature=old))

    def test_partial_gs32_dequantization_is_exact_fp32(self):
        torch.manual_seed(102)
        codes=torch.randint(-127,128,(71,65),dtype=torch.int8)
        scales=torch.rand(71,3).half()
        got=unpack_fp32(codes,scales)
        want=torch.stack([codes[:,j].float()*scales[:,j//32].float() for j in range(65)],1)
        torch.testing.assert_close(got,want,atol=0,rtol=0)
        self.assertEqual(got.dtype,torch.float32)


if __name__=='__main__':
    unittest.main(verbosity=2)
