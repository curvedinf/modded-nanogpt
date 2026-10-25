import unittest
from unittest.mock import patch
import torch
from ks3_research.fp8_blas import matrix_product, Operand, product
from ks3_research.backward import ks3_blas_fused_unpack

@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip,'Native ROCm required')
class FP8Tests(unittest.TestCase):
    def test_gemm_matches_independent_quantized_oracle_including_padding(self):
        torch.manual_seed(169)
        for m,k,n in [(1,768,1024),(257,768,1024),(1024,257,768)]:
            a=torch.randn(m,k,device='cuda');b=torch.randn(n,k,device='cuda').T
            def quant(x):
                scale=x.abs().max().clamp_min(1e-12)/240
                return (x/scale).clamp(-240,240).to(torch.float8_e4m3fnuz).float(),scale
            qa,sa=quant(a);qb,sb=quant(b)
            want=(qa@qb)*(sa*sb);got=matrix_product(a,b)
            torch.testing.assert_close(got,want,atol=3e-5,rtol=3e-5)

    def test_every_forward_and_backward_gemm_is_native_fp8(self):
        torch.manual_seed(169);d,c,n=768,1024,257
        x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        down=(torch.randn(c,d,device='cuda')*.02).requires_grad_()
        up=(torch.randn(d,c,device='cuda')*.02).requires_grad_()
        coeff=torch.zeros(c,1,6,device='cuda',requires_grad=True)
        scale=torch.full((c,),.75,device='cuda',requires_grad=True)
        shift=torch.zeros(c,device='cuda',requires_grad=True)
        original=torch._scaled_mm;calls=[]
        def checked(a,b,*args,**kwargs):
            self.assertEqual(a.dtype,torch.float8_e4m3fnuz)
            self.assertEqual(b.dtype,torch.float8_e4m3fnuz)
            self.assertEqual(kwargs['out_dtype'],torch.float32)
            calls.append((a.shape,b.shape))
            return original(a,b,*args,**kwargs)
        with patch.object(torch,'_scaled_mm',checked):
            params=(x,down,up,coeff,scale,shift)
            y=ks3_blas_fused_unpack(*params,dynamic_rows=True,fp8_products=True)
            gradients=torch.autograd.grad(y,params,torch.randn_like(y)*.01)
        self.assertEqual(len(calls),6)
        self.assertTrue(all(g.isfinite().all() for g in (y,*gradients)))
        self.assertTrue(all(g.dtype==torch.float32 for g in gradients[1:]))

    def test_fused_decode_layouts_and_reuse_match_oracle(self):
        torch.manual_seed(393)
        codes=torch.randint(-127,128,(257,65),device='cuda',dtype=torch.int8)
        scales=torch.rand(257,3,device='cuda',dtype=torch.float16)*.01
        operand=Operand(codes,scales)
        decoded=codes.float()*scales.float().repeat_interleave(32,1)[:,:65]
        expected=(decoded/operand.scale).clamp(-240,240).to(torch.float8_e4m3fnuz).float()
        torch.testing.assert_close(operand.row[:257,:65].float(),expected,atol=0,rtol=0)
        torch.testing.assert_close(operand.col[:257,:65].float(),expected,atol=0,rtol=0)
        self.assertEqual(operand.row[257:].float().count_nonzero(),0)
        self.assertEqual(operand.col[:,65:].float().count_nonzero(),0)
        with patch('ks3_research.fp8_blas.quantize',side_effect=AssertionError('GEMM must not repack')):
            out=product(operand,operand,transpose_b=True)
            transposed=product(operand,operand,transpose_a=True)
        torch.testing.assert_close(out,(expected@expected.T)*operand.scale.square(),rtol=3e-5,atol=3e-5)
        torch.testing.assert_close(transposed,(expected.T@expected)*operand.scale.square(),rtol=3e-5,atol=3e-5)

    def test_zero_and_empty_inputs(self):
        a=torch.zeros(17,33,device='cuda');b=torch.randn(33,65,device='cuda')
        self.assertTrue(torch.equal(matrix_product(a,b),torch.zeros(17,65,device='cuda')))
        self.assertEqual(matrix_product(a[:0],b).shape,(0,65))

if __name__=='__main__':unittest.main()
