import json
from pathlib import Path
import unittest
import torch
from wr_reproduction.contracts import audit,ROOT
from wr_reproduction.accumulation import RankAccumulator
from wr_reproduction.fp8 import FORMATS,encode_native,mm

class Contracts(unittest.TestCase):
    def test_sources_and_original_token_budgets(self):
        for version,steps,tokens in [('pr393',1053,290258944),('record360',1194,328663040)]:
            audit(version)
            c=json.loads((ROOT/version/'contract.json').read_text())
            self.assertEqual(c['total_steps'],steps)
            self.assertEqual(sum(s['global_tokens'] for s in c['steps']),tokens)
            for s in c['steps']:
                self.assertEqual(s['logical_rank_tokens']*s['microbatches'],s['global_tokens'])

    def test_rank_accumulation_preserves_auxiliary_objective(self):
        torch.manual_seed(393)
        w=torch.randn(4,3,requires_grad=True)
        inputs=[torch.randn(16,3)+r for r in range(8)]
        def objective(x):
            logits=x@w.T
            probabilities=logits.softmax(-1)
            counts=probabilities.argmax(-1).bincount(minlength=4).float()/len(x)
            return logits.square().mean()+.01*4*(counts*probabilities.mean(0)).sum()
        reference=torch.autograd.grad(sum(objective(x) for x in inputs)/8,w)[0]
        acc=RankAccumulator(128)
        for rank,x in enumerate(inputs):acc.add(rank,len(x),{'router':torch.autograd.grad(objective(x),w)[0]})
        torch.testing.assert_close(acc.finish()['router'],reference)
        with self.assertRaises(ValueError):acc.add(0,16,{'router':reference})
        with self.assertRaises(RuntimeError):RankAccumulator(128).finish()
        with self.assertRaises(ValueError):RankAccumulator(128).add(0,8,{'router':reference})

@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip,'Native ROCm test')
class NativeFP8(unittest.TestCase):
    def test_every_finite_original_fp8_value_is_preserved(self):
        for dtype in FORMATS:
            all_bytes=torch.arange(256,dtype=torch.int32,device='cuda').to(torch.uint8)
            original=all_bytes.view(dtype)
            mask=original.float().isfinite()
            restored=encode_native(original).float()*2
            torch.testing.assert_close(restored[mask],original.float()[mask],rtol=0,atol=0)

    def test_invalid_cache_values_are_not_silently_finite(self):
        for dtype,invalid in [(torch.float8_e4m3fn,127),(torch.float8_e5m2,124)]:
            a=torch.zeros(16,32,device='cuda').to(dtype)
            a.view(torch.uint8)[0,0]=invalid
            b=torch.ones(32,16,device='cuda').to(torch.float8_e4m3fn)
            result=mm(a,b,1.,1.)
            self.assertTrue(result[0].isnan().all())
            self.assertTrue(result[1:].isfinite().all())

    def test_native_fp8_gemms_against_quantized_fp32_oracle(self):
        torch.manual_seed(393)
        for adt,bdt in [(torch.float8_e4m3fn,torch.float8_e4m3fn),
                        (torch.float8_e5m2,torch.float8_e4m3fn),
                        (torch.float8_e4m3fn,torch.float8_e5m2),
                        (torch.float8_e5m2,torch.float8_e5m2)]:
            a=(torch.randn(128,256,device='cuda')*3).to(adt)
            b=(torch.randn(128,256,device='cuda')*2).to(bdt).T
            result=mm(a,b,.125,.25)
            reference=(a.float()@b.float())*(.125*.25)
            torch.testing.assert_close(result,reference,atol=2e-5,rtol=2e-5)

if __name__=='__main__':unittest.main()
