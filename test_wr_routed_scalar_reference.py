"""Independent CPU scalar derivatives at original PR393 precision boundaries."""
import os
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
import torch
from check_wr_ks3_routed768 import apply_gate_post


class ScalarReferenceTests(unittest.TestCase):
    def test_forward_and_all_derivatives_match_analytic_precision_contract(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(172)
        cases = 0
        for count in (1, 257, 4096):
            for dtype in (torch.bfloat16, torch.float32):
                values = torch.randn(count, 768, dtype=torch.bfloat16,
                                     generator=generator).requires_grad_()
                gate = torch.rand(count, dtype=torch.float32, generator=generator).requires_grad_()
                post = torch.tensor(.7, dtype=dtype, requires_grad=True)
                gradient = torch.randn(values.shape, dtype=torch.bfloat16, generator=generator)*.02
                output = apply_gate_post(values, gate, post)
                scale = gate.detach()*post.detach().float()
                self.assertTrue(torch.equal(output, values.detach()*scale.bfloat16()[:, None]))
                actual = torch.autograd.grad(output, (values, gate, post), gradient)
                ds = (values.detach().float()*gradient.float()).sum(-1)
                expected = ((gradient.float()*scale.bfloat16().float()[:, None]).bfloat16(),
                            ds*post.detach().float(), (ds*gate.detach()).sum().to(dtype))
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
                cases += 1
        self.assertEqual(cases, 6)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
