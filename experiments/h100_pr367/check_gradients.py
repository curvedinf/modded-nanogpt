"""Check PR367's standard gradient collector against the KS3 bank writes."""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'wr_reproduction/pr393/upstream'))
from experiments.h100_pr367.model import RoutedWithWeightGrad
from experiments.h100_ks3 import fp8


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(419)
    rows = torch.randn(1024, 512, device='cuda', dtype=torch.bfloat16).requires_grad_()
    router = (torch.randn(8, 512, device='cuda', dtype=rows.dtype) * .02).requires_grad_()
    post = torch.tensor(.8, device='cuda', requires_grad=True)
    w1 = (torch.randn(8, 1024, 512, device='cuda', dtype=rows.dtype) * .02).requires_grad_()
    w2 = torch.randn_like(w1).mul_(.02).requires_grad_()
    geometry = torch.zeros(8, 1024, 8, device='cuda')
    geometry[..., :4].normal_(0, .01)
    geometry[..., 6] = .75
    geometry.requires_grad_()
    us = w1.detach().float().abs().amax((1, 2)) / 448
    ws = w2.detach().float().abs().amax((1, 2)) / 448
    uf = (w1.detach().float() / us[:, None, None]).to(torch.float8_e4m3fn)
    wf = (w2.detach().transpose(1, 2).float() / ws[:, None, None]).to(torch.float8_e4m3fn).contiguous()
    caches = (uf, uf.transpose(1, 2).contiguous(), wf, wf.transpose(1, 2).contiguous())
    scalar = lambda value: torch.tensor([value], device='cuda')
    scales = (us, ws, scalar(8/448), scalar(2**-6), scalar(.05), scalar(0.), scalar(2**-6), scalar(0.))
    bank = torch.empty(16, 1024, 512, device='cuda', dtype=rows.dtype)
    flag = torch.zeros((), device='cuda', dtype=torch.int32)
    grad = torch.randn_like(rows) * .01

    def loss(rows, router, post, w1, w2, geometry):
        y, probs, counts = RoutedWithWeightGrad.apply(rows, router, post, w1, w2, geometry, caches, scales, bank, flag, 0)
        return (y * grad).sum() + .01 * probs.square().sum()

    parameters = (rows, router, post, w1, w2, geometry)
    reports = []
    reference = None
    for name, function in [('eager', loss), ('compiled', torch.compile(loss, fullgraph=True, dynamic=False))]:
        gradients = torch.autograd.grad(function(*parameters), parameters)
        torch.cuda.synchronize()
        assert all(torch.isfinite(value).all() for value in gradients)
        assert torch.equal(gradients[3], bank[:8])
        assert torch.equal(gradients[4], bank[8:])
        assert gradients[3].abs().sum() > 0 and gradients[4].abs().sum() > 0
        if reference is not None:
            for actual, expected in zip(gradients, reference):
                torch.testing.assert_close(actual, expected, rtol=.01, atol=.01)
        reference = [value.clone() for value in gradients]
        reports.append(dict(mode=name, bank_gradients_bitwise=True, all_finite=True))
    output = dict(passed=True, model_dim=512, reports=reports)
    Path(os.environ['KS3_GRAD_CHECK_OUTPUT']).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
