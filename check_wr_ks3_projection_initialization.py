"""CPU full-model projection fingerprints for rank comparisons at equal width."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from wr_reproduction import ks3_feedforward
from wr_reproduction.matched_study import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = ks3_feedforward.install
    captured = []

    def install(base, recipe):
        cls = original(base, recipe)
        initialize = cls.__init__
        def record(self, *a, **kw):
            initialize(self, *a, **kw)
            row = dict(seed=torch.initial_seed(), rank=recipe['rank'], projections={}, geometry={})
            for name in ('mlp_bank', 'moe_bank'):
                value = getattr(self, name).detach()
                row['projections'][name] = dict(shape=list(value.shape), elements=value.numel(),
                    sha256=hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
            for name in ('ks3_geometry', 'ks3_dense_geometry'):
                value = getattr(self, name).detach()
                assert torch.count_nonzero(value[..., :6*recipe['rank']]) == 0
                row['geometry'][name] = dict(shape=list(value.shape), elements=value.numel(), zero_coefficients=True)
            captured.append(row)
        cls.__init__ = record
        return cls

    ks3_feedforward.install = install
    from wr_reproduction.check_ks3_feedforward_cpu import main as model_audit
    sys.argv = ['model_audit', '--recipe', str(args.recipe), '--out', str(args.out)]
    model_audit()
    assert len(captured) == 1 and captured[0]['seed'] == 169
    result = json.loads(args.out.read_text())
    result.update(projection_initialization=captured[0], recipe_sha256=sha(args.recipe),
                  cpu_auditor_sha256=sha(Path(__file__)), gpu_initialized=torch.cuda.is_initialized())
    assert result['gpu_initialized'] is False
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, rank=captured[0]['rank'], projection_fingerprints=True, gpu_initialized=False)))


if __name__ == '__main__':
    main()
