"""CPU full-model audit of a smaller seeded KS3 coefficient initialization."""
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
from wr_reproduction.matched_study import assert_fence, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    assert recipe['geometry']['coefficient_init_std'] == .01
    fence = assert_fence(recipe)
    original = ks3_feedforward.initialize_spines
    observed = []

    def initialize(routed, dense, rank, std):
        assert std == .01
        before = [value.detach().clone() for value in (routed, dense)]
        rng = torch.get_rng_state().clone()
        policy = original(routed, dense, rank, std)
        assert torch.equal(rng, torch.get_rng_state()), 'Global RNG advanced'
        generator = torch.Generator(device='cpu').manual_seed(torch.initial_seed())
        summaries = []
        for value, initial in zip((routed, dense), before):
            noise = torch.randn((*value.shape[:-1], 6*rank), generator=generator,
                                dtype=torch.float32)
            expected = initial.clone()
            expected[..., :6*rank] += noise * .01
            assert torch.equal(value, expected), 'Seed-local coefficient sequence differs'
            assert torch.equal(value[..., 6*rank:], initial[..., 6*rank:]), 'Warp state changed'
            assert torch.count_nonzero(initial[..., :6*rank]) == 0
            coefficients = value[..., :6*rank].detach()
            assert abs(coefficients.mean().item()) < .0004
            assert .0096 < coefficients.std().item() < .0104
            ratio = coefficients.double().square().mean().sqrt() / (noise * .05).double().square().mean().sqrt()
            assert abs(ratio.item() - .2) < 1e-7
            summaries.append(dict(shape=list(value.shape), mean=coefficients.mean().item(),
                                  std=coefficients.std().item(), max_abs=coefficients.abs().max().item(),
                                  rms_ratio_to_std005=ratio.item(),
                                  sha256=hashlib.sha256(coefficients.contiguous().numpy().tobytes()).hexdigest()))
        observed.append(dict(policy=policy, rank=rank, summaries=summaries))
        return policy

    ks3_feedforward.initialize_spines = initialize
    from wr_reproduction.check_ks3_feedforward_cpu import main as architecture_audit
    sys.argv = ['architecture_audit', '--recipe', str(args.recipe), '--out', str(args.out)]
    architecture_audit()
    result = json.loads(args.out.read_text())
    assert len(observed) == 1 and observed[0]['policy'] == dict(
        distribution='normal', std=.01, seed=169,
        generator='isolated-cpu-generator-routed-then-dense-v1')
    assert not torch.cuda.is_initialized()
    result.update(model_initialization=observed[0], global_rng_preserved=True,
                  coefficient_sequence_exact=True, raw_warp_state_preserved=True,
                  recipe_sha256=sha(args.recipe), cpu_auditor_sha256=sha(Path(__file__)),
                  runtime_source_sha256=sha(Path(ks3_feedforward.__file__)),
                  backend_hash=fence['backend_hash'],
                  decision='CPU configuration audit only; no validation or convergence conclusion')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, std=.01, gpu_initialized=False,
                         global_rng_preserved=True, coefficient_sequence_exact=True)))


if __name__ == '__main__':
    main()
