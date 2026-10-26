"""CPU audit of staged randomized KS3 geometry, including full-model isolation."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from wr_reproduction.matched_study import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = args.stage / 'ks3_feedforward.py'
    spec = importlib.util.spec_from_file_location('wr_reproduction.ks3_feedforward', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module.initialize_spines
    observed = []

    def initialize(routed, dense, rank, std):
        rng = torch.get_rng_state().clone()
        before = [value.detach().clone() for value in (routed, dense)]
        policy = original(routed, dense, rank, std)
        assert torch.equal(rng, torch.get_rng_state()), 'Global RNG advanced'
        generator = torch.Generator(device='cpu').manual_seed(torch.initial_seed())
        summaries = []
        for value, initial in zip((routed, dense), before):
            expected = initial.clone()
            if std:
                noise = torch.randn((*value.shape[:-1], 6*rank), generator=generator,
                                    dtype=torch.float32)
                expected[..., :6*rank] += noise * std
            assert torch.equal(value, expected), 'Seed-local coefficient sequence differs'
            assert torch.equal(value[..., 6*rank:], initial[..., 6*rank:]), 'Warp state changed'
            coefficients = value[..., :6*rank].detach()
            summaries.append(dict(shape=list(value.shape), mean=coefficients.mean().item(),
                                  std=coefficients.std().item(), max_abs=coefficients.abs().max().item(),
                                  sha256=hashlib.sha256(coefficients.contiguous().numpy().tobytes()).hexdigest()))
        observed.append(dict(policy=policy, rank=rank, summaries=summaries))
        return policy

    module.initialize_spines = initialize
    from wr_reproduction.check_ks3_feedforward_cpu import main as architecture_audit
    sys.argv = ['architecture_audit', '--recipe', str(args.recipe), '--out', str(args.out)]
    architecture_audit()
    result = json.loads(args.out.read_text())
    assert len(observed) == 1 and observed[0]['policy']['std'] == .05
    model_initialization = observed.pop()
    assert model_initialization['policy']['seed'] == 169
    for summary in model_initialization['summaries']:
        assert abs(summary['mean']) < .002 and .048 < summary['std'] < .052

    # Deterministic across repetitions, distinct across fresh seeds, and exact
    # default compatibility. Exercise rank-dependent coefficient/warp layout.
    checks = []
    for rank in (1, 2, 4):
        fingerprints = []
        for seed in (168, 168, 169):
            torch.manual_seed(seed)
            routed = torch.zeros(2, 8, 32, 6*rank+2)
            dense = torch.zeros(5, 32, 6*rank+2)
            for value in (routed, dense):
                value[..., 6*rank] = .75
            initialize(routed, dense, rank, .05)
            fingerprints.append(observed.pop()['summaries'][0]['sha256'])
        assert fingerprints[0] == fingerprints[1] and fingerprints[0] != fingerprints[2]
        initialize(routed, dense, rank, 0.)
        default = observed.pop()
        assert default['policy'] == dict(distribution='constant', std=0., seed=None)
        checks.append(dict(rank=rank, deterministic=True, seed_sensitive=True,
                           warp_preserved=True, global_rng_preserved=True, default_unchanged=True))
    recipe = json.loads(args.recipe.read_text())
    rejected = 0
    for invalid in (-.05, float('nan'), float('inf'), True, None, '0.05'):
        candidate = json.loads(json.dumps(recipe))
        candidate['geometry']['coefficient_init_std'] = invalid
        try:
            module.install(torch.nn.Module, candidate)
        except ValueError as error:
            assert 'standard deviation' in str(error)
            rejected += 1
        else:
            raise AssertionError('Invalid initialization accepted')
    before = ast.parse(Path('wr_reproduction/ks3_feedforward.py').read_text())
    after = ast.parse(source.read_text())
    names = ('dense_forward', 'dense_backward', 'DenseKS3', '_matched_dense',
             '_refresh_dense_ks3', 'cast_matrix_weights_bf16', 'build', 'optimizer')
    unchanged = {}
    for name in names:
        def body(tree):
            return next(ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
        assert body(before) == body(after), name
        unchanged[name] = hashlib.sha256(body(before).encode()).hexdigest()
    assert not torch.cuda.is_initialized()
    result.update(passed=True, gpu_initialized=False, model_initialization=model_initialization,
                  seed_and_layout_checks=checks, invalid_settings_rejected=rejected,
                  unchanged_arithmetic_optimizer_and_cache_ast_sha256=unchanged,
                  staged_source_sha256=sha(source), base_source_sha256=sha(Path('wr_reproduction/ks3_feedforward.py')),
                  cpu_auditor_sha256=sha(Path(__file__)), recipe_sha256=sha(args.recipe))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, architecture=result['architecture'],
                         seed_and_layout_checks=len(checks), invalid_settings_rejected=rejected,
                         unchanged_ast_functions=len(unchanged), gpu_initialized=False)))


if __name__ == '__main__':
    main()
