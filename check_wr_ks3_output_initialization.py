"""Independent CPU math, initialization, RNG and scope checks for KS3 outputs."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from wr_reproduction.matched_study import atomic, sha


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().contiguous().numpy().tobytes()).hexdigest()


def fingerprint(model):
    excluded = {'mlp_bank', 'moe_bank', 'ks3_geometry', 'ks3_dense_geometry', 'retained_mlp_bank'}
    return {n: dict(shape=list(p.shape), sha256=tensor_sha(p))
            for n, p in model.named_parameters() if n not in excluded}


def scope_check(staged):
    base = Path('wr_reproduction/ks3_feedforward.py')
    before, after = ast.parse(base.read_text()), ast.parse(staged.read_text())

    class RemoveInitialization(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'initialize_output_projections':
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'output_init_gain' for t in node.targets):
                return None
            if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                   and t.value.id == 'self' and t.attr == 'ks3_output_projection_initialization'
                   for t in node.targets):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if any(isinstance(n, ast.Name) and n.id == 'output_init_gain' for n in ast.walk(node.test)):
                return None
            return self.generic_visit(node)

        def visit_keyword(self, node):
            if node.arg == 'ks3_output_projection_initialization':
                return None
            return self.generic_visit(node)

    stripped = RemoveInitialization().visit(after)
    assert ast.dump(before, include_attributes=False) == ast.dump(stripped, include_attributes=False), \
        'Change exceeds KS3 output initialization, validation and reporting'
    return dict(only_initialization_validation_and_reporting_changed=True,
                unchanged_model_kernel_optimizer_ast_sha256=hashlib.sha256(
                    ast.dump(before, include_attributes=False).encode()).hexdigest(),
                base_source_sha256=sha(base), staged_source_sha256=sha(staged))


def mathematical_audit():
    from ks3_research.vendor.ks3_reference import KS3
    rng = torch.Generator().manual_seed(172)
    inputs = torch.randn(37, 4, generator=rng, dtype=torch.float64)
    gradient = torch.randn(37, 4, generator=rng, dtype=torch.float64)
    w1 = torch.randn(8, 4, generator=rng, dtype=torch.float64)*.25
    initial_w2 = torch.randn(4, 8, generator=rng, dtype=torch.float64)/math.sqrt(8)
    checks = []
    for gain in (0., .1):
        w2 = initial_w2*gain
        cell = KS3(4, num_subspaces=8, dtype=torch.float64)
        with torch.no_grad():
            cell.down.weight.copy_(w1); cell.up.weight.copy_(w2)
            cell.coefficients.zero_(); cell.raw_scale.fill_(.75); cell.raw_shift.zero_()
        x = inputs.clone().requires_grad_()
        y = cell(x)
        (y*gradient).sum().backward()
        coordinates = inputs @ w1.T
        hidden = coordinates.relu().square()
        local = 2*coordinates.relu()*(gradient @ w2)
        t = 4*torch.tanh(coordinates/4)
        z = torch.tanh((torch.nn.functional.softplus(torch.tensor(.75, dtype=torch.float64))+1e-4)*t)
        bases = (2*t.pow(3)-3*t.square()+1, -2*t.pow(3)+3*t.square(),
                 t.pow(3)-2*t.square()+t, t.pow(3)-t.square(), 2*z.square()-1, 4*z.pow(3)-3*z)
        expected_coefficients = torch.stack([(local*b).sum(0) for b in bases], -1)
        for actual, expected in ((y, hidden@w2.T), (x.grad, local@w1),
                                 (cell.down.weight.grad, local.T@inputs),
                                 (cell.up.weight.grad, gradient.T@hidden),
                                 (cell.coefficients.grad, expected_coefficients)):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
        for value in (cell.raw_scale.grad, cell.raw_shift.grad):
            assert torch.count_nonzero(value) == 0
        for value in (x.grad, cell.down.weight.grad, cell.coefficients.grad):
            assert bool(torch.count_nonzero(value)) == (gain > 0)
        assert torch.count_nonzero(cell.up.weight.grad) > 0
        checks.append(dict(gain=gain, input_gradient_norm=x.grad.norm().item(),
                           input_projection_gradient_norm=cell.down.weight.grad.norm().item(),
                           spine_coefficient_gradient_norm=cell.coefficients.grad.norm().item(),
                           output_projection_gradient_norm=cell.up.weight.grad.norm().item(),
                           initial_warp_gradients_zero=True, analytical_gradients_passed=True))
    return dict(passed=True, checks=checks, validation=False, gpu_initialized=False,
                oracle='Independent FP64 matrix/ReLU-square/Hermite derivative formulas against the pinned reference',
                interpretation='Nonzero outputs enable input and spine-coefficient gradients immediately; warp gradients remain zero at zero coefficients. This is a gradient-flow check, not a performance result.')


def model_audit(args):
    from wr_reproduction.backend import install
    from wr_reproduction.matched_architecture import audit_model
    scope = scope_check(args.module)
    recipe = json.loads(args.recipe.read_text())
    recipe['architecture'] = args.architecture
    recipe['ks3_output_projection_init_gain'] = args.gain
    assert recipe['rank'] == 1 and recipe['charts'] == 1024
    install('pr393')
    from track_1_short.model.gpt import GPT
    torch.manual_seed(169)
    baseline = GPT(50257, 11, 6, 128, 768, 128, ngram_dim=768,
                   world_size=8, device=torch.device('cpu'), rank=0)
    shared = fingerprint(baseline)
    expected_rng = torch.get_rng_state().clone()
    dense_before = baseline.mlp_bank.detach().clone()
    routed_before = baseline.moe_bank.detach().view(5, 2, 8, 1024, 768)
    input_sha = tensor_sha(routed_before[:, 0])
    assert torch.count_nonzero(routed_before[:, 1]) == 0
    del routed_before, baseline
    gc.collect()
    import wr_reproduction
    spec = importlib.util.spec_from_file_location('wr_reproduction.ks3_feedforward', args.module)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    wr_reproduction.ks3_feedforward = module
    from wr_reproduction.ks3_moe import install as install_moe
    cls = install_moe('bf16', recipe['geometry'], charts=1024, subspace_rank=1, bf16_kernel_mode='tuned')
    invalid = (True, False, -.1, float('nan'), float('inf'), '0.1', None)
    for value in invalid:
        bad = dict(recipe, ks3_output_projection_init_gain=value)
        try:
            module.install(cls, bad)
        except ValueError as exc:
            assert 'initialization gain' in str(exc)
        else:
            raise AssertionError('Invalid initialization accepted: '+str(value))
    cls = module.install(cls, recipe)
    torch.manual_seed(169)
    model = cls(50257, 11, 6, 128, 768, 128, ngram_dim=768,
                world_size=8, device=torch.device('cpu'), rank=0)
    assert torch.equal(torch.get_rng_state(), expected_rng), 'Shared model/data RNG changed'
    assert shared == fingerprint(model), 'Nonfeedforward parameters changed'
    bank = model.moe_bank.detach().view(5, 2, 8, 1024, 768)
    assert tensor_sha(bank[:, 0]) == input_sha, 'Routed input projections changed'
    generator = torch.Generator().manual_seed(169)
    if args.gain:
        expected = torch.randn((5, 8, 1024, 768), generator=generator, dtype=torch.float32)*(args.gain/32)
        assert torch.equal(bank[:, 1], expected), 'Routed output initialization differs'
    else:
        assert torch.count_nonzero(bank[:, 1]) == 0
    for slot in range(8):
        if slot in model.retained_slots:
            retained = model.retained_slots.index(slot)
            assert torch.equal(model.retained_mlp_bank[retained], dense_before[slot])
            assert torch.count_nonzero(model.mlp_bank[slot]) == 0
        elif slot >= 5:
            assert torch.equal(model.mlp_bank[slot], dense_before[slot, :, :1024])
        else:
            assert torch.equal(model.mlp_bank[slot, 0], dense_before[slot, 0, :1024])
            expected_dense = (torch.randn((1024, 768), generator=generator, dtype=torch.float32)*(args.gain/32)
                              if args.gain else torch.zeros_like(model.mlp_bank[slot, 1]))
            assert torch.equal(model.mlp_bank[slot, 1], expected_dense)
    if model.retained_slots:
        assert torch.count_nonzero(model.retained_mlp_bank[len(model.retained_slots):]) == 0
    assert torch.count_nonzero(model.ks3_geometry[..., :6]) == 0
    assert torch.count_nonzero(model.ks3_dense_geometry[..., :6]) == 0
    assert model.mlp_bank.label == 'mlp_bank' and model.moe_bank.label == 'moe_bank'
    assert model.mlp_bank.reshape == (16, 1024, 768)
    assert model.mlp_bank.frozen_matrices == frozenset(
        2*s+j for s in (*model.retained_slots, 5, 6, 7) for j in (0, 1))
    pre_cast = {n: tensor_sha(p.bfloat16()) for n, p in model.named_parameters()
                if n in ('mlp_bank', 'moe_bank', 'retained_mlp_bank')}
    model.cast_matrix_weights_bf16()
    for name, expected_hash in pre_cast.items():
        value = getattr(model, name)
        assert value.dtype == torch.bfloat16 and tensor_sha(value) == expected_hash
    inventory = audit_model(model, recipe['architecture'])
    assert not torch.cuda.is_initialized()
    return dict(passed=True, gpu_initialized=False, architecture=args.architecture, gain=args.gain,
                shared_initialization_identical=True, global_rng_preserved=True,
                input_projections_identical=True, retained_mlps_and_padding_unchanged=True,
                exact_isolated_normal_draws_verified=True, original_bank_labels_shapes_and_bf16_cast_preserved=True,
                geometry_initialization_unchanged=True, invalid_gains_rejected=len(invalid),
                initialization_policy=model.ks3_output_projection_initialization, scope=scope,
                recipe_sha256=sha(args.recipe), cpu_auditor_sha256=sha(Path(__file__)), inventory=inventory)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--module', type=Path)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--architecture', choices=['ks3_all', 'ks3_first_mlp', 'ks3_last_mlp', 'ks3_both_mlp'], default='ks3_all')
    p.add_argument('--gain', type=float, default=.1)
    p.add_argument('--math-only', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    result = mathematical_audit() if args.math_only else model_audit(args)
    atomic(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}))


if __name__ == '__main__':
    main()
