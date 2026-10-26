"""CPU audit of staged routed direct-BF16 KS3, without changing live sources."""
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


def load(name, path):
    spec = importlib.util.spec_from_file_location('wr_reproduction.' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    core = load('ks3_moe', args.stage / 'ks3_moe.py')
    packing = load('ks3_weight_pack', args.stage / 'ks3_weight_pack.py')
    dense = load('ks3_feedforward', args.stage / 'ks3_feedforward.py')
    from wr_reproduction.check_ks3_feedforward_cpu import main as audit_architecture
    sys.argv = ['audit_architecture', '--recipe', str(args.recipe), '--out', str(args.out)]
    audit_architecture()
    result = json.loads(args.out.read_text())
    recipe = json.loads(args.recipe.read_text())
    cls = core.install('bf16', charts=1024, bf16_kernel_mode='tuned')
    full = dense.install(cls, recipe)
    assert full.ks3_routed_training_gs32 is False and cls.ks3_routed_training_gs32 is True
    calls = []
    class CaptureWeights:
        def __getitem__(self, grid):
            def call(*values, **kwargs):
                assert kwargs['FP8'] is False
                calls.append(dict(gs32=values[8], batched=True))
            return call
    original_weights, original_pack = packing._weights, core.pack
    packing._weights = CaptureWeights()
    def pack(value):
        calls.append(dict(gs32=True, batched=False))
        return torch.empty_like(value)
    core.pack = pack
    checks = []
    try:
        model = torch.nn.Module()
        model.moe_bank = torch.nn.Parameter(torch.empty(80, 1024, 768, device='meta'))
        model.ks3_up = torch.empty(5, 8, 1024, 768, device='meta')
        model.ks3_down = torch.empty(5, 8, 768, 1024, device='meta')
        for flag in (True, False):
            model.ks3_routed_training_gs32 = flag
            for training in (True, False, True):
                model.train(training)
                core.BF16_TUNED_KERNELS = training
                calls.clear()
                cls.prepare_ks3_eval(model)
                expected = not training or flag
                assert len(calls) == (2 if training else 45)
                assert all(call['gs32'] is expected for call in calls)
                checks.append(dict(training=training, routed_training_gs32=flag,
                                   cache_gs32=expected, calls=len(calls)))
    finally:
        packing._weights, core.pack = original_weights, original_pack
    from torch._subclasses.fake_tensor import FakeTensorMode
    arity_checks = []
    for explicit in (False, True):
        with FakeTensorMode():
            rows = torch.empty(257, 768, dtype=torch.bfloat16, requires_grad=True)
            router = torch.empty(8, 768, dtype=torch.bfloat16, requires_grad=True)
            post = torch.empty((), dtype=torch.bfloat16, requires_grad=True)
            up = torch.empty(8, 1024, 768, dtype=torch.bfloat16)
            down = torch.empty(8, 768, 1024, dtype=torch.bfloat16)
            geometry = torch.empty(8, 1024, 8, requires_grad=True)
            bank = torch.empty(80, 1024, 768, dtype=torch.bfloat16)
            arguments = (rows, router, post, up, up, up, down, geometry, bank, 0)
            output, probabilities, counts = (core.RoutedKS3.apply(*arguments, False) if explicit
                                              else core.RoutedKS3.apply(*arguments))
            assert output.shape == rows.shape and probabilities.shape == (257, 8) and counts.shape == (8,)
            (output.sum() + probabilities.sum()).backward()
            for tensor in (rows, router, post, geometry):
                assert tensor.grad is not None and tensor.grad.shape == tensor.shape
            arity_checks.append(dict(explicit_direct_flag=explicit, fake_forward_backward_passed=True))
    oracle = load('check_ks3_routed_bf16_direct', args.stage / 'check_ks3_routed_bf16_direct.py')
    def direct(rows, up, down, geometry):
        return ((rows.float() @ up.float().T) @ down.float().T + 0*geometry.sum()).bfloat16()
    def gs32(rows, up, down, geometry, rank):
        assert rank == 1
        return ((rows.float() @ up.float().T) @ down.float() + 0*geometry.sum()).bfloat16()
    oracle.direct_reference, oracle.gs32_reference = direct, gs32
    for flag in (True, False):
        x = torch.randn(31, 12, dtype=torch.bfloat16, requires_grad=True)
        router = torch.randn(8, 12, dtype=torch.bfloat16, requires_grad=True)
        post = torch.tensor(.7, dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn(8, 16, 12, dtype=torch.bfloat16, requires_grad=True)
        down = torch.randn_like(up, requires_grad=True)
        geometry = torch.randn(8, 16, 8, requires_grad=True)
        output, probabilities, assignment = oracle.reference(x, router, post, up, down, geometry, flag)
        assert output.shape == x.shape and assignment.shape == (31,)
        (output.sum() + probabilities.sum()).backward()
        for tensor in (x, router, post, up, down, geometry):
            assert tensor.grad is not None and tensor.grad.shape == tensor.shape
    rejected = 0
    for invalid in (0, None, 'false'):
        try:
            dense.install(cls, dict(recipe, routed_training_gs32=invalid))
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Non-boolean routed flag accepted')
    for changes in (dict(rank=2, charts=512), dict(charts=512)):
        try:
            dense.install(cls, dict(recipe, **changes))
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Unverified routed direct shape accepted')
    unchanged = []
    for name, functions in [('ks3_moe.py', ('_pack', '_geometry', '_vjp', '_reduce', 'backward')),
                            ('ks3_weight_pack.py', ('_weights', 'refresh'))]:
        before = ast.parse((Path('wr_reproduction') / name).read_text())
        after = ast.parse((args.stage / name).read_text())
        for function in functions:
            def body(tree):
                return next(ast.dump(n, include_attributes=False) for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == function)
            assert body(before) == body(after), (name, function)
            unchanged.append(name + ':' + function)
    assert not torch.cuda.is_initialized()
    result.update(cache_policy_checks=checks, autograd_arity_checks=arity_checks,
                  oracle_projection_shape_check=True, oracle_numerical_check_requires_gpu=True,
                  unchanged_arithmetic=unchanged, invalid_flags_and_shapes_rejected=rejected,
                  gpu_initialized=False,
                  staged_source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in args.stage.glob('*.py')},
                  cpu_auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, architecture=result['architecture'], cache_policy_checks=len(checks),
                         autograd_arity_checks=len(arity_checks), gpu_initialized=False)))


if __name__ == '__main__':
    main()
