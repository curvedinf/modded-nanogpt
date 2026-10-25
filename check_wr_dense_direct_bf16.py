"""CPU preflight for staged direct-BF16 dense KS3 and unchanged defaults."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name in ('ks3_moe', 'ks3_feedforward'):
        path = args.stage / (name + '.py')
        spec = importlib.util.spec_from_file_location('wr_reproduction.' + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    from wr_reproduction.check_ks3_feedforward_cpu import main as audit_architecture
    sys.argv = ['audit_architecture', '--recipe', str(args.recipe), '--out', str(args.out)]
    audit_architecture()
    result = json.loads(args.out.read_text())
    recipe = json.loads(args.recipe.read_text())
    from wr_reproduction import ks3_moe as core, ks3_dense_fp8
    from wr_reproduction.ks3_feedforward import install, DenseKS3
    calls = []
    class CapturePack:
        def __getitem__(self, grid):
            def call(*values, **kwargs):
                calls.append(dict(gs32=kwargs.get('GS32', True), shape=list(values[1].shape)))
            return call
    original_pack, original_fp8_refresh = core._pack, ks3_dense_fp8.refresh_weights
    core._pack = CapturePack()
    fp8_calls = []
    ks3_dense_fp8.refresh_weights = lambda model, exact: fp8_calls.append(exact)
    cache_checks = []
    try:
        for precision in ('bf16', 'fp8'):
            for flag in (True, False):
                candidate = copy.deepcopy(recipe)
                candidate.update(dense_precision=precision, dense_training_gs32=flag)
                cls = install(torch.nn.Module, candidate)
                model = torch.nn.Module()
                model.mlp_bank = torch.nn.Parameter(torch.empty(8, 2, 1024, 768, device='meta'))
                model._mlp_ks3_up = torch.empty(5, 1024, 768, device='meta')
                model._mlp_ks3_down = torch.empty(5, 768, 1024, device='meta')
                model.mlp_quantize_calls = 1
                for training in (True, False):
                    model.train(training)
                    calls.clear(); fp8_calls.clear()
                    cls._refresh_dense_ks3(model)
                    expected = precision == 'fp8' or not training or flag
                    assert len(calls) == 2*(5-len(cls.retained_slots))
                    assert all(call['gs32'] is expected for call in calls)
                    assert len(fp8_calls) == (1 if precision == 'fp8' else 0)
                    cache_checks.append(dict(precision=precision, dense_training_gs32=flag,
                                             training=training, bf16_cache_gs32=expected,
                                             active_dense_slots=5-len(cls.retained_slots),
                                             fp8_refresh_preserved=True))
    finally:
        core._pack, ks3_dense_fp8.refresh_weights = original_pack, original_fp8_refresh
    from torch._subclasses.fake_tensor import FakeTensorMode
    arity_checks = []
    for explicit in (False, True):
        with FakeTensorMode():
            rows = torch.empty(257, 768, dtype=torch.bfloat16, requires_grad=True)
            up = torch.empty(1024, 768, dtype=torch.bfloat16, requires_grad=True)
            down = torch.empty_like(up, requires_grad=True)
            geometry = torch.empty(1024, 8, requires_grad=True)
            packed_up = torch.empty_like(up, requires_grad=False)
            packed_down = torch.empty(768, 1024, dtype=torch.bfloat16)
            arguments = (rows, up, down, packed_up, packed_down, geometry, 0)
            output = DenseKS3.apply(*arguments, False) if explicit else DenseKS3.apply(*arguments)
            assert output.shape == rows.shape
            output.sum().backward()
            for tensor in (rows, up, down, geometry):
                assert tensor.grad is not None and tensor.grad.shape == tensor.shape
            arity_checks.append(dict(explicit_direct_flag=explicit, fake_forward_backward_passed=True))
    # Check the new oracle's projection orientation without invoking its GPU
    # nonlinear reference. This verifies shapes only, not numerical accuracy.
    oracle_tree = ast.parse((args.stage / 'check_ks3_feedforward_bf16_direct.py').read_text())
    reference_node = next(n for n in oracle_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reference')
    class FakeNonlinearity:
        def __getitem__(self, grid):
            return lambda coords, codes, scales, warp, shift, hidden, *a, **kw: hidden.zero_()
    import triton as tr
    namespace = dict(torch=torch, tr=tr, pack_spine=lambda value: (None, None),
                     _raw_upstream_spine=FakeNonlinearity(),
                     surrogate=lambda coords, geometry, rank: coords + 0*geometry[:, 0])
    exec(compile(ast.Module(body=[reference_node], type_ignores=[]), '<oracle-shape-check>', 'exec'), namespace)
    with FakeTensorMode():
        rows = torch.empty(257, 768, dtype=torch.bfloat16, requires_grad=True)
        up = torch.empty(1024, 768, dtype=torch.bfloat16, requires_grad=True)
        down = torch.empty(768, 1024, dtype=torch.bfloat16, requires_grad=True)
        geometry = torch.empty(1024, 8, requires_grad=True)
        output = namespace['reference'](rows, up, down, geometry)
        assert output.shape == rows.shape
        output.sum().backward()
        for tensor in (rows, up, down, geometry):
            assert tensor.grad is not None and tensor.grad.shape == tensor.shape
    assert not torch.cuda.is_initialized()
    result.update(cache_policy_checks=cache_checks, autograd_arity_checks=arity_checks,
                  oracle_projection_shape_check=True, oracle_numerical_check_requires_gpu=True,
                  gpu_initialized=False,
                  staged_source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in args.stage.glob('*.py')},
                  cpu_auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, architecture=result['architecture'],
                         cache_policy_checks=len(cache_checks), fake_autograd_checks=len(arity_checks),
                         gpu_initialized=False)))


if __name__ == '__main__':
    main()
