"""CPU model and precise admission-guard audit for staged rank-two1536 KS3."""
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
import wr_reproduction
from wr_reproduction.matched_study import sha


def guard(source):
    install = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    return next(n.test for n in install.body if isinstance(n, ast.If)
                and 'charts * subspace_rank' in ast.unparse(n.test)
                and 'precision' in ast.unparse(n.test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-module', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = Path('wr_reproduction/ks3_moe.py')
    before, after = base.read_text(), args.core_module.read_text()
    old = "if subspace_rank!=1 and (precision!='bf16' or charts*subspace_rank!=1024 or bf16_kernel_mode!='tuned'):\n        raise ValueError('Higher rank requires1024 projection features and tuned BF16 training')"
    new = "if subspace_rank!=1 and (precision!='bf16' or not (charts*subspace_rank==1024 or (subspace_rank==2 and charts==768)) or bf16_kernel_mode!='tuned'):\n        raise ValueError('Higher rank requires1024 features or rank-two1536, and tuned BF16 training')"
    assert before.count(old) == 1 and after == before.replace(old, new)
    tests = [compile(ast.Expression(guard(s)), '<higher-rank-guard>', 'eval') for s in (before, after)]
    changed = []
    count = 0
    for precision in ('bf16', 'fp8'):
        for mode in ('tuned', 'legacy'):
            for rank in (1, 2, 4):
                for features in (512, 768, 1024, 1280, 1536):
                    scope = dict(precision=precision, bf16_kernel_mode=mode,
                                 subspace_rank=rank, charts=features//rank)
                    rejected = [eval(code, {}, scope) for code in tests]
                    count += 1
                    if rejected[0] != rejected[1]:
                        assert rejected == [True, False]
                        changed.append(dict(precision=precision, mode=mode, rank=rank, features=features))
    assert changed == [dict(precision='bf16', mode='tuned', rank=2, features=1536)]
    recipe = json.loads(args.recipe.read_text())
    assert recipe['rank'] == 2 and recipe['charts'] == recipe['dense_charts'] == 768
    spec = importlib.util.spec_from_file_location('wr_reproduction.ks3_moe', args.core_module)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    wr_reproduction.ks3_moe = core
    from wr_reproduction.check_ks3_feedforward_cpu import main as model_audit
    sys.argv = ['model_audit', '--recipe', str(args.recipe), '--out', str(args.out)]
    model_audit()
    result = json.loads(args.out.read_text())
    assert not torch.cuda.is_initialized()
    result.update(rank=2, charts=768, guard_cases=count, newly_admitted=changed,
                  only_admission_guard_changed=True, base_source_sha256=sha(base),
                  staged_source_sha256=sha(args.core_module), recipe_sha256=sha(args.recipe),
                  cpu_auditor_sha256=sha(Path(__file__)),
                  decision='CPU shape/isolation audit only; independent dense and routed numerical gates and full graph warmup are required')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, rank=2, features=1536, guard_cases=count,
                         gpu_initialized=False, only_admission_guard_changed=True)))


if __name__ == '__main__':
    main()
