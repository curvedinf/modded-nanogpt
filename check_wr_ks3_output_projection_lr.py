"""CPU audit of KS3-only output projection rates and original ANVIL scalar staging."""
import argparse
from dataclasses import asdict
import ast
import copy
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from wr_reproduction.backend import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dense-features', type=int, choices=(512, 768, 1024, 1280, 1536, 2816), default=1024)
    parser.add_argument('--routed-features', type=int, choices=(512, 768, 1024, 1280, 1536), default=1024)
    parser.add_argument('--routed-multiplier', type=float, default=1.)
    parser.add_argument('--dense-multiplier', type=float, default=1.5)
    parser.add_argument('--geometry-multiplier', type=float, default=.125)
    parser.add_argument('--dense-geometry-multiplier', type=float)
    args = parser.parse_args()
    if not math.isfinite(args.geometry_multiplier) or args.geometry_multiplier < 0:
        parser.error('geometry multiplier must be finite and nonnegative')
    if not math.isfinite(args.routed_multiplier) or args.routed_multiplier <= 0:
        parser.error('routed multiplier must be finite and positive')
    if args.dense_geometry_multiplier is not None and (
            not math.isfinite(args.dense_geometry_multiplier) or args.dense_geometry_multiplier < 0):
        parser.error('dense geometry multiplier must be finite and nonnegative')
    torch.set_num_threads(1)
    install('pr393')
    from track_1_short import training
    from track_1_short.optim import anvil
    from wr_reproduction.ks3_moe import install as install_moe
    original_build = anvil.AnvilAndAdam._build_param_cfg
    spec = importlib.util.spec_from_file_location('audited_dense_overlay', args.module)
    overlay = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = overlay
    spec.loader.exec_module(overlay)
    tree = ast.parse(Path(training.__file__).read_text())
    base_table = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'param_table' for t in node.targets):
            base_table = eval(compile(ast.Expression(node.value), '<original-param-table>', 'eval'), training.__dict__)
    assert base_table is not None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update'
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'param_table'):
            base_table.update(eval(compile(ast.Expression(node.args[0]), '<original-param-table>', 'eval'), training.__dict__))
    recipe_base = json.loads(Path('configs/wr_reproduction/matched_ks3_all.json').read_text())
    recipe_base['geometry']['lr_mul'] = args.geometry_multiplier
    recipe_base['charts'] = args.routed_features
    checks = []
    features = args.dense_features
    routed_features = args.routed_features
    rate = args.routed_multiplier
    cases = [(arch, 1., None, None, requested) for arch in
             ('ks3_all','ks3_first_mlp','ks3_last_mlp','ks3_both_mlp')
             for requested in (None,2.,1.)]
    for architecture, routed, dense, dense_geometry, requested_output in cases:
        recipe = copy.deepcopy(recipe_base)
        recipe.update(architecture=architecture, projection_lr_multiplier=routed, dense_charts=features)
        if dense is not None:
            recipe['dense_projection_lr_multiplier'] = dense
        if dense_geometry is not None:
            recipe['dense_geometry_lr_multiplier'] = dense_geometry
        if requested_output is not None:
            recipe['ks3_output_projection_lr_multiplier']=requested_output
        output_factor=2. if requested_output is None else requested_output
        keep = dict(ks3_all=(), ks3_first_mlp=(0,), ks3_last_mlp=(3,), ks3_both_mlp=(0, 3))[architecture]
        parameters = {}
        def parameter(name, shape, **attributes):
            value = torch.nn.Parameter(torch.empty(shape, device='meta'))
            for key, item in attributes.items():
                setattr(value, key, item)
            parameters[name] = value
        parameter('mlp_bank', (8, 2, features, 768), reshape=(16, features, 768),
                  frozen_matrices=frozenset(2*s+j for s in (*keep, 5, 6, 7) for j in (0, 1)))
        parameter('moe_bank', (80, routed_features, 768), reshape=(80, routed_features, 768),
                  c_proj_matrices=frozenset(layer*16+8+i for layer in range(5) for i in range(8)))
        parameter('qk_bank',(48,256,768),reshape=(48,256,768))
        parameter('vo_bank',(16,768,768),reshape=(16,768,768))
        parameter('lm_head',(768,50304))
        parameter('embed',(50304,768))
        parameter('moe_router',(5,8,768))
        parameter('ks3_geometry', (5, 8, routed_features, 8))
        parameter('ks3_dense_geometry', (5, features, 8))
        if keep:
            parameter('retained_mlp_bank', (4, 2, 2816, 768), reshape=(8, 2816, 768),
                      frozen_matrices=frozenset(range(2*len(keep), 8)))
        captured = {}
        def factory(named_parameters, **kwargs):
            captured.update(copy.deepcopy(kwargs['param_table']))
            optimizer = anvil.AnvilAndAdam.__new__(anvil.AnvilAndAdam)
            optimizer.rank = 0
            optimizer.world_size = 1
            optimizer.param_table = kwargs['param_table']
            optimizer.param_cfgs = {}
            optimizer.anvil_defaults = dict(lr=.023, momentum=.95, beta2=.9, weight_decay=2.25)
            optimizer.adam_defaults = dict(lr=.008, eps=1e-10, weight_decay=.005)
            for name, value in named_parameters:
                optimizer._build_param_cfg(value, name)
            return optimizer
        training.AnvilAndAdam = factory
        anvil.AnvilAndAdam._build_param_cfg = original_build
        base = install_moe('bf16', recipe['geometry'], charts=routed_features, subspace_rank=1,
                           projection_lr_multiplier=routed)
        overlay.install(base, recipe)
        optimizer = training.AnvilAndAdam(parameters.items(), param_table=copy.deepcopy(base_table),
                                         scatter_order=['mlp_bank', 'moe_bank'], work_order=['moe_bank', 'mlp_bank'])
        for name, config in base_table.items():
            if name not in ('mlp_bank', 'moe_bank'):
                assert captured[name] == config, name
        dense_cfg = optimizer.param_cfgs[parameters['mlp_bank']]
        routed_cfg = optimizer.param_cfgs[parameters['moe_bank']]
        assert dense_cfg.lr_mul == max(1., features/768)**.5 * (routed if dense is None else dense)
        assert routed_cfg.lr_mul == max(1., routed_features/768)**.5 * routed
        assert dense_cfg.per_matrix_lr_mul == [0. if i//2 in (*keep, 5, 6, 7) else output_factor if i%2 else 1. for i in range(16)]
        assert routed_cfg.per_matrix_lr_mul == [output_factor if i in parameters['moe_bank'].c_proj_matrices else 1. for i in range(80)]
        for cfg in (dense_cfg, routed_cfg):
            assert (cfg.lr, cfg.weight_decay, cfg.momentum, cfg.beta2) == (.023, 2.25, .95, .9)
        for name in ('ks3_geometry', 'ks3_dense_geometry'):
            cfg = optimizer.param_cfgs[parameters[name]]
            expected_geometry = (dense_geometry if name == 'ks3_dense_geometry' and dense_geometry is not None
                                 else args.geometry_multiplier)
            assert (cfg.optim, cfg.comms, cfg.lr_mul, cfg.wd_mul, cfg.adam_betas, cfg.eps) == (
                'adam', 'replicated', expected_geometry, 0., (.9, .99), 1e-10)
            assert cfg.lr == .008
        retained_lr = None
        if keep:
            cfg = optimizer.param_cfgs[parameters['retained_mlp_bank']]
            assert captured['retained_mlp_bank'] == base_table['mlp_bank']
            assert cfg.lr_mul == (2816/768)**.5
            assert cfg.per_matrix_lr_mul == [1., 2.]*len(keep)+[0.]*(8-2*len(keep))
            retained_lr = cfg.lr_mul
        shared_labels=('qk_bank','vo_bank','lm_head','embed','moe_router')
        for label in shared_labels:
            param=parameters[label]
            original=anvil.AnvilAndAdam.__new__(anvil.AnvilAndAdam)
            original.rank=0;original.world_size=1;original.param_table=copy.deepcopy(base_table)
            original.param_cfgs={};original.anvil_defaults=optimizer.anvil_defaults;original.adam_defaults=optimizer.adam_defaults
            original_build(original,param,label)
            assert asdict(optimizer.param_cfgs[param])==asdict(original.param_cfgs[param]),label
        original_cfgs={}
        for label,param in parameters.items():
            if label not in ('mlp_bank','moe_bank','retained_mlp_bank'):continue
            original=anvil.AnvilAndAdam.__new__(anvil.AnvilAndAdam)
            original.rank=0;original.world_size=1;original.param_table=copy.deepcopy(captured)
            original.param_cfgs={};original.anvil_defaults=optimizer.anvil_defaults;original.adam_defaults=optimizer.adam_defaults
            source_label='mlp_bank' if label=='retained_mlp_bank' else label
            if label=='retained_mlp_bank':original.param_table['mlp_bank']=copy.deepcopy(base_table['mlp_bank'])
            original_build(original,param,source_label)
            old=asdict(original.param_cfgs[param]).copy();old['label']=label
            current=asdict(optimizer.param_cfgs[param]).copy()
            original_cfgs[label]=old['per_matrix_lr_mul']
            if label in ('mlp_bank','moe_bank') and output_factor!=2.:
                current.pop('per_matrix_lr_mul');old.pop('per_matrix_lr_mul')
            assert current==old,(architecture,requested_output,label)
        class Capture:
            def upload(self,fields,rates):self.fields=fields;self.rates=rates
        optimizer.banks={cfg.label:None for cfg in optimizer.param_cfgs.values() if cfg.optim=='anvil'}
        optimizer._param_by_label={cfg.label:param for param,cfg in optimizer.param_cfgs.items()}
        optimizer.fast_beta=.125;optimizer.fast_weight=.875;optimizer.bank_scalars=Capture()
        original_rates={label:optimizer.param_cfgs[optimizer._param_by_label[label]].lr for label in optimizer.banks}
        for schedule in (1.,.5,.01):
            for label in optimizer.banks:optimizer.param_cfgs[optimizer._param_by_label[label]].lr=original_rates[label]*schedule
            anvil.AnvilAndAdam.stage_bank_scalars(optimizer)
            for index,label in enumerate(optimizer.banks):
                cfg=optimizer.param_cfgs[optimizer._param_by_label[label]]
                expected=list(original_cfgs.get(label) or [1.]*cfg.chunk_size)
                if label=='mlp_bank':expected=[0. if i//2 in (*keep,5,6,7) else output_factor if i%2 else 1. for i in range(16)]
                if label=='moe_bank':expected=[output_factor if i%16>=8 else 1. for i in range(80)]
                assert optimizer.bank_scalars.rates[index]==[cfg.lr_mul*value*(original_rates[label]*schedule) for value in expected]
                assert optimizer.bank_scalars.fields[index]==(cfg.momentum,cfg.wd_mul*cfg.weight_decay*cfg.lr,.125,.875)
                assert all(rate==0. for rate,factor in zip(optimizer.bank_scalars.rates[index],expected) if factor==0.)
        for label in optimizer.banks:optimizer.param_cfgs[optimizer._param_by_label[label]].lr=original_rates[label]
        checks.append(dict(architecture=architecture, dense_features=features, routed_features=routed_features,
                           requested_output_multiplier=requested_output,effective_output_multiplier=output_factor,
                           dense_per_matrix_lr_multipliers=dense_cfg.per_matrix_lr_mul,
                           routed_per_matrix_lr_multipliers=routed_cfg.per_matrix_lr_mul,
                           retained_original_output_multiplier=2. if keep else None,
                           shared_parameter_configs_identical=True,only_target_matrix_rates_changed=True,
                           original_scalar_staging_verified=True,scalar_schedule_points=3,
                           requested_routed_multiplier=routed,
                           requested_dense_multiplier=dense, effective_dense_multiplier=dense_cfg.lr_mul,
                           effective_routed_multiplier=routed_cfg.lr_mul, retained_effective_multiplier=retained_lr,
                           geometry_multiplier=args.geometry_multiplier, geometry_adam_policy_verified=True,
                           dense_geometry_multiplier=(args.geometry_multiplier if dense_geometry is None else dense_geometry),
                           shared_optimizer_table_unchanged=True, target_output_rate_and_padding_verified=True))
    invalid_values=(0.,-1.,float('nan'),float('inf'),True,'1.0',None)
    for invalid in invalid_values:
        recipe=dict(recipe_base,ks3_output_projection_lr_multiplier=invalid)
        try:overlay.install(base,recipe)
        except ValueError as exc:assert 'KS3 output projection LR multiplier' in str(exc)
        else:raise AssertionError('Invalid KS3 output multiplier accepted')
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, gpu_initialized=False, checks=checks,
                  invalid_output_multipliers_rejected=len(invalid_values),
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  staged_module_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  original_optimizer_sha256=hashlib.sha256(Path(anvil.__file__).read_bytes()).hexdigest(),
                  scope='Only KS3 active output-projection matrix LR factors differ. Default absent/2x and candidate1x across all four architectures; original parameter config builder and scalar staging; shared configs, retained MLP2x, frozen padding, geometry, shape rates, rails and decay coefficients preserved.')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
