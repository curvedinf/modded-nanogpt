"""CPU audit of lower KS3 geometry momentum through unchanged original ANVIL."""
import argparse
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
from wr_reproduction.matched_study import sha
from types import SimpleNamespace


def adam_equation_audit():
    path=Path('wr_reproduction/pr393/upstream/track_1_short/optim/anvil.py')
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnvilAndAdam')
    nodes=[]
    for name in ('_adam_update_step','_adam_update'):
        node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
        node.decorator_list=[]
        nodes.append(node)
    namespace=dict(torch=torch,nn=torch.nn,Tensor=torch.Tensor,ParamConfig=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
    namespace['AnvilAndAdam']=SimpleNamespace(_adam_update_step=namespace['_adam_update_step'])
    generator=torch.Generator().manual_seed(169)
    initial=torch.randn(17,generator=generator,dtype=torch.float64)
    gradients=[torch.randn(17,generator=generator,dtype=torch.float64)*(.1+i*.03) for i in range(12)]
    checks=[]
    for beta1,beta2 in ((.9,.99),(.5,.99)):
        param=initial.clone();expected=initial.clone();lr=.008*.125;eps=1e-10
        state=dict(step=0,exp_avg=torch.zeros_like(param),exp_avg_sq=torch.zeros_like(param))
        owner=SimpleNamespace(rank=0,param_states={param:state},_step_size_t=torch.tensor(0.,dtype=torch.float64),_eff_wd_t=torch.tensor(0.,dtype=torch.float64))
        cfg=SimpleNamespace(adam_betas=(beta1,beta2),lr=.008,lr_mul=.125,comms='replicated',weight_decay=.005,wd_mul=0.,eps=eps)
        for step,gradient in enumerate(gradients,1):
            namespace['_adam_update'](owner,param,gradient,cfg)
            moment=sum((1-beta1)*beta1**(step-1-i)*g for i,g in enumerate(gradients[:step]))
            variance=sum((1-beta2)*beta2**(step-1-i)*g.square() for i,g in enumerate(gradients[:step]))
            size=lr*(1-beta2**step)**.5/(1-beta1**step)
            expected=expected-size*moment/(variance.sqrt()+eps)
            torch.testing.assert_close(param,expected,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(state['exp_avg'],moment,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(state['exp_avg_sq'],variance,rtol=1e-12,atol=1e-12)
            assert state['step']==step and owner._eff_wd_t.item()==0
        checks.append(dict(beta1=beta1,beta2=beta2,updates=12,passed=True,max_parameter_error=(param-expected).abs().max().item()))
    return dict(passed=True,source_sha256=sha(path),checks=checks,
                oracle='Immutable original Adam methods, with decorators removed only in the CPU harness, versus direct closed-form moment sums and independent parameter updates.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--staged-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dense-features', type=int, choices=(512, 768, 1024, 1280, 1536, 2816), default=1024)
    parser.add_argument('--routed-features', type=int, choices=(512, 768, 1024, 1280, 1536), default=1024)
    parser.add_argument('--routed-multiplier', type=float, default=1.)
    parser.add_argument('--dense-multiplier', type=float, default=1.5)
    parser.add_argument('--geometry-multiplier', type=float, default=.125)
    parser.add_argument('--projection-decay-multiplier', type=float, default=.5)
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
    import wr_reproduction
    for name in ('ks3_optimizer','ks3_moe'):
        module_spec=importlib.util.spec_from_file_location('wr_reproduction.'+name,args.staged_dir/(name+'.py'))
        module=importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name]=module
        module_spec.loader.exec_module(module)
        setattr(wr_reproduction,name,module)
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
    cases = [(arch, rate, None, None, betas) for arch in
             ('ks3_all','ks3_first_mlp','ks3_last_mlp','ks3_both_mlp')
             for betas in ([.9,.99], [.5,.99])]
    for architecture, routed, dense, dense_geometry, case_betas in cases:
        recipe = copy.deepcopy(recipe_base)
        recipe.update(architecture=architecture, projection_lr_multiplier=routed, dense_charts=features)
        if case_betas is None:
            recipe['geometry'].pop('betas',None)
        else:
            recipe['geometry']['betas']=case_betas
        expected_betas=(.9,.99) if case_betas is None else tuple(case_betas)
        expected_decay=1.
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
        assert dense_cfg.per_matrix_lr_mul == [0. if i//2 in (*keep, 5, 6, 7) else 2. if i%2 else 1. for i in range(16)]
        assert routed_cfg.per_matrix_lr_mul == [2. if i in parameters['moe_bank'].c_proj_matrices else 1. for i in range(80)]
        for cfg in (dense_cfg, routed_cfg):
            assert cfg.wd_mul == expected_decay
            assert (cfg.lr, cfg.weight_decay, cfg.momentum, cfg.beta2) == (.023, 2.25, .95, .9)
        for name in ('ks3_geometry', 'ks3_dense_geometry'):
            cfg = optimizer.param_cfgs[parameters[name]]
            expected_geometry = (dense_geometry if name == 'ks3_dense_geometry' and dense_geometry is not None
                                 else args.geometry_multiplier)
            assert (cfg.optim, cfg.comms, cfg.lr_mul, cfg.wd_mul, cfg.adam_betas, cfg.eps) == (
                'adam', 'replicated', expected_geometry, 0., expected_betas, 1e-10)
            assert cfg.lr == .008
        retained_lr = None
        if keep:
            cfg = optimizer.param_cfgs[parameters['retained_mlp_bank']]
            assert captured['retained_mlp_bank'] == base_table['mlp_bank']
            assert cfg.lr_mul == (2816/768)**.5
            assert cfg.wd_mul == base_table['mlp_bank'].get('wd_mul',1.)
            assert cfg.per_matrix_lr_mul == [1., 2.]*len(keep)+[0.]*(8-2*len(keep))
            retained_lr = cfg.lr_mul
        checks.append(dict(architecture=architecture, dense_features=features, routed_features=routed_features,
                           requested_geometry_betas=case_betas,
                           effective_dense_geometry_betas=list(optimizer.param_cfgs[parameters['ks3_dense_geometry']].adam_betas),
                           effective_routed_geometry_betas=list(optimizer.param_cfgs[parameters['ks3_geometry']].adam_betas),
                           effective_dense_decay_multiplier=dense_cfg.wd_mul,
                           effective_routed_decay_multiplier=routed_cfg.wd_mul,
                           retained_original_decay_preserved=True,
                           requested_routed_multiplier=routed,
                           requested_dense_multiplier=dense, effective_dense_multiplier=dense_cfg.lr_mul,
                           effective_routed_multiplier=routed_cfg.lr_mul, retained_effective_multiplier=retained_lr,
                           geometry_multiplier=args.geometry_multiplier, geometry_adam_policy_verified=True,
                           dense_geometry_multiplier=(args.geometry_multiplier if dense_geometry is None else dense_geometry),
                           shared_optimizer_table_unchanged=True, output_2x_and_padding_preserved=True))
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, gpu_initialized=False, checks=checks,
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  staged_source_hashes={name:hashlib.sha256((args.staged_dir/name).read_bytes()).hexdigest()
                      for name in ('ks3_optimizer.py','ks3_moe.py','ks3_feedforward.py')},
                  scope='Baseline and beta1=.5 KS3 geometry policy through immutable original ANVIL parameter-config builder in all four architectures. No runtime code, optimizer equations, shared policies or kernels changed.')
    result['adam_equation_audit']=adam_equation_audit()
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
