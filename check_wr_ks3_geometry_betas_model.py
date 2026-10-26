"""CPU scope, immutable Adam-equation and full-model audits of staged KS3 betas."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['HIP_VISIBLE_DEVICES']=''
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import wr_reproduction
from wr_reproduction.matched_study import sha


def unchanged_model_ast(staged):
    result={}
    for name, model in (('ks3_moe.py','KS3GPT'),('ks3_feedforward.py','FullKS3GPT')):
        hashes=[]
        for path in (Path('wr_reproduction')/name,staged/name):
            tree=ast.parse(path.read_text())
            outside=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='install']
            inside=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name==model)
            hashes.append([hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                           for n in (ast.Module(body=outside,type_ignores=[]),inside)])
        assert hashes[0]==hashes[1], 'Model or kernel AST changed: '+name
        result[name+':outside_install']=hashes[0][0]
        result[name+':model_class']=hashes[0][1]
    return result


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
    for beta2 in (.99,.999):
        param=initial.clone();expected=initial.clone();lr=.008*.125;beta1=.9;eps=1e-10
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
        checks.append(dict(beta2=beta2,updates=12,passed=True,max_parameter_error=(param-expected).abs().max().item()))
    return dict(passed=True,source_sha256=sha(path),checks=checks,
                oracle='Immutable original Adam methods, with decorators removed only in the CPU harness, versus direct closed-form moment sums and independent parameter updates.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--staged-dir',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    unchanged=unchanged_model_ast(args.staged_dir)
    equation=adam_equation_audit()
    for name in ('ks3_optimizer','ks3_moe','ks3_feedforward'):
        spec=importlib.util.spec_from_file_location('wr_reproduction.'+name,args.staged_dir/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
        spec.loader.exec_module(module);setattr(wr_reproduction,name,module)
    from wr_reproduction.check_ks3_feedforward_cpu import main as model_audit
    sys.argv=['model_audit','--recipe',str(args.recipe),'--out',str(args.out)]
    model_audit()
    result=json.loads(args.out.read_text());assert not torch.cuda.is_initialized()
    result.update(unchanged_ast_sha256=unchanged,adam_equation_audit=equation,
                  recipe_sha256=sha(args.recipe),cpu_auditor_sha256=sha(Path(__file__)),
                  base_source_hashes={n:sha(Path('wr_reproduction')/n) for n in ('ks3_optimizer.py','ks3_moe.py','ks3_feedforward.py')},
                  staged_source_hashes={n:sha(args.staged_dir/n) for n in ('ks3_optimizer.py','ks3_moe.py','ks3_feedforward.py')},
                  scope='Unchanged model/kernel AST and initialization; original Adam update equations verified independently. Geometry parameter betas alone may differ.')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,gpu_initialized=False,unchanged_model_ast=True,original_adam_equation_checks=2)))


if __name__=='__main__':main()
