"""Surgical, CPU-testable dense-feedforward overlays of immutable PR393 code.

Only the two dense call sites, their FP8 producer/arguments, and dense cache
refresh are replaced. Attention, residuals, router, gates, head and loss remain
the original AST. The original source is never written.
"""
import ast
import copy
import hashlib
from pathlib import Path

SOURCE=Path(__file__).parent/'pr393/upstream/track_1_short/model/gpt.py'

def method(name):
    tree=ast.parse(SOURCE.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPT')
    return copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))

def forward_tree():
    tree=method('forward')
    class Replace(ast.NodeTransformer):
        calls=0;arguments=0;producer=0;cache=0
        def visit_Assign(self,node):
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            if names==['mlp_up_f8']:
                self.cache+=1;return None
            if names in (['fp8_args'],['parallel_args']):
                self.arguments+=1;return None
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_mlp':
                self.calls+=1
                slot='slot' if names==['mlp_out'] else 'k'
                node.value=ast.parse(f'self._matched_dense({slot},mlp_in,mlp_fcs[{slot}],mlp_projs[{slot}],fold_p,use_fp8,retained_mlp_all)').body[0].value
                return node
            return self.generic_visit(node)
        def visit_If(self,node):
            if any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='x_f8' for t in n.targets) for n in node.body):
                self.producer+=1;return None
            return self.generic_visit(node)
    replace=Replace();tree=replace.visit(tree)
    assert (replace.calls,replace.arguments,replace.producer,replace.cache)==(2,2,1,1)
    extra=ast.parse('retained_mlp_all = self.retained_mlp_bank.flatten(0,1).unbind(0) if self.retained_slots else None').body[0]
    pos=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mlp_all' for t in n.targets))
    tree.body.insert(pos,extra)
    assert not any(isinstance(n,ast.Attribute) and n.attr=='_mlp' for n in ast.walk(tree))
    return ast.fix_missing_locations(tree)

def refresh_tree(retained_count,dense_fp8=False):
    tree=method('_quantize_mlp_weights_and_scales')
    if retained_count:
        class Replace(ast.NodeTransformer):
            def visit_Name(self,n):return ast.copy_location(ast.Constant(retained_count),n) if n.id=='NUM_USED_MLP_SLOTS' else n
            def visit_Attribute(self,n):
                n=self.generic_visit(n)
                if n.attr=='mlp_bank':n.attr='retained_mlp_bank'
                return n
        tree=Replace().visit(tree)
    else:
        # Keep precisely the host counter and routed-expert delayed scales.
        tree.body=[n for n in tree.body if isinstance(n,ast.AugAssign) or
                   (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                    any(isinstance(x,ast.Attribute) and (x.attr.startswith('_moe_') or
                        (dense_fp8 and x.attr in ('_mlp_post_amax','_mlp_dpre_amax'))) for x in ast.walk(n)))]
        tree.body=ast.parse('e4m3_max=448.\ne5m2_max=57344.').body+tree.body
        assert len(tree.body)==(7 if dense_fp8 else 5)
    tree.body.append(ast.parse('self._refresh_dense_ks3()').body[0])
    return ast.fix_missing_locations(tree)

def compile_method(tree,namespace):
    module=ast.fix_missing_locations(ast.Module(body=[tree],type_ignores=[]))
    scope=dict(namespace)
    code=ast.unparse(module)
    filename=str(SOURCE.parent/('_matched_'+tree.name+'.py'))
    # Tracebacks/torch Dynamo can inspect the generated immutable source text.
    import linecache
    linecache.cache[filename]=(len(code),None,[line+'\n' for line in code.splitlines()],filename)
    exec(compile(code,filename,'exec'),scope)
    return scope[tree.name],hashlib.sha256(code.encode()).hexdigest()
