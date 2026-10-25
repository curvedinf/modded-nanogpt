"""Reject untested precision/packing combinations before GPU allocation."""
import ast,copy,json,unittest
from pathlib import Path
from wr_reproduction.research_controller import validate_recipe

class PreparedNativeTests(unittest.TestCase):
    def test_recipe_rejects_unsupported_combinations(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_native.json').read_text())
        validate_recipe(recipe)
        for change in (dict(precision='bf16'),dict(charts=512),dict(rank=2,charts=512),
                       dict(weight_packing='loop'),dict(fp8_prepared_native_weights='yes')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
    def test_prepared_activations_require_verified_weight_path(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_producers.json').read_text())
        validate_recipe(recipe)
        for change in (dict(fp8_prepared_native_weights=False),dict(fp8_prepared_native_activations='yes')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
    def test_native_dual_requires_both_prepared_parent_paths(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_dual.json').read_text())
        validate_recipe(recipe)
        for change in (dict(fp8_prepared_native_activations=False),dict(fp8_prepared_native_weights=False),dict(fp8_prepared_native_dual='yes')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
    def test_native_gather_requires_dual_parent(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_gather.json').read_text())
        validate_recipe(recipe)
        for change in (dict(fp8_prepared_native_dual=False),dict(fp8_prepared_native_gather='yes')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
    def test_gather_preserves_exact_pinned_kernel_arithmetic(self):
        old=ast.parse(Path('wr_reproduction/pr393/upstream/track_1_short/perf/kernels/moe_mlp.py').read_text())
        new=ast.parse(Path('wr_reproduction/ks3_native_gather.py').read_text())
        original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_grad_gather_quant_kernel')
        changed=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_grad_gather_native_kernel')
        added_args=[a.arg for a in changed.args.args[-2:]];self.assertEqual(added_args,['GN','GTN'])
        changed.args.args=changed.args.args[:-2];changed.name=original.name
        old_loop=next(n for n in original.body if isinstance(n,ast.For))
        new_loop=next(n for n in changed.body if isinstance(n,ast.For))
        self.assertEqual(len(new_loop.body),len(old_loop.body)+3)
        new_loop.body=new_loop.body[:-3]
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(changed,include_attributes=False))
    def test_direct_quantization_is_explicit_training_only_ablation(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_direct_quant.json').read_text())
        validate_recipe(recipe)
        self.assertFalse(recipe['training_gs32'])
        self.assertTrue(recipe['training_quantization_change']['coefficient_gs32'])
        self.assertEqual(recipe['validation_precision'],'bf16')
        self.assertEqual(recipe['validation_window_policy'],'accepted-control')
    def test_prepared_capacity_scope_and_parameter_accounting(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_c1536.json').read_text())
        validate_recipe(recipe);change=recipe['capacity_change']
        self.assertEqual(change['expert_projection_parameters_after'],80*1536*768)
        self.assertEqual(change['geometry_parameters_after'],5*8*1536*8)
        self.assertTrue(recipe['training_gs32']);self.assertEqual(recipe['validation_precision'],'bf16')
        for width in (512,768,1280,2048):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,charts=width))
    def test_native_tile_sweep_scope(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_native_fp8_tile_sweep.json').read_text())
        validate_recipe(recipe)
        for change in (dict(charts=1536),dict(fp8_prepared_native_gather=False),dict(precision='bf16')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
    def test_existing_recipes_keep_native_preparation_disabled(self):
        for name in ('research_fp8_batched_weights_vjp32.json','research_bf16_shared_fp8_amax.json'):
            recipe=json.loads((Path('configs/wr_reproduction')/name).read_text())
            self.assertFalse(recipe.get('fp8_prepared_native_weights',False));validate_recipe(recipe)

if __name__=='__main__':unittest.main()
