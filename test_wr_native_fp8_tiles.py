"""Tie native tile policy to independent paired evidence and preserve fallbacks."""
import json,unittest
from pathlib import Path
from wr_reproduction.native_fp8_tiles import projection_tiles,weight_gradient_tiles
from wr_reproduction.research_controller import validate_recipe

class NativeTileTests(unittest.TestCase):
    def test_only_measured_gains_are_selected(self):
        evidence=json.loads(Path('research_results/wr_reproduction/research/native-fp8-tiles.json').read_text())
        changed=0
        for case in evidence['cases']:
            fn=weight_gradient_tiles if case['role'].endswith('weight_gradient') else projection_tiles
            shape=(case['m'],case['k'],case['n']) if fn==weight_gradient_tiles else (case['m'],case['n'],case['k'])
            original=fn(*shape,True)
            self.assertEqual(original,tuple(case['baseline_config']))
            want=tuple(case['best']['config']) if case['best']['paired_speedup']>=1.03 else original
            self.assertEqual(fn(*shape,True,'mi300-v2'),want)
            changed+=want!=original
        self.assertEqual(changed,8)

    def test_original_operands_small_rows_and_unchecked_shapes_keep_existing_tiles(self):
        for rows in (128,1024,4096,16384,49152):
            for n,k in ((768,512),(512,768),(768,768),(768,1536),(1536,768),(768,1024),(1024,768)):
                original=(128,128,128,8,1,32) if n==768 else (64,128,128,4,1,16)
                self.assertEqual(projection_tiles(rows,n,k,False,'mi300-v2'),original)
                if rows<8192 or (k,n)!=(1024,768):
                    self.assertEqual(projection_tiles(rows,n,k,True,'mi300-v2'),original)
            for h in (512,768,1024,1280,1536):
                original=(64,128,64,4,1,16)
                self.assertEqual(weight_gradient_tiles(rows,h,768,False,'mi300-v2'),original)
                if rows<8192 or h!=1024:self.assertEqual(weight_gradient_tiles(rows,h,768,True,'mi300-v2'),original)

    def test_recipe_scope_and_unknown_policy(self):
        recipe=json.loads(Path('configs/wr_reproduction/research_fp8_prepared_tiles_v2.json').read_text())
        validate_recipe(recipe)
        for change in (dict(charts=1536),dict(precision='bf16'),dict(fp8_prepared_native_gather=False)):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
        for fn in (projection_tiles,weight_gradient_tiles):
            with self.assertRaises(ValueError):fn(16384,768,1024,True,'unknown')

if __name__=='__main__':unittest.main()
