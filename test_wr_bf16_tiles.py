"""CPU guards tying BF16-v2 choices to independently measured paired evidence."""
import json
from pathlib import Path
import unittest
from wr_reproduction.bf16_tiles import select_projection_tiles as select
from wr_reproduction.research_controller import validate_recipe


class BF16TilesTests(unittest.TestCase):
    def test_v2_selects_only_measured_projection_gains_of_three_percent(self):
        evidence=json.loads((Path(__file__).parent/'research_results/wr_reproduction/research/ks3-bf16-expanded-tiles.json').read_text())
        changed=0
        for case in evidence['cases']:
            if case['role']=='weight_gradient':
                self.assertLess(case['best']['paired_speedup'],1.)
                continue
            args=(case['routed_capacity'],case['n'],case['k'],case['row_major_weight'])
            baseline=select(*args)
            expected=tuple(case['best']['config']) if case['best']['paired_speedup']>=1.03 else baseline
            self.assertEqual(select(*args,'mi300-v2'),expected,case['role'])
            changed+=expected!=baseline
        self.assertEqual(changed,4)

    def test_original_v1_and_unsupported_v2_shapes_remain_stable(self):
        for m in (4096,16384,49152):
            for n,k in ((768,512),(512,768),(768,768),(768,1536),(1536,768)):
                for row in (False,True):
                    if m<=32768:old=(64,128,128,4,1,16) if n==768 else (128,128,64,8,1,32)
                    elif n==768:old=(128,128,64,8,1,32) if row else (64,128,64,4,1,16)
                    else:old=(64,128,64,4,1,16) if row else (64,128,128,4,1,16)
                    self.assertEqual(select(m,n,k,row),old)
                    self.assertEqual(select(m,n,k,row,'mi300-v2'),old)
        for n,k in ((1024,768),(768,1024)):
            for row in (False,True):self.assertEqual(select(1024,n,k,row,'mi300-v2'),select(1024,n,k,row))

    def test_recipe_rejects_unchecked_precision_width_and_rank(self):
        recipe=json.loads((Path(__file__).parent/'configs/wr_reproduction/research_bf16_shared_fp8_amax.json').read_text())
        recipe['bf16_gemm_tuning']='mi300-v2';validate_recipe(recipe)
        validate_recipe(dict(recipe,rank=2,charts=512))
        for change in (dict(charts=1536),dict(rank=4,charts=256),dict(precision='fp8'),dict(bf16_kernel_mode='legacy')):
            with self.assertRaises(ValueError):validate_recipe(dict(recipe,**change))
        with self.assertRaises(ValueError):select(16384,1024,768,False,'unknown')


if __name__=='__main__':unittest.main()
