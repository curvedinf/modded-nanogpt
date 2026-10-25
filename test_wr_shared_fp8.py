"""CPU guards for measured FP8 tile selection and legacy fallback scope."""
import json
from pathlib import Path
import unittest
from wr_reproduction.shared_fp8_tiles import select_tiles,select_tiles_v2,select_packed_head_tiles,select_packed_shared_tiles


class SharedFP8Tests(unittest.TestCase):
    def test_packed_shared_pipelines_match_all_ten_paired_cases(self):
        path=Path(__file__).parent/'research_results/wr_reproduction/research/packed-shared-pipeline-sweep.json'
        evidence=json.loads(path.read_text())
        self.assertEqual(len(evidence['cases']),10)
        self.assertTrue(evidence['paired_timing'])
        for case in evidence['cases']:
            m,n,k=case['m'],case['n'],case['k']
            self.assertGreater(case['best']['paired_speedup'],1.05)
            self.assertEqual(select_packed_shared_tiles(m,n,k,(k,1),(1,k)),tuple(case['best']['config']))
            self.assertIsNone(select_packed_shared_tiles(m,n,k,(k,1),(1,k),True))
            self.assertIsNone(select_packed_shared_tiles(m,n,k,(1,m),(1,k)))

    def test_packed_shared_scope_excludes_head_narrow_small_and_weight_gradients(self):
        for m,n,k in ((16384,10240,768),(49152,768,14336),(10240,768,4096),
                      (2816,768,49152),(16384,384,768),(1024,2816,768)):
            self.assertIsNone(select_packed_shared_tiles(m,n,k,(k,1),(1,k)))

    def test_packed_head_pipeline_matches_all_eight_independent_cases(self):
        path=Path(__file__).parent/'research_results/wr_reproduction/research/packed-head-tile-sweep.json'
        evidence=json.loads(path.read_text())
        self.assertEqual(len(evidence['cases']),8)
        self.assertTrue(evidence['paired_timing'])
        for case in evidence['cases']:
            m,n,k=case['m'],case['n'],case['k']
            self.assertGreater(case['best']['paired_speedup'],1.05)
            self.assertEqual(select_packed_head_tiles(m,n,k,(k,1),(1,k)),tuple(case['best']['config']))
            self.assertIsNone(select_packed_head_tiles(m,n,k,(k,1),(1,k),True))
            self.assertIsNone(select_packed_head_tiles(m,n,k,(1,m),(1,k)))

    def test_packed_head_policy_leaves_producers_and_other_products_unchanged(self):
        for m,n,k in ((16384,2816,768),(49152,768,2816),(10240,768,4096),
                      (2816,768,49152),(16384,384,768),(16384,2304,768),
                      (16384,768,1536),(1024,10240,768)):
            self.assertIsNone(select_packed_head_tiles(m,n,k,(k,1),(1,k)))

    def test_selected_tiles_follow_independently_recorded_sweep(self):
        path=Path(__file__).parent/'research_results/wr_reproduction/research/shared-fp8-tile-sweep.json'
        evidence=json.loads(path.read_text())
        for case in evidence['cases']:
            with self.subTest(case=case['name']):
                m,n,k=case['m'],case['n'],case['k']
                self.assertEqual(select_tiles(m,n,k,(k,1),(1,k)),tuple(case['best']['config']))

    def test_v2_selects_clear_paired_gains_and_retains_other_v1_tiles(self):
        path=Path(__file__).parent/'research_results/wr_reproduction/research/shared-fp8-expanded-tile-sweep.json'
        evidence=json.loads(path.read_text())
        self.assertEqual(evidence['baseline_dispatch'],'mi300-v1')
        self.assertTrue(evidence['paired_timing'])
        self.assertEqual(len(evidence['cases']),20)
        for case in evidence['cases']:
            with self.subTest(case=case['name']):
                m,n,k=case['m'],case['n'],case['k'];best=case['best']
                expected=tuple(best['config']) if best['paired_speedup']>=1.05 else select_tiles(m,n,k,(k,1),(1,k))
                self.assertEqual(select_tiles_v2(m,n,k,(k,1),(1,k)),expected)

    def test_grouped_small_and_unsupported_layouts_keep_legacy_dispatch(self):
        for m,n,k,astride,bstride,grouped in (
                (16384,1024,768,(768,1),(1,768),True),
                (256,768,1024,(1024,1),(1,1024),False),
                (1024,16,768,(768,1),(1,768),False),
                (1024,768,16,(16,1),(1,16),False),
                (16384,768,768,(1,16384),(1,768),False),
                (16384,768,768,(768,1),(768,1),False)):
            for selector in (select_tiles,select_tiles_v2):
                self.assertIsNone(selector(m,n,k,astride,bstride,grouped))


if __name__=='__main__':unittest.main()
