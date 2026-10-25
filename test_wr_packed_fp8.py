"""Independent exhaustive byte-neighbor checks for packed FP8 decoding."""
import random
import unittest
from wr_reproduction.packed_fp8_decode import decode_word


def oracle(word,e5):
    result=0
    for index in range(4):
        byte=(word>>(8*index))&255
        invalid=(byte&127)>=124 if e5 else (byte&127)==127
        result|=(128 if invalid else (0 if byte==128 else byte))<<(8*index)
    return result


class PackedDecodeTests(unittest.TestCase):
    def test_all_two_byte_patterns_and_neighbor_borrows(self):
        for e5 in (False,True):
            for pair in range(65536):
                for word in (pair|(pair<<16),pair|((pair^65535)<<16)):
                    self.assertEqual(decode_word(word,e5),oracle(word,e5))

    def test_each_position_with_critical_neighbors_and_random_words(self):
        for e5 in (False,True):
            for neighbor in (0,1,0x7b,0x7c,0x7e,0x7f,0x80,0x81,0xfb,0xfc,0xfe,0xff):
                repeated=neighbor*0x01010101
                for position in range(4):
                    for byte in range(256):
                        word=(repeated&~(255<<(position*8)))|(byte<<(position*8))
                        self.assertEqual(decode_word(word,e5),oracle(word,e5))
            rng=random.Random(169)
            for _ in range(65536):
                word=rng.getrandbits(32)
                self.assertEqual(decode_word(word,e5),oracle(word,e5))

    def test_recipe_policy_is_explicit_and_validated(self):
        import json
        from pathlib import Path
        from wr_reproduction.research_controller import validate_recipe
        recipe=json.loads((Path(__file__).parent/'configs/wr_reproduction/research_bf16_projection_v2.json').read_text())
        validate_recipe(recipe)
        validate_recipe(dict(recipe,shared_fp8_decode='packed4'))
        validate_recipe(dict(recipe,shared_fp8_decode='packed4-selected'))
        with self.assertRaises(ValueError):validate_recipe(dict(recipe,shared_fp8_decode='unknown'))

    def test_selection_tracks_paired_evidence_and_preserves_fallback(self):
        import json
        from pathlib import Path
        from wr_reproduction.packed_fp8_decode import select_packed
        evidence=json.loads((Path(__file__).parent/'research_results/wr_reproduction/research/packed-fp8-decode.json').read_text())
        for row in evidence['timing']:
            m,n,k=(row[x] for x in ('m','n','k'))
            self.assertEqual(select_packed(m,n,k,(k,1),(1,k)),row['speedup']>=1.03)
            self.assertFalse(select_packed(m,n,k,(k,1),(1,k),True))
            self.assertFalse(select_packed(m,n,k,(1,m),(1,k)))
        self.assertFalse(select_packed(257,2816,768,(768,1),(1,768)))


if __name__=='__main__':unittest.main()
