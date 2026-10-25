import unittest
import torch
from wr_reproduction.accumulation import OriginalGradientAccumulator

class Accumulation(unittest.TestCase):
    def test_original_rank_local_adam_rounding_and_tied_embedding_merge(self):
        torch.manual_seed(393)
        inputs=[{k:[torch.randn(*shape).bfloat16() for _ in range(8)] for k,shape in
            [('lm',(5,3)),('embed',(3,5)),('bank',(2,3))]} for _ in range(2)]
        acc=OriginalGradientAccumulator(['lm','embed'])
        transpose_add=lambda src,dst:dst.add_(src.T)
        for step in range(2):
            acc.begin(step==1)
            for rank in range(8):acc.add(rank,{k:rows[rank] for k,rows in inputs[step].items()},
                tied_pair=('lm','embed'),transpose_add=transpose_add)
            actual=acc.finish()
            want_bank=(torch.stack(inputs[step]['bank']).float().sum(0)/8).bfloat16()
            torch.testing.assert_close(actual['bank'],want_bank,atol=0,rtol=0)
            if step==0:self.assertEqual(set(actual),{'bank'})
        per_rank=[]
        for rank in range(8):
            lm=inputs[0]['lm'][rank]+inputs[1]['lm'][rank]
            embed=inputs[0]['embed'][rank]+inputs[1]['embed'][rank]
            per_rank.append(lm+embed.T)
        want=(torch.stack(per_rank).float().sum(0)/8).bfloat16()
        torch.testing.assert_close(actual['lm'],want,atol=0,rtol=0)
        self.assertNotIn('embed',actual)
        self.assertTrue(all(not d for d in acc.pending))
    def test_missing_and_duplicate_ranks_fail(self):
        acc=OriginalGradientAccumulator([])
        with self.assertRaises(RuntimeError):acc.finish()
        acc.add(0,{'p':torch.ones(1)})
        with self.assertRaises(ValueError):acc.add(0,{'p':torch.ones(1)})
if __name__=='__main__':unittest.main()
