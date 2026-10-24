import unittest
from unittest.mock import patch
import torch
from control393.attention256 import fused256_attention,Fused256SelfAttention
from control393.eval_attention import training256_or_bucketed_eval,BucketedEval256SelfAttention
from test_ks3_attention import direct_check,wrapper_check


class Attention256Tests(unittest.TestCase):
    def test_chunk_and_wrapper_gradients(self):
        direct_check('cpu',torch.float32,64,128,384,fused256_attention)
        wrapper_check('cpu',torch.float32,True,64,128,Fused256SelfAttention)

    def test_training_and_evaluation_use_the_requested_paths(self):
        self.assertIs(BucketedEval256SelfAttention.attention,training256_or_bucketed_eval)
        args=(None,)*6
        with patch('control393.attention256.fused256_attention',return_value='train') as train, \
             patch('control393.eval_attention.bucketed_attention',return_value='eval') as evaluate:
            with torch.enable_grad():self.assertEqual(training256_or_bucketed_eval(*args),'train')
            with torch.no_grad():self.assertEqual(training256_or_bucketed_eval(*args),'eval')
        train.assert_called_once_with(*args);evaluate.assert_called_once_with(*args)


if __name__=='__main__':unittest.main()
