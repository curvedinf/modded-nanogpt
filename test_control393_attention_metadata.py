"""Boundary cache lifetime and batched layouts; GPU checks only via registry."""
from itertools import accumulate
import unittest
import torch
from control393.attention_metadata import (remember_boundaries,cached_boundaries,host_boundaries,
    paired_boundaries,batched_group_metadata,HostBoundaryStaging)
from control393.bucket_training import shape_groups,group_metadata
from ks3_research.attention import chunks_for


class AttentionMetadataTests(unittest.TestCase):
    def test_cache_is_immutable_and_version_checked(self):
        values=[0,17,529,529]
        tensor=remember_boundaries(torch.tensor(values),values)
        values[1]=29
        self.assertEqual(host_boundaries(tensor),(0,17,529,529))
        tensor.view(-1)[1]=31
        self.assertIsNone(cached_boundaries(tensor))
        self.assertEqual(host_boundaries(tensor),(0,31,529,529))

    def test_paired_cache_preserves_duplicates_and_rejects_stale_input(self):
        tensor=remember_boundaries(torch.tensor([0,17,529,529]),[0,17,529,529])
        paired=paired_boundaries(tensor)
        self.assertEqual(cached_boundaries(paired),(0,34,1058,1058))
        torch.testing.assert_close(paired,2*tensor,atol=0,rtol=0)
        tensor[1]=31
        paired=paired_boundaries(tensor)
        self.assertIsNone(cached_boundaries(paired))
        self.assertEqual(host_boundaries(paired),(0,62,1058,1058))

    def test_inference_tensor_never_uses_unverifiable_cache(self):
        with torch.inference_mode():
            tensor=remember_boundaries(torch.tensor([0,17]),[0,17])
            self.assertIsNone(cached_boundaries(tensor))
            self.assertEqual(host_boundaries(tensor),(0,17))

    def test_batched_metadata_keeps_uniform_and_irregular_layouts_exact(self):
        for window in (0,17,640,1408):
            chunks=chunks_for([0,17,17,1051,3741,7969,7969],window)
            offsets=[0,*accumulate(hi-lo for _,hi,lo in chunks)]
            expected=[group_metadata('cpu',chunks,ids,offsets,window,3)
                      for ids in shape_groups(chunks)]
            actual=list(batched_group_metadata('cpu',chunks,offsets,window,3))
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                for x,y in zip(a,b):
                    if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,atol=0,rtol=0)
                    else:self.assertEqual(x,y)


def run_device_checks(device):
    """Exercise ring reuse, paired layouts and invalidation before real shapes."""
    from control393.tiled_attention import async_tiled_or_bucketed_eval,tiled_or_bucketed_eval
    staging=HostBoundaryStaging(1024,16,device)
    for repeat in range(18):
        n=512+16*(repeat%3)
        boundaries=torch.tensor([0,17+repeat,n,n],dtype=torch.int32)
        uploaded=staging.upload(torch.zeros(n,dtype=torch.int32),torch.zeros(n,dtype=torch.int64),
                                boundaries,torch.zeros(2*n,dtype=torch.int32))
        cu=uploaded[2]
        assert cached_boundaries(cu)==tuple(boundaries.tolist())
        torch.testing.assert_close(cu.cpu(),boundaries,atol=0,rtol=0)
        torch.testing.assert_close(paired_boundaries(cu).cpu(),2*boundaries,atol=0,rtol=0)
    n,heads=529,3
    params=[torch.randn(n,heads,64,device=device,dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    upstream=torch.randn_like(params[0])
    cu=remember_boundaries(torch.tensor([0,17,289,n,n],device=device,dtype=torch.int32),[0,17,289,n,n])
    for mutate in (False,True):
        if mutate:
            cu[1]=31
            assert cached_boundaries(cu) is None
        a=async_tiled_or_bucketed_eval(*params,cu,137,.13)
        b=tiled_or_bucketed_eval(*params,cu,137,.13)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        ga=torch.autograd.grad(a,params,upstream);gb=torch.autograd.grad(b,params,upstream)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=0,rtol=0)
    return dict(case='host_metadata_lifetime',ring_uploads=18,paired=True,
                in_place_invalidation=True,forward_and_gradients_bitwise_equal=True)


if __name__=='__main__':unittest.main()
