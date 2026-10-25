"""CPU candidate semantics and registered GPU lifetime/gradient checks."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch
from control393.sampling import SampledSoftmax,CandidateBuilder,host_copy_column,COPY_TOKEN_ID
from control393.continuation import HoldLast
from test_ks3_sampling import CPUStaging


class MetadataCPUStaging(CPUStaging):
    def fill_slot(self,*args):
        staged=super().fill_slot(*args)
        staged.slot=NS(candidates=staged.candidates)
        staged.num_candidates=len(staged.candidates)
        return staged


def exercise(device,gpu=False):
    schedule=NS(boundaries=[(0,4),(4,8),(8,20),(20,22),(22,24)],total_steps=24)
    counts=[[10240],[16384],[18432,18432,24576],[],[]]
    sampler=SampledSoftmax(schedule,50304,64,1024,0,1,torch.device(device),counts,host_metadata=True)
    sampler.counts=HoldLast(sampler.counts)
    prefix=(np.arange(50304,dtype=np.int64)+17)%50304;prefix[::13]=-1
    sampler.set_prefix_table(prefix)
    reference=CandidateBuilder(50304,24576,1024);reference.reset(0,1)
    errors=[];steps=[]
    weight=torch.randn(64,50304,device=device,dtype=torch.bfloat16,requires_grad=True) if gpu else torch.empty(0)
    try:
        for step in range(28):
            targets=torch.from_numpy((np.arange(1024,dtype=np.int64)*37+step*151)%50304)
            p=sampler.counts[step]
            expected=reference.build(p,targets.numpy(),prefix) if p else None
            if expected is not None:expected=[x.copy() for x in expected]
            sampler.prefetch(step,targets)
            sampler.upload(step,targets)
            sample=sampler.gather(step,weight)
            if p:
                assert sample.copy_column==int(np.searchsorted(expected[0],COPY_TOKEN_ID))
                torch.testing.assert_close(sample.candidate_ids.cpu(),torch.from_numpy(expected[0]),atol=0,rtol=0)
                torch.testing.assert_close(sample.target_pos[:1024].cpu(),torch.from_numpy(expected[1]),atol=0,rtol=0)
                torch.testing.assert_close(sample.prefix_pos[:1024].cpu(),torch.from_numpy(expected[2]),atol=0,rtol=0)
                assert sampler.builder.sweep_offset==reference.sweep_offset
                if gpu:
                    old_ids=(sample.vocab_pos>=0).nonzero().flatten()
                    assert sample.copy_column==int(sample.vocab_pos[COPY_TOKEN_ID])
                    x=torch.randn(256,64,device=device,dtype=torch.bfloat16,requires_grad=True)
                    upstream=torch.randn(256,p,device=device,dtype=torch.bfloat16)*.01
                    a=x@weight.index_select(1,sample.candidate_ids)
                    b=x@weight.index_select(1,old_ids)
                    ga=torch.autograd.grad(a,(x,weight),upstream)
                    gb=torch.autograd.grad(b,(x,weight),upstream)
                    # Queue comparisons without a host readback after backward;
                    # the next candidate upload must respect readers_done.
                    errors.extend([(a-b).abs().max(),*((u-v).abs().max() for u,v in zip(ga,gb))])
                    del x,upstream,a,b,ga,gb
                steps.append(dict(step=step,candidates=p,copy_column=sample.copy_column))
            else:
                assert sample is None
            sampler.mark_readers_done()
        assert sampler.uploaded is None and sampler.staging.pending is None
        assert sampler.builder.sweep_offset==reference.sweep_offset
        if errors:assert float(torch.stack(errors).max())==0.
    finally:
        sampler.staging.prep.shutdown(wait=True)
    return dict(passed=True,checks=steps,pinned_ring_reused=True,continuation_steps=4,
                ascending_ids_and_target_prefix_positions_exact=True,
                projection_and_input_weight_gradients_bitwise_equal=bool(gpu),
                next_upload_after_backward_events_preserved=True)


class SampledMetadataTests(unittest.TestCase):
    def test_exact_candidate_sequence_and_full_vocabulary_continuation(self):
        with patch('ks3_research.sampling.CandidateStaging',MetadataCPUStaging):
            result=exercise('cpu')
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['checks']),20)

    def test_missing_copy_fails_before_gpu_access(self):
        staged=NS(slot=NS(candidates=torch.tensor([1,3,50258])),num_candidates=3)
        with self.assertRaisesRegex(ValueError,'omitted the copy token'):host_copy_column(staged)


def run_device_checks(device):
    torch.manual_seed(393174)
    return exercise(device,gpu=True)


if __name__=='__main__':unittest.main()
