"""CPU checks for diagnostic trace accounting and failure on missing evidence."""
import unittest
from wr_reproduction.profile_updates import summarize_trace


class ProfileTests(unittest.TestCase):
    def event(self,name,duration,start,category='kernel'):
        return dict(ph='X',cat=category,name=name,dur=duration,ts=start)

    def test_gpu_kernels_are_aggregated_without_cpu_or_copy_double_counting(self):
        rows=[self.event('a',10,0),self.event('a',5,5),self.event('b',20,12),
              self.event('host',1000,0,'cpu_op'),self.event('copy',100,0,'gpu_memcpy')]
        result=summarize_trace(dict(traceEvents=rows))
        self.assertEqual(result['kernel_count'],3)
        self.assertEqual(result['summed_kernel_duration_us'],35)
        self.assertEqual(result['gpu_activity_span_us'],32)
        self.assertEqual(result['top_kernels'][0]['name'],'b')
        self.assertEqual(result['top_kernels'][1]['count'],2)
        self.assertEqual(result['top_kernels'][1]['mean_us'],7.5)
        self.assertIn('not elapsed-time shares',result['warning'])
        self.assertAlmostEqual(sum(r['fraction_of_kernel_duration_sum'] for r in result['top_kernels']),1.)

    def test_missing_gpu_trace_does_not_claim_a_completed_profile(self):
        for rows in ([],[self.event('cpu',2,0,'cpu_op')],[self.event('zero',0,0)]):
            with self.assertRaises(RuntimeError):summarize_trace(dict(traceEvents=rows))

    def test_invalid_device_measurements_are_rejected(self):
        for duration in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):summarize_trace(dict(traceEvents=[self.event('bad',duration,0)]))
        with self.assertRaises(ValueError):summarize_trace(dict(traceEvents=[self.event('bad',1,float('nan'))]))


if __name__=='__main__':unittest.main()
