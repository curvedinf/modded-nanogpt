import unittest
from .researchctl import assess,compatible


class PruningTests(unittest.TestCase):
    def setUp(self):
        self.signature=dict(stage=0,candidates=10240,mtp=[1.,.5,.2],prefix=.25)
        self.reference=[dict(wall_seconds=40.,train_objective=10.,objective_signature=self.signature)]
        self.metric=dict(training_tokens=200000,wall_seconds=50.,train_objective=11.5,
                         objective_signature=self.signature)

    def test_requires_100k_and_two_bad_observations(self):
        first=assess(self.metric,self.reference,0)
        self.assertFalse(first['abort'])
        self.assertTrue(assess(self.metric,self.reference,first['bad_count'])['abort'])
        self.assertFalse(assess(dict(self.metric,training_tokens=99999),self.reference,1)['abort'])

    def test_improvement_resets_and_steps_do_not_define_comparison(self):
        result=assess(dict(self.metric,train_objective=10.5,step=99999),self.reference,1)
        self.assertEqual(result['bad_count'],0)
        self.assertFalse(result['abort'])

    def test_unmatched_objectives_and_missing_reference_do_not_prune(self):
        different=dict(self.signature,candidates=24576)
        self.assertFalse(assess(dict(self.metric,objective_signature=different),self.reference,9)['abort'])
        self.assertFalse(assess(dict(self.metric,wall_seconds=20),self.reference,9)['abort'])
        self.assertFalse(assess(dict(self.metric,wall_seconds=200),self.reference,9)['abort'])

    def test_auxiliary_loss_weights_are_checked(self):
        self.assertFalse(compatible(self.signature,dict(self.signature,mtp=[1.])))
        self.assertFalse(compatible(self.signature,dict(self.signature,prefix=.20)))


if __name__=='__main__':
    unittest.main()
