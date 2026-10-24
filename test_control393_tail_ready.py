"""Early tail use must require every buffer and restore the live raw iterate."""
from types import SimpleNamespace
import unittest
import torch
from control393.continuation import ContinuingTails


class TailReadyTests(unittest.TestCase):
    def tails(self):
        tails=ContinuingTails.__new__(ContinuingTails)
        tails.accumulators=[SimpleNamespace(labels=('moe_bank','mlp_bank'),seeded={'moe_bank'}),
                            SimpleNamespace(labels=('moe_bank',),seeded={'moe_bank'})]
        return tails

    def test_every_accumulator_label_is_required(self):
        tails=self.tails();self.assertFalse(tails.ready())
        with self.assertRaisesRegex(RuntimeError,'before every accumulator'):
            with tails.evaluation_weights(True):pass
        with tails.evaluation_weights(False) as name:self.assertEqual(name,'raw')
        tails.accumulators[0].seeded.add('mlp_bank');self.assertTrue(tails.ready())
        tails.accumulators[1].seeded.clear();self.assertFalse(tails.ready())

    def test_temporary_average_restores_raw_state_on_error(self):
        tails=self.tails();tails.accumulators[0].seeded.add('mlp_bank')
        parameters={'moe_bank':torch.nn.Parameter(torch.arange(12.).view(3,4)),
                    'mlp_bank':torch.nn.Parameter(torch.arange(12.,dtype=torch.bfloat16).view(3,4))}
        originals={name:p.detach().clone() for name,p in parameters.items()}
        addresses={name:p.data_ptr() for name,p in parameters.items()}
        tails._param=lambda name:parameters[name]
        def ship():
            with torch.no_grad():
                for p in parameters.values():p.add_(3.)
        tails.ship=ship
        with self.assertRaisesRegex(ValueError,'evaluation failure'):
            with tails.evaluation_weights(True) as name:
                self.assertEqual(name,'tail_average')
                self.assertFalse(torch.equal(parameters['moe_bank'],originals['moe_bank']))
                raise ValueError('evaluation failure')
        for name,p in parameters.items():
            self.assertEqual(p.data_ptr(),addresses[name])
            torch.testing.assert_close(p,originals[name],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
