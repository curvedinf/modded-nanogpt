"""Independent auxiliary-phase timing must preserve continuation and LR horizon."""
import json
from pathlib import Path
import unittest
import torch
from ks3_research.planning import schedule_for,stages_for


class ScheduleTests(unittest.TestCase):
    def test_reallocated_phases_preserve_lr_tail_and_real_objective(self):
        config=json.loads(Path('configs/control393/ks3_target368_resume.json').read_text())
        original=schedule_for(config,1044,torch.device('cpu'))
        config['stage_durations']=[.228,.256,.456,.06]
        changed=schedule_for(config,1044,torch.device('cpu'))
        self.assertEqual(changed.boundaries,[(0,238),(238,505),(505,1033),(1033,1096),(1096,1116)])
        self.assertEqual(changed.total_steps,original.total_steps)
        self.assertEqual(changed.split_step,original.split_step)
        for step in (800,1000,1097,1116,9999):
            self.assertEqual(changed.get_lr(step),original.get_lr(step))
        self.assertEqual(changed.mtp_weights[504].numel(),2)
        self.assertEqual(changed.mtp_weights[505].tolist(),[1.])
        self.assertEqual(changed.prefix_weight(505),0.)
        self.assertEqual(changed.lookup(9999)[0],changed.stages[-1])
        self.assertEqual(changed.mtp_weights[9999].tolist(),[1.])

    def test_invalid_durations_are_rejected(self):
        config=dict(batch_tokens=[16384]*5)
        for durations in ([.2,.2,.2],[0.,.2,.2,.6],[.2,.2,.2,.2],[float('nan'),.2,.2,.6]):
            with self.subTest(durations=durations),self.assertRaises(ValueError):
                stages_for(dict(config,stage_durations=durations))


if __name__=='__main__':unittest.main()
