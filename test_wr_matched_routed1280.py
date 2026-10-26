"""CPU checks for registered oracle commands and frozen verification handoff."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import run_wr_matched_routed1280 as wrapper


class RoutedWidthTests(unittest.TestCase):
    def recipe(self):
        return wrapper.read(wrapper.ROOT / 'configs/wr_reproduction/matched_ks3_all_routed1280_normlr.json')

    def original_run(self):
        return dict(run_id='test-only', directory='/tmp/test-only',
                    command=['original-train-command'], source_hashes={'pinned': 'source'})

    def test_only_correctness_gets_routed_oracle(self):
        for kind in ('correctness', 'discovery', 'verification'):
            frozen = {'code_commit': 'frozen'} if kind == 'verification' else None
            with patch.object(wrapper, 'ORIGINAL_CREATE', return_value=self.original_run()) as create, \
                 patch.object(wrapper, 'atomic'):
                run = wrapper.create_run(self.recipe(), 269, kind, 'test-only', frozen)
                create.assert_called_once_with(self.recipe(), 269, kind, 'test-only', frozen)
                self.assertEqual(run['source_hashes'], {'pinned': 'source'})
                if kind == 'correctness':
                    self.assertEqual(run['command'][1], str(wrapper.ORACLE))
                    self.assertEqual(run['command'][2:4], ['--features', '1280'])
                    self.assertEqual(run['command'][-1], '/tmp/test-only/events.jsonl')
                    self.assertEqual(run['auxiliary_source_hashes'][str(wrapper.ORACLE.relative_to(wrapper.ROOT))], wrapper.sha(wrapper.ORACLE))
                else:
                    self.assertEqual(run['command'], ['original-train-command'])
                    self.assertNotIn('auxiliary_source_hashes', run)

    def test_scope_rejected_before_registration(self):
        for changes in (dict(architecture='mlp'), dict(charts=1024), dict(dense_charts=768),
                        dict(rank=2), dict(precision='fp8'), dict(routed_training_gs32=False)):
            recipe = copy.deepcopy(self.recipe()); recipe.update(changes)
            with patch.object(wrapper, 'ORIGINAL_CREATE') as create:
                with self.assertRaisesRegex(RuntimeError, 'restricted'):
                    wrapper.create_run(recipe, 168, 'correctness', 'test-only')
                create.assert_not_called()

    def test_oracle_source_change_blocks_queue(self):
        with patch.object(wrapper, 'read', return_value={'oracle_sha256': 'wrong'}), \
             patch.object(wrapper, 'run_queue') as queue:
            with self.assertRaisesRegex(RuntimeError, 'oracle changed'):
                wrapper.main()
            queue.assert_not_called()


if __name__ == '__main__':
    unittest.main()
