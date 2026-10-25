"""CPU-only staged architecture and eight-rank dense FP8 state audit."""
import os
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import torch
sys.path.insert(0, str(Path.cwd()))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, default=Path('logs/wr-reproduction/matched-study/staging/dense-fp8'))
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    original_source = load('original_source_for_cpu_audit', Path('wr_reproduction/ks3_feedforward_source.py'))
    source = load('wr_reproduction.ks3_feedforward_source', stage / 'ks3_feedforward_source.py')
    source.SOURCE = original_source.SOURCE
    for retained in (0, 1, 2):
        assert ast.dump(original_source.refresh_tree(retained)) == ast.dump(source.refresh_tree(retained, False))
    assert ast.dump(original_source.forward_tree()) == ast.dump(source.forward_tree())
    load('wr_reproduction.ks3_dense_fp8', stage / 'ks3_dense_fp8.py')
    overlay = load('wr_reproduction.ks3_feedforward', stage / 'ks3_feedforward.py')
    original_install = overlay.install
    state_checks = []

    def audited_install(base, recipe):
        cls = original_install(base, recipe)
        original_init = cls.__init__

        def checked_init(self, *a, **kw):
            original_init(self, *a, **kw)
            from wr_reproduction.rank_state import RankStates
            from track_1_short.model.gpt import FP8_POST_HEADROOM, FP8_DPRE_HEADROOM, FP8_GRAD_SCALE
            assert recipe['dense_precision'] == 'fp8'
            dense_gs32 = getattr(self, 'ks3_dense_training_gs32', True)
            assert dense_gs32 is recipe.get('dense_training_gs32', True)
            assert tuple(self._mlp_post_scales.shape) == (5,)
            assert tuple(self._mlp_dpre_scales.shape) == (5,)
            assert tuple(self._mlp_post_amax.shape) == (5, 1)
            assert tuple(self._mlp_dpre_amax.shape) == (5, 1)
            assert self._mlp_up_f8.shape[0] == len(self.retained_slots)
            states = RankStates(self)
            required = {'_mlp_post_scales', '_mlp_post_amax', '_mlp_dpre_scales', '_mlp_dpre_amax'}
            assert required <= set(states.names)
            assert all(not name.startswith('ks3_dense_') for name in states.names)
            pointers = {name: self.get_buffer(name).data_ptr() for name in required}
            for rank in range(8):
                states.states[rank]['_mlp_post_amax'].fill_((rank + 1) * 2.)
                states.states[rank]['_mlp_dpre_amax'].fill_((rank + 1) * 4.)
                states.enter(rank)
                states.refresh_activations()
                states.leave(rank)
            for rank in range(8):
                for kind, value, gain, floor in [('post', (rank + 1) * 2., FP8_POST_HEADROOM / 448., 1e-6),
                                                  ('dpre', (rank + 1) * 4., FP8_DPRE_HEADROOM / 57344., 1e-12)]:
                    got = states.states[rank]['_mlp_' + kind + '_scales']
                    torch.testing.assert_close(got, torch.full_like(got, max(value * gain, floor)))
                    assert torch.count_nonzero(states.states[rank]['_mlp_' + kind + '_amax']) == 0
            assert pointers == {name: self.get_buffer(name).data_ptr() for name in required}
            self.rearm_fp8_bootstrap()
            assert torch.all(self._mlp_post_scales == 1.)
            assert torch.all(self._mlp_dpre_scales == FP8_GRAD_SCALE)
            assert not torch.count_nonzero(self._mlp_post_amax)
            assert not torch.count_nonzero(self._mlp_dpre_amax)
            keep = tuple(self.retained_slots)
            mapping = {slot: index for index, slot in enumerate((*keep, *(s for s in range(5) if s not in keep)))}
            assert sorted(mapping.values()) == list(range(5))
            for index, slot in enumerate(keep):
                assert mapping[slot] == index
            state_checks.append(dict(architecture=recipe['architecture'], retained_slots=list(keep),
                                     dense_scale_slots=mapping, rank_count=8,
                                     independent_delayed_scales=True, captured_addresses_preserved=True,
                                     original_headroom_and_reset=True,
                                     original_retained_fp8_scale_indices=True,
                                     dense_training_projection_gs32=dense_gs32,
                                     weight_caches_shared_across_ranks=True))
        cls.__init__ = checked_init
        return cls

    overlay.install = audited_install
    sys.argv = ['check_ks3_feedforward_cpu', '--recipe', args.recipe, '--out', args.out]
    runpy.run_module('wr_reproduction.check_ks3_feedforward_cpu', run_name='__main__')
    result = json.loads(Path(args.out).read_text())
    assert len(state_checks) == 1 and not torch.cuda.is_initialized()
    result.update(dense_fp8_state_checks=state_checks, bf16_default_refresh_ast_identical=True,
                  original_forward_transform_identical=True,
                  staged_source_hashes={name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                                        for name in ('ks3_feedforward.py', 'ks3_feedforward_source.py', 'ks3_dense_fp8.py')})
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, gpu_initialized=False, dense_fp8_state_checks=state_checks)))


if __name__ == '__main__':
    main()
