"""Create a KS3 MoE variant from the immutable PR367 retrieval model."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def prepare(destination):
    destination = Path(destination)
    upstream = ROOT / 'wr_reproduction/pr367/upstream'
    manifest = json.loads((upstream.parent / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((upstream / name).read_bytes()).hexdigest() == digest, name
    shutil.copytree(upstream, destination, ignore=shutil.ignore_patterns('__pycache__'))
    kernels = destination / 'track_1_short/perf/kernels'
    original_kernels = ROOT / 'wr_reproduction/pr393/upstream/track_1_short/perf/kernels'
    for name in ('moe_mlp.py', 'moe_g3.py', 'mlp_g3.py', 'anvil_opt.py'):
        shutil.copy2(original_kernels / name, kernels / name)
    path = destination / 'track_1_short/model/gpt.py'
    text = path.read_text()
    text = replace_once(text, 'NUM_LAYERS = 11', '''NUM_LAYERS = 11
MOE_LAYERS = (1, 3, 5, 6, 8)
MOE_INDEX = {layer: i for i, layer in enumerate(MOE_LAYERS)}
DENSE_LAYERS = (0, 2, 4, 9, 10)
DENSE_SLOT = {layer: i for i, layer in enumerate(DENSE_LAYERS)}
NUM_USED_DENSE_SLOTS = len(DENSE_LAYERS) + 1''')
    text = replace_once(text, 'PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = 8, 11',
                        'PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = 8, len(DENSE_LAYERS)')
    text = replace_once(text, 'NUM_MLP_SLOTS = 12', 'NUM_MLP_SLOTS = 8')
    text = replace_once(text,
        'frozenset(2 * layer + j for layer in NO_MLP_LAYERS for j in (0, 1))',
        'frozenset(2 * slot + j for slot in range(NUM_USED_DENSE_SLOTS, NUM_MLP_SLOTS) for j in (0, 1))')
    text = replace_once(text, '        late_attn_in = None\n        for i in range(self.num_layers):',
                        '        late_attn_in = None\n        moe_aux = 0.0\n        for i in range(self.num_layers):')
    text = replace_once(text, '            c_fc = mlp_fcs[i]\n            c_proj = mlp_projs[i]\n', '')
    before = '''            if use_fp8:
                # One quantize of the input, shared with the parallel MLP.
                x_f8 = quantize_dual_layout_fused(mlp_in.detach().view(-1, mlp_in.size(-1)), self._mlp_x_scale, fmt=torch.float8_e4m3fn)
                fp8_args = self._mlp_fp8_args(i, mlp_up_f8[i], x_f8, fold_p)
            else:
                fp8_args = None
            mlp_out = self._mlp(mlp_in, c_fc, c_proj, fp8_args)'''
    after = '''            dense = i not in MOE_INDEX
            if use_fp8 and (dense or i == PARALLEL_MLP_LAYER):
                x_f8 = quantize_dual_layout_fused(mlp_in.detach().view(-1, mlp_in.size(-1)), self._mlp_x_scale, fmt=torch.float8_e4m3fn)
            if dense:
                slot = DENSE_SLOT[i]
                fp8_args = self._mlp_fp8_args(slot, mlp_up_f8[slot], x_f8, fold_p) if use_fp8 else None
                mlp_out = self._mlp(mlp_in, mlp_fcs[slot], mlp_projs[slot], fp8_args)
            else:
                mlp_out, aux = self._moe(i, mlp_in, fold_p)
                moe_aux = moe_aux + aux'''
    text = replace_once(text, before, after)
    before = '        return self._loss(norm(x), input_seq, target_seq, mtp_weights, prefix_weight, schedule_cfg.sampled_loss)'
    after = '''        loss = self._loss(norm(x), input_seq, target_seq, mtp_weights, prefix_weight, schedule_cfg.sampled_loss)
        return loss + 0.01 * moe_aux if self.training else loss'''
    text = replace_once(text, before, after)
    text = replace_once(text,
        '        self._mlp_fold_p[:NUM_LAYERS - 1].copy_(post_lambdas[:NUM_LAYERS - 1])',
        '''        for layer, slot in DENSE_SLOT.items():
            if layer != NUM_LAYERS - 1:
                self._mlp_fold_p[slot].copy_(post_lambdas[layer])''')
    ast.parse(text)
    path.write_text(text)

    path = destination / 'track_1_short/tail_average.py'
    text = path.read_text()
    text = replace_once(text, 'DECONTRACTION_LABELS = ("embed", "lm_head", "mlp_bank", "qk_bank", "vo_bank")',
                        'DECONTRACTION_LABELS = ("embed", "lm_head", "mlp_bank", "qk_bank", "vo_bank", "moe_bank")')
    text = replace_once(text, 'TailAccumulator("tail-avg", ("vo_bank", "mlp_bank"),',
                        'TailAccumulator("tail-avg", ("vo_bank", "mlp_bank", "moe_bank"),')
    text = replace_once(text, '{"vo_bank": None, "mlp_bank": None}', '{"vo_bank": None, "mlp_bank": None, "moe_bank": None}')
    text = replace_once(text, 'TailAccumulator("bank-blend", ("qk_bank", "vo_bank", "mlp_bank"),',
                        'TailAccumulator("bank-blend", ("qk_bank", "vo_bank", "mlp_bank", "moe_bank"),')
    text = replace_once(text, '"mlp_bank": BANK_BLEND - RICHARDSON_W}',
                        '"mlp_bank": BANK_BLEND - RICHARDSON_W, "moe_bank": BANK_BLEND - RICHARDSON_W}')
    ast.parse(text)
    path.write_text(text)

    path = destination / 'train_gpt.py'
    text = path.read_text()
    text = replace_once(text, '    model: nn.Module = GPT(',
        '    from experiments.h100_pr367.model import install\n    KS3GPT = install()\n    model: nn.Module = KS3GPT(')
    text = replace_once(text, '            uncompiled_model.complete_yarn_tables()',
        '            uncompiled_model.prepare_ks3_eval()\n            uncompiled_model.complete_yarn_tables()')
    marker = '            # The clock is stopped: flush the log and collect the loop\'s garbage here.'
    text = replace_once(text, marker,
        '            print0(f"[wr-result] steps:{step} loss:{val_loss:.9f} seconds:{training_time_ms/1000:.9f}", console=True)\n'
        '            coefficients = uncompiled_model.ks3_geometry.detach()[..., :4].flatten(1)\n'
        '            print0(f"[ks3-geometry] coefficient_rms_by_layer:{coefficients.square().mean(1).sqrt().tolist()} nonzero_fraction_by_layer:{(coefficients != 0).float().mean(1).tolist()}", console=True)\n' + marker)
    ast.parse(text)
    path.write_text(text)

    # As in PR393, compare the graph with a grad-enabled eager call. A
    # separately compiled no_grad graph can round the router input differently
    # and flip top-one choices; that is not a CUDA replay correctness test.
    path = destination / 'track_1_short/perf/cuda_graphs/step_graphs.py'
    text = path.read_text()
    text = replace_once(text, '    @torch.no_grad()\n    def _self_check(self, cs: CapturedStep):',
                        '    def _self_check(self, cs: CapturedStep):')
    text = replace_once(text, '        replay, eager = cs.loss.item(), self._loss(cs).item()',
                        '        # PR393 MoE protocol: compare the same grad-enabled compiled graph.\n'
                        '        replay, eager = cs.loss.item(), self._loss(cs).item()')
    ast.parse(text)
    path.write_text(text)
    return manifest['commit']
