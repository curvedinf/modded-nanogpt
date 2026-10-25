"""Opt-in raw validation at the accepted PR393 control's attention state.

Training follows every original stage. A terminal stage-four evaluation needs
the stage-two/three rotary frequencies as well as their windows: YaRN's apply
operation is path dependent and cannot be undone by applying the inverse ratio.
Capture the actual live frequencies once, on the training stream. Restore only
these nonlearned buffers and the evaluation arguments, before pausing the clock.
"""
from dataclasses import replace
import hashlib
import json


WINDOWS = (640, 1408)


class AcceptedEvaluation:
    def __init__(self):
        self.snapshot = None
        self.captured_step = None

    def observe(self, manager, step):
        # No scalar device readback, synchronization, or full rotary-table copy.
        if self.snapshot is None and (manager.ws_short * 128, manager.ws_long * 128) == WINDOWS:
            self.snapshot = [(y.angular_freq.detach().clone(), y.attn_scale)
                             for y in manager.model.yarns]
            self.captured_step = step

    def prepare(self, manager):
        """Return actual forward arguments and an audit record; never edit weights."""
        cfg = manager.get_forward_args()
        original = (cfg.ws_short, cfg.ws_long)
        report = dict(policy='accepted-control', training_windows=list(original),
                      attention_windows=list(original), yarn_snapshot_step=self.captured_step,
                      restored=False, compatible=False, setup_on_training_clock=True)
        if self.snapshot is None:
            # Earlier stages cannot masquerade as a matched control evaluation.
            if original[0] >= WINDOWS[0] or original[1] >= WINDOWS[1]:
                raise RuntimeError('Accepted YaRN state was not captured from continuous training')
            return cfg, report
        if original != WINDOWS:
            if original[0] < WINDOWS[0] or original[1] < WINDOWS[1]:
                raise RuntimeError('Research schedule moved backwards after accepted YaRN capture')
            for yarn, (frequency, scale) in zip(manager.model.yarns, self.snapshot, strict=True):
                yarn.angular_freq.copy_(frequency)
                yarn.attn_scale = scale
                # Rewrite existing graph-owned buffers; no learned parameter changes.
                yarn._build_rows(0, yarn.max_seq_len)
            cfg = replace(cfg, ws_short=WINDOWS[0], ws_long=WINDOWS[1])
            report['restored'] = True
        # This tiny readback occurs only in terminal setup, on the original timer.
        digest = hashlib.sha256()
        for yarn, (frequency, scale) in zip(manager.model.yarns, self.snapshot, strict=True):
            digest.update(frequency.detach().cpu().contiguous().numpy().tobytes())
            digest.update(json.dumps([yarn.head_dim, yarn.paired, scale]).encode())
        report.update(attention_windows=list(WINDOWS), compatible=True,
                      yarn_snapshot_sha256=digest.hexdigest())
        return cfg, report
