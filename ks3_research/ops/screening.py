"""Dynamic loss/time screening, independent of model turns and GPU libraries.

Training references use the latest compatible nonfuture observation per run,
within 15 seconds, then the best of those runs. Full validation uses the best
canonical measurement already achieved by that elapsed time (a model can stop
and retain its score). No step alignment, future samples, or interpolation.
"""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

POLICY_VERSION = 'dynamic-best-below5-one-sided-0.01-v1'
TRAIN_DEFINITION = 'mi300-softcap-mtp-prefix-ce-v1'
VAL_DEFINITION = 'fineweb-first10485760-canonical-softcap23-final-yarn-v1'
WINDOWS = ((128,384), (384,896), (640,1408), (640,1408), (768,1664))
SEQUENCES = (896,2048,3072,3072,3072)
DEFAULT_BATCHES = (16384,32768,49152,40960,16384)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def events(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    result = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            result.append(row)
    return result


def load_policy(path):
    policy = read(path)
    if not policy or (policy.get('margin'), policy.get('direction'),
                      policy.get('consecutive_observations')) != (.01, 'worse_only', 1):
        raise ValueError('The below-5 dynamic-best one-sided 0.01 screening policy is required')
    policy.update(version=POLICY_VERSION, warmup_loss_below=5.0,
                  max_training_reference_age_seconds=15.0, mtp_weight_tolerance=1e-6,
                  watchdog_poll_seconds=1.0,
                  reference_method='latest_compatible_nonfuture_per_run_then_best',
                  full_validation_reference='best_canonical_score_achieved_by_elapsed_time')
    return policy


def training_signature(metric, config):
    sig = dict(metric.get('objective_signature') or {})
    stage = sig.get('stage')
    if not isinstance(stage, int) or not 0 <= stage < 5 or not sig.get('mtp'):
        return None
    if 'candidates' not in sig or 'prefix' not in sig:
        return None
    sig.setdefault('loss_definition', TRAIN_DEFINITION)
    sig.setdefault('attention_windows', list(WINDOWS[stage]))
    sig.setdefault('train_max_seq_len', SEQUENCES[stage])
    sig.setdefault('attention_segment_cap', 2560)
    if config.get('architecture') in ('pr393-moe','pr393-moe-ks3'):
        # Registered FP64 audits151/153/154 found incorrect grouped training
        # masks in earlier kernels. Their canonical validation is unchanged,
        # but their training proxy is not the corrected context objective.
        sig.setdefault('attention_mask_semantics',config.get(
            'attention_mask_semantics','legacy-pr393-grouped-mask-v1'))
    # Sampled normalizers depend on batch candidate construction. For full CE,
    # changing batch size does not change the loss definition.
    if sig['candidates']:
        sig.setdefault('batch_tokens', config.get('batch_tokens', DEFAULT_BATCHES)[stage])
    else:
        sig.pop('batch_tokens', None)
    return sig


def point(metric, config, run_id):
    kind = metric.get('event')
    if kind == 'training':
        signature = training_signature(metric, config)
        value = metric.get('train_objective')
    elif kind == 'validation':
        if metric.get('full_validation') is not True or metric.get('val_tokens') != 10485760:
            return None
        signature = metric.get('evaluation_signature')
        if signature is None:
            # Legacy interim evaluations did not restore canonical final short
            # windows/YaRN. Only an actual final-stage evaluation is comparable.
            extension_start = config.get('plan', {}).get('base_steps', math.inf) + 52
            if not metric.get('final') or metric.get('step', -1) < extension_start:
                return None
            signature = dict(loss_definition=VAL_DEFINITION, val_tokens=10485760,
                             attention_windows=[768,2560])
        value = metric.get('val_loss')
    else:
        return None
    wall = metric.get('wall_seconds')
    if signature is None or not isinstance(value, (int,float)) or not isinstance(wall, (int,float)):
        return None
    if not math.isfinite(wall) or wall < 0:
        return None
    return dict(run_id=run_id, kind=kind, loss=value, wall_seconds=wall,
                signature=signature, source_step=metric.get('step'))


def compatible(a, b, policy):
    if a['kind'] != b['kind']:
        return False
    x, y = a['signature'], b['signature']
    if a['kind'] == 'validation':
        return x == y
    fixed = ('stage', 'candidates', 'loss_definition', 'attention_windows',
             'train_max_seq_len', 'attention_segment_cap', 'batch_tokens',
             'attention_mask_semantics')
    if any(x.get(key) != y.get(key) for key in fixed):
        return False
    wx, wy = x.get('mtp', []), y.get('mtp', [])
    return (bool(wx) and len(wx) == len(wy)
            and max(abs(i-j) for i,j in zip(wx, wy)) <= policy['mtp_weight_tolerance']
            and abs(x.get('prefix', 0)-y.get('prefix', 0)) <= 1e-6)


def assess(candidate, references, warmup_reached, policy):
    """One observation is sufficient once the persistent below-five gate opens."""
    value = candidate['loss']
    result = dict(policy_version=policy['version'], warmup_reached=warmup_reached,
                  eligible=False, abort=False, candidate_objective=value if math.isfinite(value) else None,
                  metric_kind=candidate['kind'], candidate_signature=candidate['signature'],
                  margin=policy['margin'])
    if not math.isfinite(value):
        return dict(result, reason='nonfinite_loss', abort=True)
    result['warmup_reached'] = warmup_reached or value < policy['warmup_loss_below']
    if not result['warmup_reached']:
        return dict(result, reason='awaiting_strictly_below_5')
    by_run = {}
    for ref in references:
        if ref['run_id'] == candidate['run_id'] or not math.isfinite(ref['loss']):
            continue
        age = candidate['wall_seconds'] - ref['wall_seconds']
        if age < 0 or not compatible(candidate, ref, policy):
            continue
        if candidate['kind'] == 'training' and age > policy['max_training_reference_age_seconds']:
            continue
        previous = by_run.get(ref['run_id'])
        if previous is None or (candidate['kind'] == 'training' and ref['wall_seconds'] > previous['wall_seconds']) or (
                candidate['kind'] == 'validation' and ref['loss'] < previous['loss']):
            by_run[ref['run_id']] = ref
    if not by_run:
        return dict(result, reason='no_comparable_reference')
    best = min(by_run.values(), key=lambda p:(p['loss'], -p['wall_seconds'], p['run_id']))
    # Decimal strings make an exactly +0.01 logged difference inclusive, without
    # a binary subtraction such as 4.01-4.0 rounding just below the threshold.
    gap = Decimal(str(value)) - Decimal(str(best['loss']))
    abort = gap >= Decimal(str(policy['margin']))
    return dict(result, eligible=True, reason='dynamic_best_training_proxy' if candidate['kind']=='training'
                else 'dynamic_best_full_validation', abort=abort, gap=float(gap),
                reference_run_id=best['run_id'], reference_wall_seconds=best['wall_seconds'],
                reference_age_seconds=candidate['wall_seconds']-best['wall_seconds'],
                reference_objective=best['loss'], reference_signature=best['signature'],
                reference_source_step=best['source_step'], comparable_runs=len(by_run))


class ReferenceCurves:
    """Refresh immutable or growing recorded curves; never include this candidate."""
    def __init__(self, root, state, candidate_run_id):
        self.root, self.state, self.candidate = Path(root), Path(state), candidate_run_id
        self.cache = {}

    def _file(self, path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return []
        stamp = (stat.st_mtime_ns, stat.st_size)
        if path not in self.cache or self.cache[path][0] != stamp:
            self.cache[path] = (stamp, events(path))
        return self.cache[path][1]

    def refresh(self):
        points = []
        active=read(self.root/'configs/control393/control_set.json',{})
        if active.get('status')=='completed':
            for control in active['controls']:
                directory=Path(control['directory'])/'artifacts'
                config=read(directory/'config.json',{})
                for row in self._file(directory/'metrics.jsonl'):
                    p=point(row,config,'control393-'+control['run_id'])
                    if p is not None and math.isfinite(p['loss']):points.append(p)
        mlp = read(self.state/'mlp_reference.json', {})
        for row in ([] if active.get('status')=='completed' else mlp.get('observations', [])):
            p = point(dict(row, event='training'), {}, 'mlp-control')
            if p is not None and math.isfinite(p['loss']):
                points.append(p)
        control = read(self.root/'configs/ks3/mlp_control.json', {})
        if active.get('status')!='completed' and control.get('status') == 'completed' and control.get('validation_tokens') == 10485760:
            points.append(dict(run_id='mlp-control', kind='validation',
                loss=control['validation_loss'], wall_seconds=control['end_to_end_wall_seconds'],
                signature=dict(loss_definition=VAL_DEFINITION, val_tokens=10485760,
                               attention_windows=[768,2560]), source_step=control.get('training_steps',1)-1))
        for record_path in sorted((self.state/'runs').glob('*/run.json')):
            record = read(record_path, {})
            run_id = record.get('run_id')
            if record.get('kind') != 'trial' or not run_id or run_id == self.candidate:
                continue
            directory = record_path.parent/'artifacts'
            config = read(directory/'config.json', {})
            if active.get('status')=='completed' and config.get('architecture')!='pr393-moe-ks3':
                continue
            if config.get('mode') != 'trial':
                continue
            rows = self._file(directory/'metrics.jsonl')
            training = [r for r in rows if r.get('event') == 'training']
            # Registered trials start from initialization and advance monotonically.
            # Explicit noncontinuous modes and stage-jump smoke/pilot runs never enter.
            if any(r.get('mode') in ('smoke','pilot') for r in rows) or config.get('continuous_training') is False:
                continue
            if any(b.get('step',-1) <= a.get('step',-1) or b.get('wall_seconds',-1) <= a.get('wall_seconds',-1)
                   for a,b in zip(training, training[1:])):
                continue
            for row in rows:
                p = point(row, config, run_id)
                if p is not None and math.isfinite(p['loss']):
                    points.append(p)
        self.points = points
        return points

    def snapshot(self):
        payload = dict(version=POLICY_VERSION, excluded_candidate=self.candidate, observations=self.points)
        payload['sha256'] = hashlib.sha256(json.dumps(self.points, sort_keys=True).encode()).hexdigest()
        return payload
