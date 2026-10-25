"""CPU-only screening/evidence for the isolated, raw-weight PR393 WR study.

Only assess/load_policy are shared with the old study. No old reference loader,
canonical evaluation, step alignment, or schedule interpolation is used here.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from ks3_research.ops.screening import assess, load_policy

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'logs/wr-reproduction/research'
RESULTS = ROOT / 'research_results/wr_reproduction/research'
CONTROL_LOSS = 3.454925298690796
CONTROL_SECONDS = 902.3592145430011
TRAIN_DEFINITION = 'pr393-cplm-softcap23-mtp-prefix-switchaux-tokenmean-rankmean-v1'
VAL_DEFINITION = 'pr393-cplm-full10485760-raw-no-canonical-no-tail-bf16-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(dict(utc=now(), **value), allow_nan=False) + '\n')


def events(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # An independently monitored writer may have an incomplete last line.
    return rows


class Metadata:
    """Exact FP32 MTP/prefix values from the audited CPU-derived source contract."""
    def __init__(self, data=None):
        from wr_reproduction.contracts import audit
        self.manifest = audit('pr393')
        self.contract = read(ROOT / 'wr_reproduction/pr393/contract.json')
        self.contract_sha256 = sha(ROOT / 'wr_reproduction/pr393/contract.json')
        tree = ast.parse((ROOT / 'wr_reproduction/pr393/upstream/track_1_short/sampled_softmax.py').read_text())
        self.counts = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'CANDIDATES_BY_STAGE' for t in n.targets))
        self.data = data or validation_data()

    def stage(self, step):
        return next((i for i, (_, end) in enumerate(self.contract['boundaries']) if step < end), 4)

    def training(self, step):
        i = self.stage(step)
        start, end = self.contract['boundaries'][i]
        stage, row = self.contract['stages'][i], self.contract['steps'][step]
        counts = self.counts.get(i)
        return dict(loss_definition=TRAIN_DEFINITION, stage=i,
                    attention_windows=[128*x for x in stage['window_sizes']],
                    mtp=row['mtp'], prefix=row['prefix'],
                    candidates=counts[(step-start)*len(counts)//(end-start)] if counts else 0,
                    batch_tokens=row['global_tokens'], global_batch_tokens=row['global_tokens'],
                    logical_batch_tokens=row['logical_rank_tokens'], logical_ranks=8,
                    train_max_seq_len=stage['train_max_seq_len'], attention_segment_cap=2560,
                    attention_mask_semantics='faithful-pr393-document-local-mi300-v1',
                    cplm=True, softcap=[23., 5., 7.5], moe_aux_weight=.01,
                    reduction='mean-of-eight-logical-rank-token-means',
                    prefix_table='build_prefix_table_bucket(vocab_size,0,1)',
                    sampling='original-rank-local-stride20011-forced-targets-and-copy',
                    source_commit=self.manifest['commit'])

    def validation(self, step):
        stage = self.contract['stages'][self.stage(step)]
        return dict(loss_definition=VAL_DEFINITION, val_tokens=10485760,
                    val_data=self.data, cplm=True, softcap=[23., 5., 7.5],
                    attention_windows=[128*x for x in stage['window_sizes']],
                    attention_segment_cap=2560, align_to_bos=False,
                    val_global_batch_tokens=2097152, logical_ranks=8,
                    weights='raw', canonical_mask=False, tail_shipping=False,
                    final_window_extension=False, expert_evaluation='bf16',
                    mtp=False, prefix=False, moe_aux=False)


def validation_data():
    import glob
    base = Path(__import__('os').environ.get('DATA_PATH', str(ROOT)))
    paths = sorted(glob.glob(str(base / 'data/fineweb10B/fineweb_val_*.bin')))
    if not paths:
        raise FileNotFoundError('Original validation data is required')
    return [dict(name=Path(p).name, bytes=Path(p).stat().st_size, sha256=sha(p)) for p in paths]


def point(row, run_id):
    kind = {'train': 'training', 'validation': 'validation'}.get(row.get('event'))
    if kind is None:
        return None
    sig = row.get('objective_signature' if kind == 'training' else 'evaluation_signature')
    value = row.get('objective' if kind == 'training' else 'loss')
    wall = row.get('train_seconds')
    if not isinstance(sig, dict) or not isinstance(value, (float, int)) or not isinstance(wall, (float, int)):
        return None
    if not math.isfinite(wall) or wall < 0:
        return None
    if kind == 'training':
        required = {'stage', 'attention_windows', 'mtp', 'prefix', 'candidates', 'global_batch_tokens',
                    'logical_batch_tokens', 'logical_ranks', 'cplm', 'moe_aux_weight', 'loss_definition'}
        if not required <= sig.keys() or sig['loss_definition'] != TRAIN_DEFINITION:
            return None
    else:
        if (row.get('final') is not False or row.get('val_tokens') != 10485760
                or sig.get('loss_definition') != VAL_DEFINITION or sig.get('val_tokens') != 10485760
                or sig.get('attention_windows') != [640, 1408]
                or sig.get('weights') != 'raw' or sig.get('expert_evaluation') != 'bf16'
                or any(sig.get(k) is not False for k in ('canonical_mask', 'tail_shipping', 'final_window_extension'))):
            return None
    return dict(run_id=run_id, kind=kind, loss=value, wall_seconds=wall,
                signature=sig, source_step=row.get('step'), evidence=row.get('evidence'))


def compatible_exact(candidate, reference):
    if candidate['kind'] != reference['kind']:
        return False
    x, y = candidate['signature'], reference['signature']
    # assess() handles the original 1e-6 MTP/prefix tolerance; all other fields
    # (including global/logical batch and aux terms) must match exactly here.
    omit = {'mtp', 'prefix'} if candidate['kind'] == 'training' else set()
    return {k: v for k, v in x.items() if k not in omit} == {k: v for k, v in y.items() if k not in omit}


def finite_reference(p):
    # Candidate NaNs must reach assess() to stop immediately. Reference NaNs
    # remain in immutable source evidence, but cannot enter JSON snapshots or
    # serve as comparative observations for later candidates.
    return bool(p and math.isfinite(p['loss']) and math.isfinite(p['wall_seconds'])
                and p['wall_seconds'] >= 0)


def decision(candidate, references, latched, policy, *, stop=False, verification=False):
    # Reject impossible time matches before comparing full signatures. This is
    # the same eligibility rule as assess(), not a new pruning threshold.
    wall, kind = candidate['wall_seconds'], candidate['kind']
    refs = [r for r in references if finite_reference(r) and r['kind']==kind and r['wall_seconds']<=wall
            and (kind!='training' or wall-r['wall_seconds']<=policy['max_training_reference_age_seconds'])
            and compatible_exact(candidate, r)]
    result = assess(candidate, refs, latched, policy)
    if result.get('reference_run_id'):
        selected = next(r for r in refs if r['run_id'] == result['reference_run_id']
                        and r['wall_seconds'] == result['reference_wall_seconds'])
        result['reference_evidence'] = selected.get('evidence')
    if stop:
        result.update(abort=True, reason='explicit_study_stop')
    elif verification and math.isfinite(candidate['loss']):
        result.update(abort=False, comparative_pruning_exempt=True,
                      proxy_would_abort=result['abort'], reason='user_authorized_three_seed_verification')
    return result


def derive_saved_references(metadata, destination=STATE / 'reference-index.json'):
    rows, sources = [], []
    for run_id in ('pr393-control-20261024T234443Z', 'pr393-ks3-block-20261025T002719Z'):
        path = ROOT / 'logs/wr-reproduction' / run_id / 'events.jsonl'
        if not path.exists():
            path = RESULTS / (run_id + '-events.jsonl')
        source_hash = sha(path)
        sources.append(dict(path=str(path), sha256=source_hash))
        last_step = None
        for line, original in enumerate(events(path), 1):
            row = dict(original)
            if row.get('event') == 'time_limit':
                last_step = row['step']
            if row.get('event') == 'train':
                row['objective_signature'] = metadata.training(row['step'] - 1)
            elif row.get('event') == 'validation' and last_step is not None and row.get('final') is False:
                row['evaluation_signature'] = metadata.validation(last_step)
                row['step'] = last_step
            else:
                continue
            row['evidence'] = dict(path=str(path), sha256=source_hash, line=line,
                                   row_sha256=digest(original), contract_sha256=metadata.contract_sha256,
                                   derivation='train step N is source update N-1; validation uses time_limit step')
            p = point(row, run_id)
            if finite_reference(p):
                rows.append(p)
    result = dict(created_utc=now(), sources=sources, observations=rows,
                  contract_sha256=metadata.contract_sha256, upstream_manifest=metadata.manifest)
    atomic(destination, result)
    return result


class References:
    def __init__(self, state, run_id):
        self.state, self.run_id, self.cache = Path(state), run_id, {}

    def refresh(self):
        result = [p for p in read(self.state / 'reference-index.json', {}).get('observations', [])
                  if finite_reference(p)]
        # Registry is append-only. Only explicitly registered continuous training
        # can contribute, never diagnostic/warmup/step-jump runs.
        registered = {r['run_id'] for r in events(self.state / 'registry.jsonl') if r.get('event') == 'registered'}
        for rid in sorted(registered - {self.run_id}):
            directory = self.state / 'runs' / rid
            config = read(directory / 'run.json', {})
            if config.get('continuous_from_initialization') is not True or config.get('kind') not in ('control', 'discovery', 'verification'):
                continue
            path = directory / 'events.jsonl'
            if not path.exists():
                continue
            stamp = (path.stat().st_size, path.stat().st_mtime_ns)
            if self.cache.get(rid, (None,))[0] != stamp:
                hashed = sha(path)
                points = []
                for line, row in enumerate(events(path), 1):
                    p = point(row, rid)
                    if finite_reference(p):
                        p['evidence'] = dict(path=str(path), sha256=hashed, line=line, row_sha256=digest(row))
                        points.append(p)
                self.cache[rid] = (stamp, points)
            result.extend(self.cache[rid][1])
        return result


class Screen:
    def __init__(self, run, state=STATE):
        self.run, self.state = Path(run), Path(state)
        self.config = read(self.run / 'run.json')
        self.stop_files = [Path(p) for p in self.config.get('stop_files',[str(self.state/'STOP')])]
        self.run_id = self.config['run_id']
        self.refs = References(state, self.run_id)
        self.policy = load_policy(ROOT / 'configs/ks3/screening_policy.json')
        self.policy['full_validation_reference'] = 'WR-raw-bf16-full10485760-score-achieved-by-elapsed-time'
        self.latched = read(self.run / 'screening_state.json', {}).get('warmup_reached', False)
        self.references = self.refs.refresh()
        snapshot = dict(observations=self.references)
        self.reference_hash = digest(snapshot)
        atomic(self.run / ('references-' + self.reference_hash + '.json'), snapshot)
        atomic(self.run / 'screening_policy.json', self.policy)
        self.refreshed = time.monotonic()

    def observe(self, row):
        if time.monotonic()-self.refreshed >= 1.:
            references = self.refs.refresh()
            # Cached immutable runs reuse point objects. Equal evidence needs
            # no repeated multi-megabyte JSON serialization/hash on the GIL.
            if references != self.references:
                self.references = references
                snapshot = dict(observations=self.references)
                hashed = digest(snapshot)
                if hashed != self.reference_hash:
                    self.reference_hash = hashed
                    atomic(self.run / ('references-' + hashed + '.json'), snapshot)
            self.refreshed = time.monotonic()
        candidate = point(row, self.run_id)
        if candidate is None:
            raise ValueError('Research observation has incomplete or incompatible metadata')
        result = decision(candidate, self.references, self.latched, self.policy,
                          stop=any(p.exists() for p in self.stop_files), verification=self.config['kind'] in ('control','verification'))
        if self.config['kind']=='control' and result.get('comparative_pruning_exempt'):
            result['reason']='user_authorized_matched_mlp_control'
        self.latched = result['warmup_reached']
        result.update(step=row.get('step'), train_seconds=row['train_seconds'], reference_snapshot_sha256=self.reference_hash)
        append(self.run / 'pruning.jsonl', result)
        atomic(self.run / 'screening_state.json', dict(updated_utc=now(), **result))
        return result


def qualifies(row, expected_signature):
    p = point(row, 'candidate')
    return bool(p and p['kind'] == 'validation' and p['signature'] == expected_signature
                and math.isfinite(p['loss']) and p['loss'] < CONTROL_LOSS
                and p['wall_seconds'] <= CONTROL_SECONDS)


def queue_verification(state, discovery, result):
    path = Path(state) / 'verification.json'
    saved = read(path, dict(groups=[]))
    recipe_id = digest(dict(recipe=discovery['recipe'], source_hashes=discovery['source_hashes']))
    if not any(g['recipe_id'] == recipe_id for g in saved['groups']):
        saved['groups'].append(dict(recipe_id=recipe_id, discovery_run_id=discovery['run_id'],
                                   frozen_run=discovery, discovery_result=result, status='queued',
                                   seeds=[dict(seed=s, status='queued') for s in (169, 170, 171)],
                                   exception='No comparative pruning; stop/error/nonfinite remain active', utc=now()))
        atomic(path, saved)
    return saved


def verification_statistics(rows):
    losses = [r['loss'] for r in rows]
    return dict(scores=rows, count=len(rows), mean=statistics.mean(losses),
                sample_std=statistics.stdev(losses) if len(rows) > 1 else 0.,
                minimum=min(losses), maximum=max(losses),
                count_beating_control=sum(r['loss'] < CONTROL_LOSS and r['train_seconds'] <= CONTROL_SECONDS for r in rows))
