#!/usr/bin/env python3
"""Reproduce the frozen PR393 MLP/KS3 comparison; see MI300.md.

Preparation, download and source/data verification use only the standard library.
The original frozen training/controller modules are imported only by `run`.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'reproductions/pr393_ks3_v2/manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_sources(root, manifest):
    errors = []
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            errors.append(name)
    if errors:
        raise ValueError('Frozen reproduction files differ or are missing: ' + ', '.join(errors))
    return len(manifest['files'])


def dataset_entries(dataset):
    return dataset['files'] + dataset['tokenizer_files']


def verify_data(data_root, dataset):
    expected = {Path(item['path']).name for item in dataset['files']}
    directory = data_root / 'data/fineweb10B'
    actual = {p.name for pattern in ('fineweb_train_*.bin', 'fineweb_val_*.bin')
              for p in directory.glob(pattern)}
    if actual != expected:
        raise ValueError(f'Data shard set differs: missing={sorted(expected - actual)}, '
                         f'extra={sorted(actual - expected)}. Run the download command in a separate data root.')
    for item in dataset_entries(dataset):
        path = data_root / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Data checksum/size mismatch or missing file: {path}')
    return len(dataset['files'])


def download(data_root, dataset):
    for item in dataset_entries(dataset):
        path = data_root / item['path']
        if path.exists():
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError(f'Existing file does not match the frozen data: {path}')
            print('Verified', path, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = item.get('url') or (f"https://huggingface.co/datasets/{dataset['repository']}/resolve/"
                                 f"{dataset['revision']}/{path.name}")
        temporary = path.with_name(path.name + '.part-' + uuid.uuid4().hex)
        print('Downloading', url, flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('xb') as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size != item['bytes'] or sha(temporary) != item['sha256']:
                raise ValueError('Downloaded checksum/size mismatch: ' + url)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    verify_data(data_root, dataset)


def environment(root, manifest, *, gpu=False):
    versions, errors = {}, []
    for line in (root / manifest['requirements']).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        if actual != expected:
            errors.append(f'{name}: expected {expected}, found {actual}')
    if platform.python_version() != manifest['environment']['python']:
        errors.append(f"Python: expected {manifest['environment']['python']}, found {platform.python_version()}")
    if errors:
        raise ValueError('Environment differs from the recorded run:\n' + '\n'.join(errors))
    record = dict(utc=now(),python=platform.python_version(),platform=platform.platform(),packages=versions)
    if gpu:
        import torch
        if torch.version.hip != manifest['environment']['hip']:
            raise ValueError(f"Expected HIP {manifest['environment']['hip']}; found {torch.version.hip}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Expose exactly one MI300X with ROCR_VISIBLE_DEVICES.')
        props = torch.cuda.get_device_properties(0)
        if ('MI300X' not in props.name or not props.gcnArchName.startswith('gfx942')
                or props.total_memory < manifest['hardware']['minimum_vram_bytes']):
            raise ValueError(f'Expected MI300X 192 GB/gfx942; found {props}')
        free, total = torch.cuda.mem_get_info()
        if free < manifest['hardware']['minimum_free_vram_bytes']:
            raise ValueError(f'GPU is busy or lacks free memory: {free} bytes available. Use an idle MI300X.')
        record['gpu'] = dict(name=props.name,architecture=props.gcnArchName,
                             total_memory=total,free_memory=free,hip=torch.version.hip)
    return record


def child_environment(root, output, data_root, manifest):
    return dict(WR_PROBE_STATE=str(output),WR_PROBE_RESULTS=str(output / 'results'),
                WR_PROBE_MANAGER=str(output / 'manager'),WR_PROBE_PUBLIC_STATE=str(output),
                WR_PROBE_FENCE=str(root / manifest['fence']),
                WR_PROBE_PROTOCOL=str(root / manifest['protocol']),
                WR_PROBE_VERIFY_SEEDS='369,370,371',DATA_PATH=str(data_root),
                TIKTOKEN_CACHE_DIR=str(data_root / 'data/fineweb10B/tokenizer-cache'))


def stop_files(root, output):
    return [root / 'logs/wr-reproduction/probe-study/STOP',
            root / 'research_results/wr_reproduction/probe-study/STOP',
            output / 'STOP',output / 'results/STOP',output / 'manager/STOP']


def check_stop(root, output):
    active = [str(path) for path in stop_files(root, output) if path.exists()]
    if active:
        raise ValueError('Explicit STOP marker: ' + ', '.join(active))


def prepare(root, manifest, arm, seed, output, data_root, *, python=None):
    check_stop(root, output)
    owner = output / 'reproduction.json'
    if output.exists() and any(output.iterdir()) and not owner.exists():
        raise ValueError('Output is not an isolated reproduction directory: ' + str(output))
    if owner.exists() and read(owner)['manifest'] != manifest['id']:
        raise ValueError('Output belongs to a different reproduction manifest: ' + str(output))
    if not owner.exists():
        write(owner, dict(manifest=manifest['id'],created_utc=now()))
    original = read(root / manifest['arms'][arm]['config'])
    run = copy.deepcopy(original)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rid = f'reproduce-{arm}-seed{seed}-{stamp}-{uuid.uuid4().hex[:8]}'
    directory = output / 'runs' / rid
    directory.mkdir(parents=True, exist_ok=False)
    env = child_environment(root, output, data_root, manifest)
    # Keep the venv interpreter path: resolving its symlink would discard the venv.
    command = [str(Path(python or sys.executable).absolute()), '-m', 'wr_reproduction.probe_train',
               '--run', str(directory)]
    head = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
    run.update(run_id=rid,directory=str(directory),seed=seed,
               kind='control' if arm == 'mlp' else 'verification',command=command,
               unit='foreground-reproduction',created_utc=now(),environment=env,
               protocol_path=env['WR_PROBE_PROTOCOL'],fence_path=env['WR_PROBE_FENCE'],
               launch_head_commit=head.stdout.strip() if head.returncode == 0 else None,
               stop_files=[str(p) for p in stop_files(root, output)],calibration=None,
               comparative_pruning_exempt=True,continuous_from_initialization=True,
               reproduction=dict(manifest=manifest['id'],arm=arm,original_run_id=original['run_id'],
                                 original_seed=original['seed'],data_root=str(data_root),
                                 dataset_manifest_sha256=manifest['files'][manifest['dataset']]))
    write(directory / 'run.json', run)
    write(directory / 'config.json', run['recipe'])
    write(output / 'results' / (rid + '-config.json'), run)
    write(output / 'latest-run.json',dict(run=run,status='prepared',utc=now()))
    with (output / 'registry.jsonl').open('a') as registry:
        registry.write(json.dumps(dict(utc=now(),event='registered',run_id=rid,directory=str(directory),
                                      kind=run['kind'],seed=seed,code_commit=run['code_commit'])) + '\n')
    return run


def summarize(output, manifest):
    rows = []
    for path in sorted((output / 'results').glob('*-result.json')):
        result = read(path)
        val = result.get('validation')
        if not val:
            rows.append(dict(run_id=result['run_id'],status=result['status'],validation=None))
            continue
        arm = 'mlp' if result['recipe']['architecture'] == 'mlp' else 'ks3'
        loss = val.get('loss')
        elapsed = val.get('train_seconds')
        finite = isinstance(loss, (float,int)) and math.isfinite(loss)
        comparable = (result['status'] == 'complete' and finite
                      and isinstance(elapsed, (float,int)) and 0 <= elapsed <= 900
                      and val.get('val_tokens') == 10485760
                      and val.get('evaluation_signature') == manifest['evaluation_signature'])
        rows.append(dict(run_id=result['run_id'],arm=arm,seed=result['seed'],status=result['status'],
                         loss=loss,train_seconds=elapsed,steps=val.get('step'),
                         matches_evaluation_and_budget=comparable,
                         loss_minus_published_mlp=loss - manifest['arms']['mlp']['records'][0]['loss'] if finite else None))
    groups = {}
    for arm in ('mlp','ks3'):
        scores = [r['loss'] for r in rows if r.get('arm') == arm and r.get('matches_evaluation_and_budget')]
        if scores:
            groups[arm] = dict(count=len(scores),mean=statistics.mean(scores),minimum=min(scores),
                               sample_std=statistics.stdev(scores) if len(scores) > 1 else None)
    report = dict(utc=now(),manifest=manifest['id'],runs=rows,statistics=groups,
                  published_references={k:v['records'] for k,v in manifest['arms'].items()},
                  metric='Full validation loss versus actual elapsed training seconds; steps are diagnostic only.')
    if all(a in groups for a in ('mlp','ks3')):
        report['ks3_mean_minus_reproduced_mlp_mean'] = groups['ks3']['mean'] - groups['mlp']['mean']
    write(output / 'summary.json', report)
    return report


def execute(root, manifest, arm, seeds, output, data_root):
    lock_path = root / 'logs/wr-reproduction/gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another research/reproduction process owns this checkout\'s GPU lease.') from error
        check_stop(root, output)
        forbidden = ['NUM_SCHEDULED_ITERATIONS','MOE_GRAD_FP8','MOE_DW_PROMOTE','MOE_DX_PROMOTE']
        if any(name in os.environ for name in forbidden):
            raise ValueError('Remove inherited recipe overrides: ' + ', '.join(k for k in forbidden if k in os.environ))
        # The check process exits before training, releasing its GPU context.
        inspection = subprocess.run([sys.executable,'-c',
            'import json,reproduce_mi300 as r; print(json.dumps(r.environment(r.ROOT,r.read(r.MANIFEST),gpu=True)))'],
            cwd=root,capture_output=True,text=True)
        if inspection.returncode:
            raise ValueError('Hardware/environment preflight failed:\n' + inspection.stderr)
        runtime = json.loads(inspection.stdout)
        os.environ.update(child_environment(root, output, data_root, manifest))
        from wr_reproduction.probe_controller import execute as original_execute

        def stop_requested(signum, frame):
            output.mkdir(parents=True, exist_ok=True)
            (output / 'STOP').touch()
        previous = {sig:signal.signal(sig, stop_requested) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            for seed in seeds:
                verify_sources(root, manifest)
                run = prepare(root, manifest, arm, seed, output, data_root)
                write(Path(run['directory']) / 'environment.json', runtime)
                print('Running', run['run_id'], '\nConsole:', Path(run['directory']) / 'console.log', flush=True)
                result = original_execute(run)
                summarize(output, manifest)
                if result['status'] != 'complete':
                    raise RuntimeError(f"{run['run_id']} ended as {result['status']}; inspect its console/events.")
                print(json.dumps(result['validation'], indent=2), flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify','download','prepare','run','summary'])
    parser.add_argument('--model', choices=['mlp','ks3'], default='ks3')
    parser.add_argument('--seeds', type=int, nargs='+', help='Defaults: MLP 169; KS3 369 370 371.')
    parser.add_argument('--data-root', type=Path, default=ROOT,
                        help='Contains data/fineweb10B; identical to upstream DATA_PATH semantics.')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/reproduce-pr393-ks3-v2')
    parser.add_argument('--source-only', action='store_true', help='verify only: no data or GPU required.')
    parser.add_argument('--environment', action='store_true', help='verify only: also check installed package versions, without using the GPU.')
    args = parser.parse_args()
    manifest = read(MANIFEST)
    data_root, output = args.data_root.resolve(), args.output.resolve()
    seeds = args.seeds if args.seeds is not None else manifest['arms'][args.model]['seeds']
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        parser.error('Seeds must be unique integers between 0 and 2**32 - 1.')
    if args.action == 'summary':
        print(json.dumps(summarize(output, manifest), indent=2))
        return
    print('Verified frozen files:', verify_sources(ROOT, manifest), flush=True)
    dataset = read(ROOT / manifest['dataset'])
    if args.action == 'download':
        download(data_root, dataset)
        return
    if args.action == 'verify' and args.environment:
        print(json.dumps(environment(ROOT, manifest), indent=2))
    if not (args.action == 'verify' and args.source_only):
        print('Verified data shards:', verify_data(data_root, dataset), flush=True)
    if args.action == 'prepare':
        for seed in seeds:
            run = prepare(ROOT, manifest, args.model, seed, output, data_root)
            print('Prepared (no training started):', Path(run['directory']) / 'run.json')
    elif args.action == 'run':
        execute(ROOT, manifest, args.model, seeds, output, data_root)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(1)
