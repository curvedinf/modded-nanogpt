"""One unchanged PR367 convergence reproduction, with immutable source evidence."""
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'wr_reproduction/pr367/upstream'
RESULTS = ROOT / 'research_results/wr_reproduction/h100-pr367'
LOGS = ROOT / 'logs/wr-reproduction/h100-pr367-baseline'
PYTHON = '/root/venvs/pr393-h100/bin/python'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append(row):
    with (RESULTS / 'registry.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=utc(), **row)) + '\n')


def stopped():
    return any((p / 'STOP').exists() for p in (RESULTS, LOGS))


def main():
    manifest = json.loads((SOURCE.parent / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest, name
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--',
        'experiments/h100_pr367', 'wr_reproduction/pr367'], cwd=ROOT, text=True)
    if dirty:
        raise SystemExit('Commit the new reproduction code first: ' + dirty)
    runner_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # The download verifier writes readiness only after every file is checked.
    while not (RESULTS / 'dataset-ready.json').exists():
        if stopped():
            raise SystemExit('PR367 study stopped while waiting for data')
        time.sleep(2)
    if stopped():
        raise SystemExit('PR367 study STOP is present')
    subprocess.run([PYTHON, '-c', 'from exact_match import SlotIndex, StreamIndex, memfd'], check=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = LOGS / stamp
    run.mkdir(parents=True)
    shutil.copytree(SOURCE, run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    overrides = dict(PATH='/root/venvs/pr393-h100/bin:' + os.environ['PATH'],
        CUDA_HOME='/usr/local/cuda', DATA_PATH=str(ROOT),
        LD_LIBRARY_PATH='/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
        LOCAL_KERNELS='devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned')
    environment = dict(os.environ, **overrides)
    for name in ('TRAIN_SEED', 'NUM_SCHEDULED_ITERATIONS', 'PYTHONPATH'):
        environment.pop(name, None)
    settings = dict(utc=utc(), kind='pr367-original-baseline',
        upstream_commit=manifest['commit'],
        runner_commit=runner_commit,
        source_changes=[], model_dim=512, steps=540, seed='original random default',
        validation_tokens=10485760, retrieval_training_tokens=37000000000,
        environment=overrides, dataset=json.loads((RESULTS / 'dataset-ready.json').read_text()))
    (run / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    (run / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    script = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/source"\n'
    script += 'unset TRAIN_SEED NUM_SCHEDULED_ITERATIONS PYTHONPATH\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in overrides.items())
    script += 'exec bash run.sh\n'
    (run / 'reproduce.sh').write_text(script)
    subprocess.run(['nvidia-smi'], stdout=(run / 'gpu-environment.txt').open('w'), check=True)
    subprocess.run(['uv', 'pip', 'freeze', '--python', PYTHON], stdout=(run / 'packages.txt').open('w'), check=True)
    process = subprocess.Popen(['bash', 'run.sh'], cwd=run / 'source', env=environment,
        stdout=(run / 'console.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)
    (run / 'launcher.pid').write_text(str(process.pid) + '\n')
    append(dict(run_id=stamp, status='running', pid=process.pid, settings=settings))
    print(run, flush=True)

    def stop(*_):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait()
        append(dict(run_id=stamp, status='stopped'))
        raise SystemExit('PR367 reproduction stopped')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while process.poll() is None:
        if stopped():
            stop()
        time.sleep(2)
    (run / 'exit_code.txt').write_text(str(process.returncode) + '\n')
    unchanged = all(hashlib.sha256((run / 'source' / p).read_bytes()).hexdigest() == h
                    for p, h in manifest['files'].items())
    console = (run / 'console.log').read_text(errors='replace')
    matches = re.findall(r'step:(\d+)/(\d+) val_loss:([\d.]+) train_time:([\d.]+)ms', console)
    result = dict(run_id=stamp, status='complete' if process.returncode == 0 else 'failed',
        exit_code=process.returncode, settings=settings, source_hashes_unchanged=unchanged)
    if matches:
        step, total, loss, milliseconds = matches[-1]
        result.update(steps=int(step), total_steps=int(total), validation_loss=float(loss),
            official_seconds=float(milliseconds) / 1000, target_met=float(loss) <= 3.28,
            loss_print_precision=4)
    elif process.returncode == 0:
        result['status'] = 'missing_validation'
    (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    out = RESULTS / 'runs' / stamp
    out.mkdir(parents=True)
    for name in ('result.json', 'settings.json', 'source_manifest.json', 'reproduce.sh', 'packages.txt', 'gpu-environment.txt'):
        shutil.copy2(run / name, out / name)
    (out / 'console-tail.txt').write_text('\n'.join(console.splitlines()[-80:]) + '\n')
    append(result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
