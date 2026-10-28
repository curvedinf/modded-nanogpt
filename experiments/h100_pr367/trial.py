"""Run a frozen PR367 retrieval + KS3 MoE candidate after the exact control."""
import argparse
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

from experiments.h100_pr367.baseline import ROOT, RESULTS, PYTHON, append, utc
from experiments.h100_pr367.source import prepare

LOGS = ROOT / 'logs/wr-reproduction/h100-pr367-ks3'


def stopped():
    return any((p / 'STOP').exists() for p in (RESULTS, LOGS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=413)
    args = parser.parse_args()
    if stopped():
        raise SystemExit('PR367 study stopped')
    controls = [json.loads(p.read_text()) for p in (RESULTS / 'runs').glob('*/result.json')]
    controls = [r for r in controls if r['status'] == 'complete' and r['settings']['kind'] == 'pr367-original-baseline'
                and r.get('source_hashes_unchanged') and r.get('steps') == 540]
    if not controls:
        raise SystemExit('Complete the unchanged PR367 full-validation control first')
    for evidence in ('fp8-512-check.json', 'gradient-collector-check.json'):
        data = json.loads((RESULTS / evidence).read_text())
        assert data.get('passed', all(c['passed'] for c in data.get('cases', [])))
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--',
        'experiments/h100_pr367', 'experiments/h100_ks3', 'wr_reproduction/pr367',
        'configs/wr_reproduction/h100_pr367_ks3.json'], cwd=ROOT, text=True)
    if dirty:
        raise SystemExit('Commit the candidate code first: ' + dirty)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = LOGS / stamp
    run.mkdir(parents=True)
    upstream = prepare(run / 'source')
    for folder in ('experiments/h100_pr367', 'experiments/h100_ks3', 'wr_reproduction', 'ks3_research'):
        shutil.copytree(ROOT / folder, run / 'overlay' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', 'upstream'))
    recipe = json.loads((ROOT / 'configs/wr_reproduction/h100_pr367_ks3.json').read_text())
    overrides = dict(recipe['environment'])
    # PR367's standard optimizer gathers BF16 banks before this first port's
    # local FP8 publication. Do not claim the PR393 owner-only transport path.
    overrides.update(TRAIN_SEED=str(args.seed),
        PATH='/root/venvs/pr393-h100/bin:' + os.environ['PATH'], CUDA_HOME='/usr/local/cuda',
        DATA_PATH=str(ROOT), PYTHONPATH=str(run / 'overlay'),
        LD_LIBRARY_PATH='/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
        LOCAL_KERNELS='devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned')
    env = {k: v for k, v in os.environ.items() if not k.startswith('KS3_') and k != 'NUM_SCHEDULED_ITERATIONS'}
    env.update(overrides)
    manifest = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                for directory in ('source', 'overlay') for p in (run / directory).rglob('*.py')}
    settings = dict(utc=utc(), kind='pr367-ks3-moe', runner_commit=commit, upstream_commit=upstream,
        seed=args.seed, steps=540, validation_tokens=10485760, retrieval_training_tokens=37000000000,
        moe_layers=[1, 3, 5, 6, 8], experts=8, top_k=1, charts=1024, rank=1, cubic=True,
        geometry_lr_multiplier=.125, transport='BF16 bank all-gather; local cached dual-layout FP8',
        shared_model_training_and_evaluation='unchanged PR367', environment=overrides,
        reference_run_ids=[r['run_id'] for r in controls], recipe=recipe)
    (run / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    (run / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    script = '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")/source"\nunset NUM_SCHEDULED_ITERATIONS\n'
    script += ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in overrides.items())
    script += 'exec bash run.sh\n'
    (run / 'reproduce.sh').write_text(script)
    subprocess.run(['nvidia-smi'], stdout=(run / 'gpu-environment.txt').open('w'), check=True)
    subprocess.run(['uv', 'pip', 'freeze', '--python', PYTHON], stdout=(run / 'packages.txt').open('w'), check=True)
    process = subprocess.Popen(['bash', 'run.sh'], cwd=run / 'source', env=env,
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
        append(dict(run_id=stamp, status='stopped', settings=settings))
        raise SystemExit('PR367 KS3 trial stopped')

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while process.poll() is None:
        if stopped():
            stop()
        time.sleep(2)
    (run / 'exit_code.txt').write_text(str(process.returncode) + '\n')
    unchanged = all(hashlib.sha256((run / p).read_bytes()).hexdigest() == h for p, h in manifest.items())
    console = (run / 'console.log').read_text(errors='replace')
    matches = re.findall(r'\[wr-result\] steps:(\d+) loss:([\d.]+) seconds:([\d.]+)', console)
    result = dict(run_id=stamp, status='complete' if process.returncode == 0 else 'failed',
        exit_code=process.returncode, settings=settings, execution_hashes_unchanged=unchanged)
    if matches:
        steps, loss, seconds = matches[-1]
        result.update(steps=int(steps), validation_loss=float(loss), official_seconds=float(seconds),
                      target_met=float(loss) <= 3.28, loss_print_precision=9)
    elif process.returncode == 0:
        result['status'] = 'missing_validation'
    (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    out = RESULTS / 'runs' / stamp
    out.mkdir(parents=True)
    for name in ('result.json', 'settings.json', 'execution_manifest.json', 'reproduce.sh', 'packages.txt', 'gpu-environment.txt'):
        shutil.copy2(run / name, out / name)
    (out / 'console-tail.txt').write_text('\n'.join(console.splitlines()[-100:]) + '\n')
    append(result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
