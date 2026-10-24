"""Timed, serial launcher with source/data provenance and honest complete-run outcomes."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from control393.launch_config import load_config, plan


def compare(result, config, code, wall):
    faster = wall < config['control_wall_seconds']
    full = result.get('full_validation') is True and result.get('validation_tokens') == 10485760
    better = full and result.get('val_loss', float('inf')) <= config.get('target_val_loss', config['control_val_loss'])
    return dict(faster_than_control=faster, beats_control_loss=better,
                target_reached=bool(code == 0 and better),
                exploratory_win=bool(code == 0 and faster and better), record_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/control393/base.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    if os.environ.get('DATA_PATH') not in (None, '', '.'):
        parser.error('DATA_PATH must use the control data directory for this study')
    Path('logs').mkdir(exist_ok=True)
    gpu_lock = open('logs/ks3-gpu.lock', 'w')
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another KS3 run owns this GPU; measured trials run serially')
    config = load_config(options.config)
    mode = 'pilot' if options.pilot else 'smoke' if options.smoke else 'trial'
    if mode == 'trial':
        config['plan'] = plan(config, None)
    else:
        config['plan'] = (dict(base_steps=1120, total_steps=1192) if mode == 'smoke'
                          else dict(base_steps=1122, total_steps=1194))
    config.update(seed=options.seed, mode=mode)
    out = options.out or Path('logs')/datetime.now(timezone.utc).strftime(f'ks3-{mode}-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path('train_control393.py'), Path('run_control393.sh'), Path('control393/norm_map_pr375.npy'), Path('control393/provenance.json'), Path('control393/LICENSE.upstream'), *sorted(Path('control393').rglob('*.py')), *sorted(Path('.').glob('test_ks3*.py')), Path('LICENSE'),
               Path('requirements-mi300.lock.txt'), Path('ks3_research/vendor/PROVENANCE.json'),
               Path('configs/ks3/screening_policy.json'), Path('AGENTS.md'),
               Path('ks3_research/vendor/LICENSE.k-splanifolds'),
               *sorted(Path('ks3_research').rglob('*.py')), *sorted(Path('mi300').rglob('*.py'))]
    sources.extend(sorted(Path('.').glob('test_control393*.py')))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with tarfile.open(out/'source.tar.gz', 'w:gz') as archive:
        for p in sources:
            archive.add(p, arcname=str(p))
    baseline = json.loads(Path(config['control_manifest']).read_text())
    for item in baseline['data']:
        p = Path(item['path'])
        digest = hashlib.sha256()
        with p.open('rb') as f:
            while chunk := f.read(8*1024*1024):
                digest.update(chunk)
        if p.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Data differs from the control: {p}')
    config['data_sha256'] = {d['path']: d['sha256'] for d in baseline['data']}
    config['source_sha256'] = hashes
    config['ks3_source'] = json.loads(Path('ks3_research/vendor/PROVENANCE.json').read_text())
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    with (out/'gpu-before.txt').open('w') as hardware:
        subprocess.run(['rocm-smi','--showproductname','--showuse','--showtemp','--showpower'],
                       stdout=hardware,stderr=subprocess.STDOUT,check=True)
    command = ['.venv-mi300/bin/torchrun', '--nnodes=1', '--nproc_per_node=1',
               '--master_addr=127.0.0.1', '--master_port='+os.environ.get('MASTER_PORT','29502'),
               'train_control393.py', '--run-dir', str(out.resolve())]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', GLOO_SOCKET_IFNAME='lo')
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    env['KS3_WALL_START'] = str(started)
    print(json.dumps({'run_dir':str(out), 'mode':mode, 'plan':config['plan']}), flush=True)
    with (out/'stdout.log').open('w', buffering=1) as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        (out/'process.json').write_text(json.dumps({'launcher_pid':os.getpid(),'torchrun_pid':process.pid,
                                                  'started_utc':started_utc,'command':command},indent=2)+'\n')
        for line in process.stdout:
            log.write(line)
            print(line, end='', flush=True)
        code = process.wait()
    wall = time.perf_counter()-started
    result_path = out/'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update(exit_code=code, complete_wall_seconds=wall, mode=mode,
                  started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat())
    if mode == 'trial':
        result.update(compare(result,config,code,wall))
    result_path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
