"""Prepare an isolated full-WR KS3 run; execute the printed launch.sh afterward."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]

def main():
    upstream = ROOT / 'wr_reproduction/pr393/upstream'
    manifest = json.loads((upstream.parent / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((upstream / name).read_bytes()).hexdigest() == digest, name
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'logs/wr-reproduction/h100-ks3' / stamp
    run.mkdir(parents=True)
    shutil.copytree(upstream, run / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    path = run / 'source/train_gpt.py'
    source = path.read_text()
    marker = '    model: nn.Module = GPT('
    assert source.count(marker) == 1
    source = source.replace(marker, '    from experiments.h100_ks3.port import install as install_h100_ks3\n    KS3GPT = install_h100_ks3()\n    model: nn.Module = KS3GPT(')
    marker = '            uncompiled_model.complete_yarn_tables()'
    assert source.count(marker) == 1
    source = source.replace(marker, '            # Refresh KS3 from final shipped weights, on the training clock.\n            uncompiled_model.prepare_ks3_eval()\n' + marker)
    path.write_text(source)
    for folder in ['experiments/h100_ks3', 'wr_reproduction', 'ks3_research']:
        shutil.copytree(ROOT / folder, run / 'overlay' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', 'upstream'))
    (run / 'execution_manifest.json').write_text(json.dumps({
        str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['source', 'overlay'] for p in (run / folder).rglob('*.py')
    }, indent=2) + '\n')
    script = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/source"
export PATH=/root/venvs/pr393-h100/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export LOCAL_KERNELS=devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned
export DATA_PATH=/root/modded-nanogpt
export PYTHONPATH="$(pwd)/../overlay"
unset NUM_SCHEDULED_ITERATIONS
nvidia-smi > ../gpu-environment.txt
uv pip freeze --python /root/venvs/pr393-h100/bin/python > ../packages.txt
set +e
bash run.sh > ../console.log 2>&1
result=$?
printf '%s\\n' "$result" > ../exit_code.txt
exit "$result"
'''
    (run / 'launch.sh').write_text(script)
    (run / 'launch.sh').chmod(0o755)
    print(run / 'launch.sh')

if __name__ == '__main__':
    main()
