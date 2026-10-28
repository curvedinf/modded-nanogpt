#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/source"
export PATH=/root/venvs/pr393-h100/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export LOCAL_KERNELS=devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned
export DATA_PATH=/root/modded-nanogpt
unset NUM_SCHEDULED_ITERATIONS PYTHONPATH
nvidia-smi > ../gpu-environment.txt
python -m pip freeze > ../packages.txt 2>/dev/null || uv pip freeze --python /root/venvs/pr393-h100/bin/python > ../packages.txt
set +e
bash run.sh > ../console.log 2>&1
result=$?
printf '%s\n' "$result" > ../exit_code.txt
exit "$result"
