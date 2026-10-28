#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/source"
unset TRAIN_SEED NUM_SCHEDULED_ITERATIONS PYTHONPATH
export PATH=/root/venvs/pr393-h100/bin:/root/.codex/packages/app-server-daemon/releases/0.165.0-x86_64-unknown-linux-musl/codex-path:/root/.codex/tmp/arg0-linux/codex-arg0Fy3uFI:/root/.codex/packages/app-server-daemon/releases/0.165.0-x86_64-unknown-linux-musl/codex-path:/root/.codex/tmp/arg0-linux/codex-arg0CTGwD5:/usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex-path:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export DATA_PATH=/root/modded-nanogpt
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:
export LOCAL_KERNELS=devenpzak/flash-attn3-12864=/root/venvs/pr393-fa3-pinned
exec bash run.sh
