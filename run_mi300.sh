#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export GLOO_SOCKET_IFNAME=lo
exec .venv-mi300/bin/torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29501}" train_gpt_mi300.py "$@"
