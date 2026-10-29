#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
    TRITON_PTXAS_PATH=$(python -m track_1_short.ks3.setup_ptxas)
    export TRITON_PTXAS_PATH
fi
exec torchrun --standalone --nproc_per_node=8 train_gpt.py
