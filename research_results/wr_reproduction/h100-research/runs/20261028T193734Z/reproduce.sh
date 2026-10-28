#!/usr/bin/env bash
set -euo pipefail
export TRAIN_SEED=413
export KS3_H100_OPT=separate-dx-prepared-128-gate-tanh-fp16v
export KS3_H100_TRANSPORT=codes
export KS3_H100_PACK_ROWS=16
export KS3_H100_FUSED_ANVIL=1
export KS3_H100_GEOMETRY_LR=0.25
exec /root/modded-nanogpt/logs/wr-reproduction/h100-research/20261028T193734Z/launch.sh
