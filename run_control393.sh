#!/bin/bash
set -euo pipefail
cd /root/modded-nanogpt
exec .venv-mi300/bin/python -m control393.launch "$@"
