#!/bin/bash
# Launch the single-image scene-reprocess worker (forced interior/exterior
# re-run of one delivered frame, driven from the delivery gallery).
set -euo pipefail
cd "$(dirname "$0")"
if [ -f /etc/arem-worker.env ]; then set -a; source /etc/arem-worker.env; set +a; fi
export RCLONE_R2="${RCLONE_R2:-r2}"
export R2_BUCKET="${R2_BUCKET:-arem-training-data}"
# bundle NVIDIA libs so torch CUDA finds cuDNN/cuBLAS
_NVLIBS=$(echo /home/jordan/miniconda3/envs/arem-photo-ai/lib/python3.11/site-packages/nvidia/*/lib | tr " " ":")
export LD_LIBRARY_PATH="${_NVLIBS}:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"
echo "[run_reprocess] python=$PYTHON_BIN"
exec "$PYTHON_BIN" -m scripts.reprocess_worker
