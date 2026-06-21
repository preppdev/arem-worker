#!/bin/bash
# Launch the warm preview worker (single-frame enhance for the uploader app).
set -euo pipefail
cd "$(dirname "$0")"
if [ -f /etc/arem-worker.env ]; then set -a; source /etc/arem-worker.env; set +a; fi
export RCLONE_R2="${RCLONE_R2:-r2}"
export R2_BUCKET="${R2_BUCKET:-arem-training-data}"
# bundle NVIDIA libs so onnxruntime/torch CUDA find cuDNN/cuBLAS
_NVLIBS=$(echo /home/jordan/miniconda3/envs/arem-photo-ai/lib/python3.11/site-packages/nvidia/*/lib | tr " " ":")
export LD_LIBRARY_PATH="${_NVLIBS}:${LD_LIBRARY_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"
echo "[run_preview] python=$PYTHON_BIN max_px=${PREVIEW_MAX_PX:-1600}"
exec "$PYTHON_BIN" -m scripts.preview_worker
