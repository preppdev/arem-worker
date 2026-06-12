#!/bin/bash
# Launch the arem-video-render stub worker (mirrors run_inpaint_sandbox.sh).
set -euo pipefail
cd "$(dirname "$0")"
if [ -f /etc/arem-worker.env ]; then set -a; source /etc/arem-worker.env; set +a; fi
export RCLONE_R2="${RCLONE_R2:-r2}"
export R2_BUCKET="${R2_BUCKET:-arem-training-data}"
PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"
exec "$PYTHON_BIN" /home/jordan/arem-worker/scripts/video_render_worker.py "$@"
