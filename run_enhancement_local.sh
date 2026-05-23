#!/bin/bash
# Launch the autonomous enhancement-pipeline worker locally on the 3090.
# Polls /api/internal/enhancement-jobs/claim and processes claimed jobs by
# running each step's detector + repair across every image in the parent
# shoot. Independent of run_local.sh (the main editing worker).
#
# Use case: pick up reviewer-queued enhancement passes within seconds.
#
# Usage:
#   bash run_enhancement_local.sh          # poll forever (Ctrl-C to stop)
#   bash run_enhancement_local.sh --once   # claim+process one job, exit
#   bash run_enhancement_local.sh --dry-run

set -euo pipefail
cd "$(dirname "$0")"

# Load credentials. When invoked via systemd's arem-enhancement-local.service,
# EnvironmentFile=/etc/arem-worker.env already injects these; otherwise
# source the same file for manual `bash` invocations.
if [ -f /etc/arem-worker.env ]; then
  set -a; source /etc/arem-worker.env; set +a
fi
: "${WORKER_TOKEN:?WORKER_TOKEN not set — see /etc/arem-worker.env}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY required for the Haiku detector}"
: "${GOOGLE_AI_API_KEY:?GOOGLE_AI_API_KEY required for the Nano Banana repair}"

export DASHBOARD_URL="${DASHBOARD_URL:-https://arem-editing-dashboard.vercel.app}"
export WORKER_ID="${WORKER_ID:-enhancement-local-$(hostname -s)}"
export IDLE_SLEEP="${IDLE_SLEEP:-5}"
export R2_PRODUCTION_BUCKET="${R2_PRODUCTION_BUCKET:-arem-production-edit-jobs}"
export R2_TRAINING_BUCKET="${R2_TRAINING_BUCKET:-arem-training-data}"
export RCLONE_R2="${RCLONE_R2:-r2}"

PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"

echo "[run_enhancement_local] worker_id=$WORKER_ID"
echo "[run_enhancement_local] dashboard=$DASHBOARD_URL"
echo "[run_enhancement_local] python=$PYTHON_BIN"

exec "$PYTHON_BIN" scripts/enhancement_worker.py "$@"
