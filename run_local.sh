#!/bin/bash
# Launch the arem-worker pipeline locally on the 3090 against the live
# dashboard queue. Pulls one job (or loops when invoked without --once).
#
# Use case: throughput comparison vs RunPod, or as a permanent overflow
# worker for the queue.
#
# Usage:
#   bash run_local.sh          # poll forever (Ctrl-C to stop)
#   bash run_local.sh --once   # claim one job, process, exit

set -euo pipefail
cd "$(dirname "$0")"

# Load credentials. When invoked via systemd's arem-worker-local.service,
# EnvironmentFile=/etc/arem-worker.env already injects these into the
# process env, so the sources below are no-ops. For manual
# `bash run_local.sh` invocations, prefer /etc/arem-worker.env (survives
# reboot — /tmp is tmpfs) over the legacy /tmp/vercel-prod.env path.
if [ -f /etc/arem-worker.env ]; then
  set -a; source /etc/arem-worker.env; set +a
elif [ -f /tmp/vercel-prod.env ]; then
  set -a; source /tmp/vercel-prod.env; set +a
fi
# Sanity check the bare-minimum required vars. Empty/unset = abort.
: "${WORKER_TOKEN:?WORKER_TOKEN not set — install /etc/arem-worker.env (see arem-worker-local.service header) or 'vercel env pull /tmp/vercel-prod.env --environment=production --yes' first}"

# Worker config
export DASHBOARD_URL="${DASHBOARD_URL:-https://arem-editing-dashboard.vercel.app}"
export WORKER_ID="local-3090-$(hostname -s)"
export WORK_ROOT="/tmp/arem-worker-local"
# Local worker polls aggressively so it claims new jobs within a few
# seconds of them landing — that's how it stays "ahead" of RunPod's
# overflow threshold for steady-state load.
export IDLE_SLEEP="${IDLE_SLEEP:-5}"
mkdir -p "$WORK_ROOT"

# Pipeline paths
export AREM_REPO="$(pwd)/pipeline"
export UPRIGHT_REPO="$(pwd)/pipeline"
export CLASSIFIER_PATH="$(pwd)/pipeline/classifier_v2.pth"
export CHECKPOINT_STAGE1="$(pwd)/checkpoints/stage1_jxl_v1_best_lpips.pth"
export CHECKPOINT_INTERIOR="$(pwd)/checkpoints/may26_interior_w32_b4_4gpu_ep35_inference.pth"
export CHECKPOINT_EXTERIOR="$(pwd)/checkpoints/may26_exterior_w32_b4_4gpu_ep29_inference.pth"

# Python interpreter (use the conda env that has all deps already)
export PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"

echo "[run_local] worker_id=$WORKER_ID"
echo "[run_local] dashboard=$DASHBOARD_URL"
echo "[run_local] ckpts: $(ls -1 checkpoints/ | wc -l) files"
echo "[run_local] python: $PYTHON_BIN"
echo

exec "$PYTHON_BIN" worker.py "$@"
