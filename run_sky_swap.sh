#!/bin/bash
# Launch the standalone sky-swap worker on the 3090.
#
# Polls R2 sky-swap-jobs/requests/ for jobs queued by the production
# pipeline. For each, runs v9 sky-replace and uploads to a Dropbox
# subfolder + a new R2 key. Heartbeat + remote restart mirror the
# inpaint sandbox pattern.
set -euo pipefail
cd "$(dirname "$0")"

# systemd injects /etc/arem-worker.env via EnvironmentFile; for manual
# runs source it.
if [ -f /etc/arem-worker.env ]; then
  set -a; source /etc/arem-worker.env; set +a
fi
: "${WORKER_TOKEN:?WORKER_TOKEN not set — see /etc/arem-worker.env}"

export RCLONE_R2="${RCLONE_R2:-r2}"
export R2_BUCKET="${R2_BUCKET:-arem-training-data}"
export SKYSWAP_IDLE_UNLOAD_SEC="${SKYSWAP_IDLE_UNLOAD_SEC:-300}"

PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"

echo "[run_sky_swap] python=$PYTHON_BIN idle_unload=${SKYSWAP_IDLE_UNLOAD_SEC}s"

if [ "$#" -eq 0 ]; then
  set -- --loop --interval 5
fi
exec "$PYTHON_BIN" -m scripts.sky_swap_worker "$@"
