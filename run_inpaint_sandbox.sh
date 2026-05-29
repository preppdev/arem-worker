#!/bin/bash
# Launch the inpaint-sandbox daemon on the 3090. Polls an R2 queue
# (inpaint-sandbox/requests/) for dashboard-drawn boxes, turns each into a
# tight SAM2 mask, runs the local inpainters (Flux Fill + ObjectClear),
# and writes a side-by-side composite back to R2.
#
# Models are loaded lazily on the first request and unloaded after
# SANDBOX_IDLE_UNLOAD_SEC of inactivity, so this can run always-on
# alongside arem-worker-local + arem-enhancement-local without
# permanently holding ~10 GB of the shared GPU.
#
# Usage:
#   bash run_inpaint_sandbox.sh           # poll forever (default)
#   bash run_inpaint_sandbox.sh ""        # (pass empty arg for one poll cycle)
set -euo pipefail
cd "$(dirname "$0")"

# systemd injects /etc/arem-worker.env via EnvironmentFile; for manual
# runs, source it ourselves.
if [ -f /etc/arem-worker.env ]; then
  set -a; source /etc/arem-worker.env; set +a
fi
: "${WORKER_TOKEN:?WORKER_TOKEN not set — see /etc/arem-worker.env}"

# FLUX weights are local, but export HF_TOKEN in case anything resolves
# a gated repo at load time.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# Bria Eraser API key (premium full-res object removal). Kept outside the
# repo; export it for the daemon if present.
if [ -z "${BRIA_API_KEY:-}" ] && [ -f "$HOME/.bria_api_key" ]; then
  export BRIA_API_KEY="$(cat "$HOME/.bria_api_key")"
fi

export RCLONE_R2="${RCLONE_R2:-r2}"
export R2_BUCKET="${R2_BUCKET:-arem-training-data}"
export SANDBOX_IDLE_UNLOAD_SEC="${SANDBOX_IDLE_UNLOAD_SEC:-300}"

PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"

echo "[run_inpaint_sandbox] python=$PYTHON_BIN idle_unload=${SANDBOX_IDLE_UNLOAD_SEC}s"

# Default to --loop unless the caller passes its own args (e.g. --once).
if [ "$#" -eq 0 ]; then
  set -- --loop --interval 5
fi
exec "$PYTHON_BIN" -m scripts.inpaint_sandbox_worker "$@"
