#!/bin/bash
# Container entrypoint: configure rclone, fetch checkpoints (one-shot per
# worker), then dispatch on MODE.
#
# MODE values:
#   unset / "handler"      → production RunPod serverless handler (default)
#   "backfill_stage1"      → run scripts/backfill_stage1.py with OFFSET +
#                            LIMIT + optional SINCE/FORCE env. One-shot
#                            pod: processes its slice of candidate Jobs
#                            and exits.
#
# Logging: all stdout+stderr is tee'd to /tmp/entrypoint.log, and on EXIT
# the log is uploaded to r2:arem-training-data/debug-logs/<hostname>-<ts>.log
# so failed runs are diagnosable post-hoc (no SSH needed). The upload is
# best-effort; if rclone isn't configured yet (early entrypoint failure)
# the log just sits on the dying container.

set -euo pipefail

LOG_FILE=/tmp/entrypoint.log
exec > >(tee -a "$LOG_FILE") 2>&1

upload_log() {
  local rc=$?
  echo "[entrypoint] exit rc=$rc — uploading log to R2"
  if command -v rclone >/dev/null && [ -f /root/.config/rclone/rclone.conf ]; then
    local ts; ts=$(date -u +%Y%m%dT%H%M%SZ)
    local host; host=$(hostname 2>/dev/null || echo "unknown")
    rclone copyto "$LOG_FILE" \
      "r2:arem-training-data/debug-logs/${host}-${ts}-rc${rc}.log" \
      --retries 1 --timeout 30s 2>/dev/null || true
  fi
  return $rc
}
trap upload_log EXIT

# ---- rclone config from env (R2 + Dropbox) ----
mkdir -p /root/.config/rclone
# Decode the Dropbox token from base64 (bypasses env-var character mangling
# we observed when passing the raw JSON blob through RunPod's env).
if [ -n "${DROPBOX_TOKEN_B64:-}" ]; then
  DROPBOX_TOKEN_JSON=$(printf '%s' "$DROPBOX_TOKEN_B64" | base64 -d)
else
  # Back-compat: accept the raw JSON if someone sets it that way.
  DROPBOX_TOKEN_JSON="${RCLONE_DROPBOX_TOKEN:-}"
fi
echo "[entrypoint] dropbox token len: ${#DROPBOX_TOKEN_JSON} starts: ${DROPBOX_TOKEN_JSON:0:1}"

cat > /root/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
acl = private

[dropbox]
type = dropbox
token = ${DROPBOX_TOKEN_JSON}
EOF
chmod 600 /root/.config/rclone/rclone.conf

# Immediately drop an "alive" marker into R2 before the heavy checkpoint
# downloads. Tells us: did we successfully configure rclone? Without this
# marker, an early failure (OOM during a 12GB checkpoint download,
# disk-full kill) gives us no visibility because the EXIT trap can be
# pre-empted by SIGKILL.
host=$(hostname 2>/dev/null || echo unknown)
ts_start=$(date -u +%Y%m%dT%H%M%SZ)
printf '[%s] entrypoint starting on %s\nMODE=%s OFFSET=%s LIMIT=%s SINCE=%s\nMEM:\n%s\nDISK:\n%s\n' \
  "$ts_start" "$host" "${MODE:-}" "${OFFSET:-}" "${LIMIT:-}" "${SINCE:-}" \
  "$(free -h 2>&1 || echo n/a)" \
  "$(df -h 2>&1 || echo n/a)" \
  > /tmp/startup.log
rclone copyto /tmp/startup.log \
  "r2:arem-training-data/debug-logs/${host}-${ts_start}-startup.log" \
  --retries 1 --timeout 30s 2>/dev/null || \
  echo "[entrypoint] WARN: failed to upload startup marker"
echo "[entrypoint] startup marker uploaded"

# Continuous log-uploader: pushes /tmp/entrypoint.log to R2 every 5s
# while the container runs. The EXIT trap also uploads, but if the
# container is SIGKILL'd by RunPod's manager (startup timeout, OOM,
# etc.) the trap doesn't fire — so without this tail-uploader we
# would never see python's stderr from a crashed worker. The
# upload is best-effort and silent on failure; if rclone hiccups we
# just try again on the next tick.
(
  while sleep 5; do
    [ -s "$LOG_FILE" ] || continue
    rclone copyto "$LOG_FILE" \
      "r2:arem-training-data/debug-logs/${host}-${ts_start}-live.log" \
      --retries 1 --timeout 10s 2>/dev/null || true
  done
) &
UPLOADER_PID=$!

# Best-effort: stop the uploader when the foreground process exits
# normally so we don't leave it running after a clean handler shutdown.
trap "kill $UPLOADER_PID 2>/dev/null || true; upload_log" EXIT

# ---- Checkpoints (one-time download per cold-started worker) ----
mkdir -p /workspace/checkpoints
fetch() {
  local r2_path="$1"; local local_path="$2"
  if [ -f "$local_path" ]; then return; fi
  echo "[entrypoint] downloading $r2_path → $local_path"
  rclone copyto "$r2_path" "$local_path" --progress=false
}
fetch "r2:arem-training-data/checkpoints/stage1_jxl_full_v1/best_lpips.pth" \
      "/workspace/checkpoints/stage1_jxl_v1_best_lpips.pth"
fetch "r2:arem-training-data/models/stage2/may26_interior_w32_b4_4gpu_ep35_inference.pth" \
      "/workspace/checkpoints/may26_interior_w32_b4_4gpu_ep35_inference.pth"
fetch "r2:arem-training-data/models/stage2/may26_exterior_w32_b4_4gpu_ep29_inference.pth" \
      "/workspace/checkpoints/may26_exterior_w32_b4_4gpu_ep29_inference.pth"
echo "[entrypoint] checkpoints ready"

case "${MODE:-handler}" in
  handler)
    echo "[entrypoint] starting RunPod serverless handler"
    exec python -u /workspace/handler.py
    ;;
  backfill_stage1)
    echo "[entrypoint] backfill_stage1 mode  OFFSET=${OFFSET:-0}  LIMIT=${LIMIT:-50}  SINCE=${SINCE:-default}"
    cd /workspace
    ARGS=(--offset "${OFFSET:-0}" --limit "${LIMIT:-50}")
    if [ -n "${SINCE:-}" ]; then ARGS+=(--since "$SINCE"); fi
    if [ -n "${FORCE:-}" ] && [ "${FORCE}" != "0" ]; then ARGS+=(--force); fi
    exec python -u -m scripts.backfill_stage1 "${ARGS[@]}"
    ;;
  *)
    echo "[entrypoint] ERROR: unknown MODE: $MODE"
    exit 2
    ;;
esac
