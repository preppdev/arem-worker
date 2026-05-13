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

set -euo pipefail

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
