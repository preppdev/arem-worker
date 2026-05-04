#!/bin/bash
# Container entrypoint: configure rclone, fetch checkpoints (one-shot per
# worker), then start the RunPod serverless handler.

set -euo pipefail

# ---- rclone config from env (R2 + Dropbox) ----
mkdir -p /root/.config/rclone
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
token = ${RCLONE_DROPBOX_TOKEN}
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
fetch "r2:arem-training-data/checkpoints/stage1_jxl_v1/best_lpips.pth" \
      "/workspace/checkpoints/stage1_jxl_v1_best_lpips.pth"
fetch "r2:arem-training-data/checkpoints/interior_full_v1/latest.pth" \
      "/workspace/checkpoints/interior_full_v1_latest.pth"
fetch "r2:arem-training-data/checkpoints/exterior_full_v1/latest.pth" \
      "/workspace/checkpoints/exterior_full_v1_latest.pth"
echo "[entrypoint] checkpoints ready"

exec python -u /workspace/handler.py
