#!/bin/bash
# Container entrypoint: configure rclone, download checkpoints (one-shot per
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

# ---- Restormer checkpoints (cold-start one-time download) ----
mkdir -p /workspace/checkpoints
if [ ! -f "/workspace/checkpoints/interior_full_v1_latest.pth" ]; then
  echo "[entrypoint] downloading interior checkpoint from R2..."
  rclone copyto "r2:arem-training-data/checkpoints/interior_full_v1/latest.pth" \
                "/workspace/checkpoints/interior_full_v1_latest.pth" \
                --progress=false
fi
if [ ! -f "/workspace/checkpoints/exterior_full_v1_latest.pth" ]; then
  echo "[entrypoint] downloading exterior checkpoint from R2..."
  rclone copyto "r2:arem-training-data/checkpoints/exterior_full_v1/latest.pth" \
                "/workspace/checkpoints/exterior_full_v1_latest.pth" \
                --progress=false
fi
echo "[entrypoint] checkpoints ready"

exec python -u /workspace/handler.py
