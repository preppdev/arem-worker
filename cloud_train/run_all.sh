#!/bin/bash
# Sequential B200 training run: router -> room -> detector, all at native res.
# Pod env needs: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
# Results land in r2:arem-training-data/cloud-train/runs/<UTC date>/
set -euo pipefail
STAMP=$(date -u +%Y%m%d-%H%M)
R2DEST="r2:arem-training-data/cloud-train/runs/$STAMP"
export CLOUD_DATA=/workspace/data

echo "=== [0/5] setup"
command -v rclone >/dev/null || (curl -fsSL https://rclone.org/install.sh | bash)
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
EOF
pip install -q --no-input pillow 2>/dev/null || true

echo "=== [1/5] pull dataset (~170GB)"
mkdir -p $CLOUD_DATA
rclone copy r2:arem-training-data/cloud-train/meta $CLOUD_DATA/meta --transfers 16
rclone copy r2:arem-training-data/cloud-train/detector $CLOUD_DATA/detector --transfers 8
rclone copy r2:arem-training-data/cloud-train/originals $CLOUD_DATA/originals --transfers 64 --checkers 64 --stats 60s --stats-log-level NOTICE
echo "pulled: $(find $CLOUD_DATA/originals -name '*.jpg' | wc -l) images"

cd "$(dirname "$0")"
# background log shipper: push live logs to R2 every 4 min for remote monitoring
( while true; do sleep 240; rclone copy /workspace --include "*.log" $R2DEST/live-logs 2>/dev/null || true; done ) &
LOGSHIP_PID=$!
trap "kill $LOGSHIP_PID 2>/dev/null" EXIT
run_and_ship () {  # name script outdir
  echo "=== $1 START $(date -u +%H:%M)"
  OUT=$3 python3 $2 2>&1 | tee /workspace/$1.log
  rclone copy $3 $R2DEST/$1 --transfers 8
  rclone copyto /workspace/$1.log $R2DEST/$1/train.log
  echo "=== $1 SHIPPED -> $R2DEST/$1"
}

# stage skip logic: a stage with an existing checkpoint is shipped, not
# retrained (crash-resume + operator early-accept, e.g. router 2026-06-12)
stage () {  # name script outdir ckpt
  if [ -f "$4" ]; then
    echo "=== $1: existing checkpoint accepted — shipping without retraining"
    rclone copy "$3" "$R2DEST/$1" --transfers 8
  else
    run_and_ship "$1" "$2" "$3"
  fi
}

echo "=== [2/5] router (ResNet-18 native)"
stage router train_router_native.py /workspace/runs/router_native /workspace/runs/router_native/router_v4_native.pth

echo "=== [3/5] room classifier (ConvNeXt-Base native)"
stage room train_room_native.py /workspace/runs/room_native_b /workspace/runs/room_native_b/best.pt

echo "=== [4/5] detector (FRCNN native)"
stage detector train_detector_native.py /workspace/runs/detector_native /workspace/runs/detector_native/best.pth

echo "=== [5/5] ALL_DONE $R2DEST"
# self-stop the pod if runpodctl is available (saves billing on completion)
runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
