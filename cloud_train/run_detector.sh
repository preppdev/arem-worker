#!/bin/bash
# Detector-only B200 run (parallel pod). Pulls just the annotated subset.
set -euo pipefail
STAMP=$(date -u +%Y%m%d-%H%M)
R2DEST="r2:arem-training-data/cloud-train/runs/$STAMP-detector"
export CLOUD_DATA=/workspace/data
mkdir -p $CLOUD_DATA/detector
rclone copy r2:arem-training-data/cloud-train/detector $CLOUD_DATA/detector --transfers 8
python3 - <<'PY'
import json
paths = set()
for split in ("train", "val"):
    for r in json.load(open(f"/workspace/data/detector/ann_{split}_native.json")):
        paths.add(r["path"])
open("/tmp/det_files.txt", "w").write("\n".join(sorted(paths)))
print(len(paths), "files needed")
PY
rclone copy r2:arem-training-data/cloud-train/originals $CLOUD_DATA/originals --files-from /tmp/det_files.txt --transfers 32
echo "pulled: $(find $CLOUD_DATA/originals -name '*.jpg' | wc -l) images"
( while true; do sleep 240; rclone copy /workspace --include "*.log" $R2DEST/live-logs 2>/dev/null || true; done ) &
trap "kill $! 2>/dev/null" EXIT
cd "$(dirname "$0")"
echo "=== detector START $(date -u +%H:%M)"
OUT=/workspace/runs/detector_native python3 train_detector_native.py 2>&1 | tee /workspace/detector.log
grep -q "_DONE" /workspace/detector.log && touch /workspace/runs/detector_native/.done
rclone copy /workspace/runs/detector_native $R2DEST/detector --transfers 8
rclone copyto /workspace/detector.log $R2DEST/detector/train.log
echo "=== DETECTOR_SHIPPED $R2DEST"
runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
sleep 300
