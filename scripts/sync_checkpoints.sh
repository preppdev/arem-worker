#!/bin/bash
# Sync model checkpoints from R2 (bucket: arem-training-data) to the
# worker box. Required for the worker to actually run inference — the
# .pth files are too large for git.
#
# Reads R2 creds from ~/.config/rclone/rclone.conf (set up by
# provision.sh step 6).
#
# Idempotent: rclone copies only files that don't exist locally or
# differ in size. Skipped silently if checkpoints are already current.
#
# Usage:
#   bash $HOME/arem-worker/scripts/sync_checkpoints.sh

set -euo pipefail

WORKER_DIR="${WORKER_DIR:-$HOME/arem-worker}"
CKPT_DIR="$WORKER_DIR/checkpoints"
R2_REMOTE="${R2_REMOTE:-r2}"
R2_BUCKET="${R2_BUCKET:-arem-training-data}"

GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'; RED=$'\033[31m'; RESET=$'\033[0m'
say()  { printf "${CYAN}[checkpoints]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[warn]${RESET}   %s\n" "$*"; }
die()  { printf "${RED}[fatal]${RESET}  %s\n" "$*" >&2; exit 1; }

command -v rclone >/dev/null || die "rclone not installed (apt-get install -y rclone)"
[ -f "$HOME/.config/rclone/rclone.conf" ] || die "no rclone config at ~/.config/rclone/rclone.conf"

mkdir -p "$CKPT_DIR"

# The three checkpoints the worker pipeline reads (see run_local.sh).
# Keep these aligned with the CHECKPOINT_* env exports there.
FILES=(
  "stage1_jxl_v1_best_lpips.pth"
  "may26_interior_w32_b4_4gpu_ep35_inference.pth"
  "may26_exterior_w32_b4_4gpu_ep29_inference.pth"
)

# Plus the room/interior classifier (small; lives next to pipeline code).
CLASSIFIER_DEST="$WORKER_DIR/pipeline/classifier_v2.pth"

say "syncing 3 checkpoints + classifier from $R2_REMOTE:$R2_BUCKET"
for f in "${FILES[@]}"; do
  if [ -f "$CKPT_DIR/$f" ]; then
    say "  ✓ $f already present"
    continue
  fi
  # Try stage1 path first, then stage2 paths
  for src in "checkpoints/stage1_jxl_full_v1/best_lpips.pth" "models/stage2/$f" "checkpoints/$f"; do
    if rclone copyto -q "$R2_REMOTE:$R2_BUCKET/$src" "$CKPT_DIR/$f" 2>/dev/null; then
      say "  ↓ $f (from $src)"
      break
    fi
  done
  [ -f "$CKPT_DIR/$f" ] || warn "could not locate $f in R2"
done

# Classifier — small file, lives in pipeline/
if [ ! -f "$CLASSIFIER_DEST" ]; then
  if rclone copyto -q "$R2_REMOTE:$R2_BUCKET/models/classifier/classifier_v2.pth" "$CLASSIFIER_DEST" 2>/dev/null; then
    say "  ↓ classifier_v2.pth → pipeline/"
  else
    warn "could not pull classifier_v2.pth"
  fi
fi

# Summary
echo
say "checkpoints in $CKPT_DIR:"
ls -lh "$CKPT_DIR" 2>/dev/null | tail -n +2 | awk '{printf "  %s  %s\n", $5, $9}'
if [ -f "$CLASSIFIER_DEST" ]; then
  size=$(ls -lh "$CLASSIFIER_DEST" | awk '{print $5}')
  echo "  $size  pipeline/classifier_v2.pth"
fi
