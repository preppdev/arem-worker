#!/bin/bash
# AREM dev environment bootstrap.
#
# Run on a fresh dev machine (laptop / VPS / cloud shell) to get a
# working clone of the project: repos, Node deps, Python env, Vercel
# link, scratch dirs. Secret material (DB URLs, rclone tokens, SSH
# keys) is NOT copied — those need to come from your secure store.
#
# Tested on Ubuntu 22.04/24.04 and macOS 14+. Should work on any
# POSIX shell with git, curl, and a reachable network.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/preppdev/arem-worker/main/dev_bootstrap.sh | bash
#   or copy this file locally and: bash dev_bootstrap.sh

set -euo pipefail

# ── config ──────────────────────────────────────────────────────────────
DASHBOARD_REPO="https://github.com/preppdev/arem-editing.git"      # the dashboard
WORKER_REPO="https://github.com/preppdev/arem-worker.git"          # this repo
DEV_ROOT="${DEV_ROOT:-$HOME/arem-dev}"                              # where to clone
PYTHON_VENV="${PYTHON_VENV:-$HOME/arem-dev/.venv}"                   # python env
NODE_VERSION_MIN="20"
PYTHON_VERSION_MIN="3.11"

GREEN=$(printf '\033[32m'); RED=$(printf '\033[31m'); YELLOW=$(printf '\033[33m'); RESET=$(printf '\033[0m')
say() { printf "${GREEN}[bootstrap]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[bootstrap]${RESET} %s\n" "$*"; }
die() { printf "${RED}[bootstrap]${RESET} %s\n" "$*"; exit 1; }

# ── preflight ──────────────────────────────────────────────────────────
say "preflight checks"
command -v git >/dev/null || die "git not installed"
command -v curl >/dev/null || die "curl not installed"

if ! command -v node >/dev/null; then
  warn "node not installed — install Node $NODE_VERSION_MIN+ first"
  warn "  Ubuntu: sudo apt install nodejs npm   (or use nvm for newer versions)"
  warn "  macOS:  brew install node"
  die "exit"
fi
NODE_MAJOR=$(node --version | sed 's/^v//;s/\..*//')
[ "$NODE_MAJOR" -ge "$NODE_VERSION_MIN" ] || die "Node $NODE_VERSION_MIN+ required (have v$NODE_MAJOR)"

if ! command -v python3 >/dev/null; then
  die "python3 not installed"
fi
PY_MAJ=$(python3 -c 'import sys;print(sys.version_info[0])')
PY_MIN=$(python3 -c 'import sys;print(sys.version_info[1])')
if [ "$PY_MAJ" -lt 3 ] || { [ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 11 ]; }; then
  die "Python $PYTHON_VERSION_MIN+ required (have ${PY_MAJ}.${PY_MIN})"
fi

if ! command -v vercel >/dev/null; then
  warn "vercel CLI not installed — installing globally"
  npm install -g vercel
fi

if ! command -v rclone >/dev/null; then
  warn "rclone not installed — needed for R2/Dropbox access"
  warn "  Ubuntu: sudo apt install rclone"
  warn "  macOS:  brew install rclone"
fi

# ── clone repos ────────────────────────────────────────────────────────
say "cloning repos to $DEV_ROOT"
mkdir -p "$DEV_ROOT"
cd "$DEV_ROOT"

for spec in "arem-editing-dashboard:$DASHBOARD_REPO" "arem-worker:$WORKER_REPO"; do
  dir="${spec%%:*}"; url="${spec#*:}"
  if [ -d "$dir/.git" ]; then
    say "  ✓ $dir (updating)"
    (cd "$dir" && git pull --ff-only origin main >/dev/null 2>&1 || true)
  else
    say "  ↓ $dir"
    git clone --depth 1 "$url" "$dir" >/dev/null
  fi
done

# ── dashboard deps ─────────────────────────────────────────────────────
say "installing dashboard Node deps"
cd "$DEV_ROOT/arem-editing-dashboard"
npm install --no-audit --no-fund >/dev/null
npx prisma generate >/dev/null
say "  ✓ npm install + prisma generate done"

# ── python env (optional; only needed for training/enrichment work) ───
if [ ! -d "$PYTHON_VENV" ]; then
  say "creating python venv at $PYTHON_VENV (worker + enrichment scripts)"
  python3 -m venv "$PYTHON_VENV"
fi
# shellcheck disable=SC1091
source "$PYTHON_VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet \
  'torch>=2.4' 'torchvision' 'numpy<2' 'pillow' 'opencv-python-headless' \
  'rawpy' 'exifread' 'psycopg2-binary' 'transformers>=5' 'safetensors' \
  '@neondatabase/serverless' 2>/dev/null || true
# Note: ML-heavy deps (segmentation_models_pytorch, albumentations, lensfunpy)
# only needed for training/enrichment scripts — install on demand.
deactivate
say "  ✓ python venv ready (light deps only; install ML deps as needed)"

# ── manual steps ──────────────────────────────────────────────────────
cat <<EOF

${GREEN}✓ bootstrap complete${RESET}

Repos at:        $DEV_ROOT
Python venv at:  $PYTHON_VENV  (activate with: source $PYTHON_VENV/bin/activate)

${YELLOW}Manual steps to finish setup:${RESET}

  1. Copy /tmp/db.env from the 3090 box (mode 0600):
       scp jordan@<3090-box>:/tmp/db.env /tmp/db.env
       chmod 600 /tmp/db.env

  2. Copy rclone config:
       mkdir -p ~/.config/rclone
       scp jordan@<3090-box>:~/.config/rclone/rclone.conf ~/.config/rclone/

  3. Link the Vercel project (one-time, paste the linking URL Vercel shows):
       cd $DEV_ROOT/arem-editing-dashboard
       vercel link

  4. Generate an SSH key + add to the 3090 box for remote control:
       ssh-keygen -t ed25519 -C "dev@<your-machine>"
       ssh-copy-id jordan@<3090-box>

  5. Test it works:
       cd $DEV_ROOT/arem-editing-dashboard
       set -a && source /tmp/db.env && set +a
       npx tsc --noEmit
       (should compile clean)

Once those four steps land, your dev machine has the same powers as
the 3090 box: dashboard work, DB queries, R2/Dropbox access, deploys.
GPU-bound jobs you SSH into the 3090 (or provision RunPod/Vast).

EOF
