#!/bin/bash
# arem-worker: pull the latest repo changes and re-install anything that
# moved (unit files, autostart, requirements.txt). Run periodically on
# each deployed worker box.
#
# Idempotent. Doesn't touch credentials, conda env, or model checkpoints
# (those are stable). If you need to also refresh checkpoints, run
# scripts/sync_checkpoints.sh separately.
#
# Usage:
#   bash $HOME/arem-worker/update.sh

set -euo pipefail

WORKER_DIR="${WORKER_DIR:-$HOME/arem-worker}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-arem-photo-ai}"

GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
say()  { printf "${CYAN}[update]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[ok]${RESET}     %s\n" "$*"; }
warn() { printf "${YELLOW}[warn]${RESET}   %s\n" "$*"; }

[ -d "$WORKER_DIR/.git" ] || { echo "no repo at $WORKER_DIR — run provision.sh first"; exit 1; }
cd "$WORKER_DIR"

# ── 1. Pull ────────────────────────────────────────────────────────────────
say "git pull"
PREV_HEAD=$(git rev-parse HEAD)
git pull --ff-only origin main 2>&1 | tail -3
NEW_HEAD=$(git rev-parse HEAD)
if [ "$PREV_HEAD" = "$NEW_HEAD" ]; then
  ok "already up-to-date"
fi

# ── 2. Re-sync systemd units (only when changed) ───────────────────────────
say "systemd units"
NEEDS_RELOAD=false
for unit in arem-worker-local.service arem-host-heartbeat.service arem-host-heartbeat.timer; do
  if ! cmp -s "$unit" "/etc/systemd/system/$unit" 2>/dev/null; then
    say "  $unit changed; installing"
    sudo install -m 0644 "$unit" "/etc/systemd/system/$unit"
    NEEDS_RELOAD=true
  fi
done
if [ -f arem-worker-sudoers ] && ! cmp -s arem-worker-sudoers /etc/sudoers.d/arem-worker 2>/dev/null; then
  say "  sudoers changed; installing"
  sudo install -m 0440 arem-worker-sudoers /etc/sudoers.d/arem-worker
  sudo visudo -c -f /etc/sudoers.d/arem-worker >/dev/null
fi
if $NEEDS_RELOAD; then
  sudo systemctl daemon-reload
fi
ok "units"

# ── 3. Re-sync autostart ───────────────────────────────────────────────────
say "autostart entry"
if [ -f arem-fleet-tui.desktop ]; then
  mkdir -p "$HOME/.config/autostart"
  if ! cmp -s arem-fleet-tui.desktop "$HOME/.config/autostart/arem-fleet-tui.desktop" 2>/dev/null; then
    install -m 0755 arem-fleet-tui.desktop "$HOME/.config/autostart/arem-fleet-tui.desktop"
    say "  reinstalled (will pick up on next GNOME login)"
  fi
fi

# ── 4. Python requirements ─────────────────────────────────────────────────
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME" 2>/dev/null && {
    say "pip install -r requirements.txt"
    pip install --quiet -r requirements.txt
    conda deactivate
  }
else
  warn "conda not found at $CONDA_ROOT — skipping requirements.txt sync"
fi

# ── 5. Restart worker so it picks up new code ──────────────────────────────
say "restart arem-worker-local (kills any in-flight job)"
sudo systemctl restart arem-worker-local.service
sleep 2
systemctl is-active --quiet arem-worker-local && ok "worker restarted clean" || warn "worker not active — check journalctl -u arem-worker-local -n 30"

echo
ok "update complete ($PREV_HEAD → $NEW_HEAD)"
echo "  Verify on /workers within 30s"
