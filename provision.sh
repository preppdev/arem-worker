#!/bin/bash
# arem-worker: provision a fresh Ubuntu 22.04+ box as a production worker.
#
# Idempotent — safe to re-run. Designed to be the single command the operator
# runs after a fresh OS install:
#
#   curl -sSL https://raw.githubusercontent.com/preppdev/arem-worker/main/provision.sh | bash
#
# Or, if you've already scp'd it onto the box:
#
#   bash provision.sh
#
# What it does, in order:
#   1.  Preflight  — distro check, sudo, NVIDIA GPU present
#   2.  System deps — apt: build-essential, git, curl, rclone, jq, dkms,
#       Python 3.11, gnome-terminal (for the TUI display)
#   3.  NVIDIA driver — installs the 580-open cohort + the matching
#       per-kernel module package for the running kernel. Loads modules.
#   4.  Repo clone — git clone arem-worker → $WORKER_DIR (default: $HOME/arem-worker)
#   5.  Python env — miniconda3 if missing, then `arem-photo-ai` env with
#       requirements.txt installed.
#   6.  Credentials — stops here if /etc/arem-worker.env or rclone.conf is
#       missing, with copy-paste instructions for the two viable sources
#       (scp from existing box / vercel env pull from a dev machine).
#   7.  systemd units + sudoers — installs the .service / .timer / sudoers
#       files and enables them.
#   8.  TUI autostart — installs ~/.config/autostart/arem-fleet-tui.desktop
#       so the local status TUI launches on GNOME login.
#   9.  Model checkpoints — runs scripts/sync_checkpoints.sh (rclone from R2)
#       if the checkpoints aren't already present.
#   10. Verify — prints service status + a reminder to check /workers
#       on the dashboard.
#
# Re-runnability: each step skips if already done. So if you cancel in
# the middle (e.g. for credentials), just re-run.

set -euo pipefail

# ── config (override via env) ───────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/preppdev/arem-worker.git}"
WORKER_USER="${WORKER_USER:-$(whoami)}"
WORKER_HOME="${WORKER_HOME:-$HOME}"
WORKER_DIR="${WORKER_DIR:-$WORKER_HOME/arem-worker}"
CONDA_ROOT="${CONDA_ROOT:-$WORKER_HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-arem-photo-ai}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
NVIDIA_SERIES="${NVIDIA_SERIES:-580-open}"

# ── pretty printing ─────────────────────────────────────────────────────────
RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
say()  { printf "${CYAN}[provision]${RESET} %s\n" "$*"; }
ok()   { printf "${GREEN}[ok]${RESET}        %s\n" "$*"; }
warn() { printf "${YELLOW}[warn]${RESET}      %s\n" "$*"; }
die()  { printf "${RED}[fatal]${RESET}     %s\n" "$*" >&2; exit 1; }

# ── 1. Preflight ────────────────────────────────────────────────────────────
say "1/10 preflight"
[ "$EUID" -eq 0 ] && die "Run as your worker user, not root. The script sudo's when needed."
command -v sudo >/dev/null || die "sudo not installed"
command -v curl >/dev/null || die "curl not installed (apt-get install -y curl)"

. /etc/os-release 2>/dev/null || die "/etc/os-release missing — can't identify distro"
if [ "${ID:-}" != "ubuntu" ]; then
  warn "tested on Ubuntu 22.04+. Detected: ${ID:-?} ${VERSION_ID:-?}. Proceeding anyway."
fi

if ! lspci 2>/dev/null | grep -qi "nvidia"; then
  warn "no NVIDIA GPU detected via lspci. Worker will install but won't process jobs."
fi
ok "preflight"

# ── 2. System packages ─────────────────────────────────────────────────────
say "2/10 system packages (apt)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  build-essential git curl rclone jq dkms ca-certificates \
  "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev" \
  ubuntu-drivers-common \
  gnome-terminal
ok "apt packages"

# ── 3. NVIDIA driver ───────────────────────────────────────────────────────
say "3/10 NVIDIA driver ($NVIDIA_SERIES + matching kernel modules)"
KERN=$(uname -r)
need_driver=false
if ! lsmod | grep -q '^nvidia '; then
  need_driver=true
fi
if $need_driver; then
  warn "nvidia kernel module not loaded; installing"
  sudo apt-get install -y --no-install-recommends \
    "nvidia-driver-$NVIDIA_SERIES" \
    "linux-modules-nvidia-$NVIDIA_SERIES-$KERN" \
    "linux-modules-nvidia-$NVIDIA_SERIES-generic-hwe-22.04" 2>&1 | tail -20 \
    || warn "nvidia install reported issues — may still succeed after reboot"
  sudo modprobe nvidia 2>&1 || warn "modprobe nvidia failed — may need a reboot"
  sudo modprobe nvidia_uvm 2>&1 || true
fi
if nvidia-smi -L >/dev/null 2>&1; then
  ok "$(nvidia-smi -L | head -1)"
else
  warn "nvidia-smi not working yet. Reboot and re-run provision.sh."
fi

# ── 4. Clone repo ──────────────────────────────────────────────────────────
say "4/10 repo: $REPO_URL → $WORKER_DIR"
if [ ! -d "$WORKER_DIR/.git" ]; then
  git clone "$REPO_URL" "$WORKER_DIR"
else
  (cd "$WORKER_DIR" && git pull --ff-only origin main 2>&1 | tail -3)
fi
ok "repo at $WORKER_DIR"

# ── 5. Python env ──────────────────────────────────────────────────────────
say "5/10 conda env $CONDA_ENV_NAME"
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  say "  installing miniconda → $CONDA_ROOT"
  TMP_INST=$(mktemp -t miniconda.XXXXXX.sh)
  curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMP_INST"
  bash "$TMP_INST" -b -p "$CONDA_ROOT"
  rm "$TMP_INST"
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV_NAME"
pip install --quiet --upgrade pip
pip install --quiet -r "$WORKER_DIR/requirements.txt"
conda deactivate
ok "conda env $CONDA_ENV_NAME ready"

# ── 6. Credentials gate ────────────────────────────────────────────────────
say "6/10 credentials check"
missing_creds=()
[ -f /etc/arem-worker.env ] || missing_creds+=("/etc/arem-worker.env")
[ -f "$HOME/.config/rclone/rclone.conf" ] || missing_creds+=("~/.config/rclone/rclone.conf")
if [ ${#missing_creds[@]} -gt 0 ]; then
  cat <<EOF

${YELLOW}${BOLD}Credentials required before continuing:${RESET}
  Missing: ${missing_creds[*]}

These two files can't be in the public repo. Get them via either path:

${BOLD}(A) scp from an existing AREM worker box${RESET}
    Run on YOUR dev machine (laptop), which can reach both boxes:
      scp jordan@<existing-worker>:/etc/arem-worker.env /tmp/arem-worker.env
      scp jordan@<existing-worker>:~/.config/rclone/rclone.conf /tmp/rclone.conf
      scp /tmp/arem-worker.env /tmp/rclone.conf $WORKER_USER@$(hostname -I | awk '{print $1}'):/tmp/

${BOLD}(B) vercel env pull from a dev machine${RESET}
    On your laptop (with vercel CLI logged in + linked to arem-editing-dashboard):
      cd arem-editing/  # or wherever the dashboard repo lives
      vercel env pull /tmp/arem.env --environment=production --yes
      scp /tmp/arem.env $WORKER_USER@$(hostname -I | awk '{print $1}'):/tmp/arem-worker.env
    (rclone.conf still has to come via path A — it's not in Vercel env.)

Then back on THIS box:
    sudo install -m 0640 -o $WORKER_USER -g $WORKER_USER /tmp/arem-worker.env /etc/arem-worker.env
    mkdir -p ~/.config/rclone
    install -m 0600 /tmp/rclone.conf ~/.config/rclone/rclone.conf
    bash $WORKER_DIR/provision.sh    # re-run — steps 1-5 will be no-ops

EOF
  exit 0
fi
ok "credentials present"

# ── 7. systemd units + sudoers ─────────────────────────────────────────────
say "7/10 systemd units + sudoers"
for unit in arem-worker-local.service arem-host-heartbeat.service arem-host-heartbeat.timer; do
  if ! cmp -s "$WORKER_DIR/$unit" "/etc/systemd/system/$unit" 2>/dev/null; then
    sudo install -m 0644 "$WORKER_DIR/$unit" "/etc/systemd/system/$unit"
  fi
done
sudo install -m 0440 "$WORKER_DIR/arem-worker-sudoers" /etc/sudoers.d/arem-worker
sudo visudo -c -f /etc/sudoers.d/arem-worker >/dev/null || die "sudoers syntax check failed"
sudo systemctl daemon-reload
sudo systemctl enable --now arem-worker-local.service
sudo systemctl enable --now arem-host-heartbeat.timer
ok "systemd units enabled + active"

# ── 8. TUI autostart ───────────────────────────────────────────────────────
say "8/10 TUI autostart"
mkdir -p "$HOME/.config/autostart"
install -m 0755 "$WORKER_DIR/arem-fleet-tui.desktop" "$HOME/.config/autostart/arem-fleet-tui.desktop"
ok "TUI will auto-launch on GNOME login"

# ── 9. Model checkpoints ───────────────────────────────────────────────────
say "9/10 model checkpoints"
if [ -x "$WORKER_DIR/scripts/sync_checkpoints.sh" ]; then
  bash "$WORKER_DIR/scripts/sync_checkpoints.sh" || warn "checkpoint sync reported issues — check rclone.conf"
else
  warn "scripts/sync_checkpoints.sh missing — re-pull the repo or fetch checkpoints manually"
fi

# ── 10. Verify + summary ───────────────────────────────────────────────────
say "10/10 verify"
echo
echo "${BOLD}services:${RESET}"
systemctl is-active --quiet arem-worker-local      && echo "  ${GREEN}● arem-worker-local       active${RESET}"      || echo "  ${RED}● arem-worker-local       $(systemctl is-active arem-worker-local)${RESET}"
systemctl is-active --quiet arem-host-heartbeat.timer && echo "  ${GREEN}● arem-host-heartbeat     active${RESET}"  || echo "  ${RED}● arem-host-heartbeat     $(systemctl is-active arem-host-heartbeat.timer)${RESET}"
echo
echo "${BOLD}post-install:${RESET}"
echo "  - Within 60s, this box should appear on https://arem-editing-dashboard.vercel.app/workers"
echo "  - First job claim happens whenever Dropbox queue depth > 0"
echo "  - Update later with: bash $WORKER_DIR/update.sh"
echo "  - View live TUI:     bash -c 'watch -n 2 -t -c python3 $WORKER_DIR/arem-fleet-tui.py'"
echo
ok "provisioning complete"
