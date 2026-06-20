#!/bin/bash
# Daily fleet self-heal — runs on each node via arem-selfheal.timer. Keeps the
# fleet self-updating + self-healing with no manual intervention:
#   1. git pull the worker code (ff-only; never clobber local divergence)
#   2. reconcile PINNED critical pip deps (no surprise upgrades)
#   3. heal missing/corrupt model files from R2 (canonical source, sha-verified)
#   4. restart managed services — ONLY if code/deps/models actually changed
#      (so a no-op day causes zero disruption; runs ~4am, staggered)
#   5. verify each service is healthy; ROLL BACK the code on failure; report
#      to the dashboard (which alerts Discord on any degraded/rollback)
# Safe by construction: pinned deps, sha-verified models, ff-only git,
# rollback-on-failure, queue-based workers (a brief restart re-processes,
# never loses a job). Born from the Compute-2 sky-swap outage (2026-06-20)
# where a node silently ran for days missing onnxruntime + the sky model.
set -uo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
FLEET="$REPO/deploy/fleet"
[ -f /etc/arem-worker.env ] && { set -a; source /etc/arem-worker.env; set +a; }
DASHBOARD_URL="${DASHBOARD_URL:-https://arem-editing-dashboard.vercel.app}"
PY="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"
PIP="$(dirname "$PY")/pip"
HOST="$(hostname -s)"
LOG=/home/jordan/fleet_selfheal.log
exec >>"$LOG" 2>&1
echo "===== self-heal $(date -u +%FT%TZ) on $HOST ====="

FAILURES=(); HEALED=()

# single-instance lock
exec 9>/tmp/arem-selfheal.lock
flock -n 9 || { echo "another self-heal running; exit"; exit 0; }

# 1. code update (ff-only — refuse to clobber a hand-edited node)
BEFORE="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
git fetch -q origin || FAILURES+=("git fetch failed")
if ! git merge -q --ff-only origin/main 2>/dev/null; then
  echo "WARN: ff-only merge not possible (local diverged?) — leaving code as-is"
fi
AFTER="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
[ "$BEFORE" != "$AFTER" ] && { echo "code $BEFORE -> $AFTER"; HEALED+=("code:${AFTER:0:8}"); }

# 2. reconcile PINNED critical deps
if [ -f "$FLEET/critical-deps.txt" ]; then
  PIPOUT="$("$PIP" install -r "$FLEET/critical-deps.txt" 2>&1)"
  echo "$PIPOUT" | tail -3
  echo "$PIPOUT" | grep -qE "Successfully installed|Uninstalling" && HEALED+=("deps")
  echo "$PIPOUT" | grep -qiE "ERROR:|Could not find|No matching" && FAILURES+=("pip reconcile error")
fi

# 3. heal model files from R2 (path \t r2key \t sha256)
if [ -f "$FLEET/models.tsv" ]; then
  while IFS=$'\t' read -r mpath r2key msha; do
    [ -z "${mpath:-}" ] && continue; case "$mpath" in \#*) continue;; esac
    cur=""; [ -f "$mpath" ] && cur="$(sha256sum "$mpath" 2>/dev/null | cut -d' ' -f1)"
    if [ "$cur" != "$msha" ]; then
      echo "healing model $mpath (have:${cur:0:8} want:${msha:0:8})"
      mkdir -p "$(dirname "$mpath")"
      if rclone copyto "r2:$r2key" "$mpath" 2>/dev/null && \
         [ "$(sha256sum "$mpath" 2>/dev/null | cut -d' ' -f1)" = "$msha" ]; then
        HEALED+=("model:$(basename "$mpath")")
      else
        FAILURES+=("model heal failed: $(basename "$mpath")")
      fi
    fi
  done < "$FLEET/models.tsv"
fi

# 4. restart managed services — only if something actually changed
NEED_RESTART=0
[ "$BEFORE" != "$AFTER" ] && NEED_RESTART=1
[ ${#HEALED[@]} -gt 0 ] && NEED_RESTART=1
[ "${1:-}" = "--force-restart" ] && NEED_RESTART=1
restart_all() {
  [ -f "$FLEET/services.txt" ] || return
  while read -r svc; do
    [ -z "${svc:-}" ] && continue; case "$svc" in \#*) continue;; esac
    systemctl list-unit-files "$svc.service" --no-legend 2>/dev/null | grep -q . || continue
    echo "restart $svc"; sudo systemctl restart "$svc.service" 2>/dev/null || FAILURES+=("restart failed: $svc")
  done < "$FLEET/services.txt"
}
if [ "$NEED_RESTART" = 1 ]; then restart_all; else echo "no change — no restart"; fi

# 5. verify health
sleep 6
if [ -f "$FLEET/services.txt" ]; then
  while read -r svc; do
    [ -z "${svc:-}" ] && continue; case "$svc" in \#*) continue;; esac
    systemctl list-unit-files "$svc.service" --no-legend 2>/dev/null | grep -q . || continue
    st="$(systemctl is-active "$svc.service" 2>/dev/null || echo unknown)"
    [ "$st" = active ] || FAILURES+=("$svc not active ($st)")
  done < "$FLEET/services.txt"
fi

# 5b. roll back code if a failure appeared AND we changed code this run
ROLLED_BACK=false
if [ ${#FAILURES[@]} -gt 0 ] && [ "$BEFORE" != "$AFTER" ] && [ "$BEFORE" != unknown ]; then
  echo "FAILURE after code update — rolling back $AFTER -> $BEFORE"
  git reset --hard "$BEFORE" >/dev/null 2>&1 && restart_all && ROLLED_BACK=true
fi

# 6. report to the dashboard (records per host + alerts Discord on degraded)
STATUS="ok"; [ ${#FAILURES[@]} -gt 0 ] && STATUS="degraded"
tojson() { printf '%s\n' "${@:-}" | sed '/^$/d' | "$PY" -c 'import sys,json;print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))' 2>/dev/null || echo '[]'; }
curl -fsS -m 25 -X POST "$DASHBOARD_URL/api/internal/fleet-heal" \
  -H "x-worker-token: ${WORKER_TOKEN:-}" -H "content-type: application/json" \
  -d "{\"host\":\"$HOST\",\"status\":\"$STATUS\",\"gitBefore\":\"$BEFORE\",\"gitAfter\":\"$AFTER\",\"rolledBack\":$ROLLED_BACK,\"healed\":$(tojson "${HEALED[@]:-}"),\"failures\":$(tojson "${FAILURES[@]:-}")}" \
  >/dev/null 2>&1 || echo "report POST failed (non-fatal)"

echo "self-heal done: status=$STATUS healed=${#HEALED[@]} failures=${#FAILURES[@]} rolledBack=$ROLLED_BACK"
