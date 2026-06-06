#!/bin/bash
# Host-level heartbeat. Pings the dashboard once per minute with a JSON
# blob of host health metrics. Independent of the worker process — if
# this script can run, the box is up, network is reachable, and Vercel
# is responding.
#
# Installed as a systemd timer: arem-host-heartbeat.timer fires once a
# minute and runs arem-host-heartbeat.service which executes this script.

set -euo pipefail

# Load creds (WORKER_TOKEN, DASHBOARD_URL). When invoked via systemd's
# arem-host-heartbeat.service, EnvironmentFile=/etc/arem-worker.env
# already injects these. The sources below cover manual invocation;
# prefer the persistent /etc/ location over the legacy tmpfs /tmp/ one.
if [ -f /etc/arem-worker.env ]; then
  set -a; source /etc/arem-worker.env; set +a
elif [ -f /tmp/vercel-prod.env ]; then
  set -a; source /tmp/vercel-prod.env; set +a
fi

DASHBOARD_URL="${DASHBOARD_URL:-https://arem-editing-dashboard.vercel.app}"
WORKER_ID="host-$(hostname -s)"

# Collect host metrics. All best-effort — failures fall back to empty.
UPTIME_SEC=$(awk '{print int($1)}' /proc/uptime)
LOAD_1=$(awk '{print $1}' /proc/loadavg)
MEM_FREE_MB=$(free -m | awk '/^Mem:/ {print $7}')   # available
MEM_TOTAL_MB=$(free -m | awk '/^Mem:/ {print $2}')
DISK_FREE_GB=$(df -BG /tmp | awk 'NR==2 {gsub("G",""); print $4}')
DISK_FREE_HOME_GB=$(df -BG /home | awk 'NR==2 {gsub("G",""); print $4}')

# GPU metrics via nvidia-smi (skip cleanly if not installed/working)
GPU_TEMP=""; GPU_MEM_USED_MB=""; GPU_MEM_TOTAL_MB=""; GPU_UTIL=""
if command -v nvidia-smi >/dev/null 2>&1; then
  read GPU_TEMP GPU_MEM_USED_MB GPU_MEM_TOTAL_MB GPU_UTIL < <(nvidia-smi \
    --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' ' ' || echo)
fi

# Defensive: when the nvidia kernel module is loaded but the driver fails
# to talk to the GPU (e.g. post-reboot module-version mismatch), nvidia-smi
# emits an English error sentence to stdout instead of numbers. Without
# this guard those words get interpolated unquoted into the JSON below,
# the dashboard's req.json() returns null, and the heartbeat silently
# fails to land in the DB. Numeric-validate each value or zero it out.
[[ "$GPU_TEMP"          =~ ^[0-9]+$ ]] || GPU_TEMP=""
[[ "$GPU_MEM_USED_MB"   =~ ^[0-9]+$ ]] || GPU_MEM_USED_MB=""
[[ "$GPU_MEM_TOTAL_MB"  =~ ^[0-9]+$ ]] || GPU_MEM_TOTAL_MB=""
[[ "$GPU_UTIL"          =~ ^[0-9]+$ ]] || GPU_UTIL=""

METADATA=$(cat <<EOF
{
  "kind": "host",
  "hostname": "$(hostname)",
  "uptime_sec": ${UPTIME_SEC:-0},
  "load_1min": ${LOAD_1:-0},
  "mem_free_mb": ${MEM_FREE_MB:-0},
  "mem_total_mb": ${MEM_TOTAL_MB:-0},
  "disk_free_tmp_gb": ${DISK_FREE_GB:-0},
  "disk_free_home_gb": ${DISK_FREE_HOME_GB:-0},
  "gpu_temp_c": ${GPU_TEMP:-null},
  "gpu_mem_used_mb": ${GPU_MEM_USED_MB:-null},
  "gpu_mem_total_mb": ${GPU_MEM_TOTAL_MB:-null},
  "gpu_util_pct": ${GPU_UTIL:-null}
}
EOF
)

# Snippet inserted into host_heartbeat.sh after METADATA generation.
# Builds a JSON array describing each long-running systemd unit so the
# dashboard can render one card per service.

SERVICES_LIST=(
  "system:arem-sky-swap.service"
  "system:arem-enhancement-local.service"
  "system:arem-inpaint-sandbox.service"
  "system:arem-pretagger.service"
  "system:arem-worker-local.service"
  "user:arem-video-render.service"
)

# Short slug used in workerId (must NOT contain dots / spaces; matches
# the dashboard's expected service name)
service_slug() {
  case "$1" in
    arem-sky-swap.service)        echo "sky-swap" ;;
    arem-enhancement-local.service) echo "enhancement" ;;
    arem-inpaint-sandbox.service) echo "inpaint-sandbox" ;;
    arem-pretagger.service)       echo "pretagger" ;;
    arem-worker-local.service)    echo "worker-local" ;;
    arem-video-render.service)    echo "video-render" ;;
    *) echo "$1" | sed 's/\.service$//' ;;
  esac
}

systemctl_show_safe() {
  # Args: $1=scope (system|user), $2=unit. Echoes a tab-separated tuple:
  #   active_state  sub_state  active_enter_ts  n_restarts  uptime_sec
  local scope="$1" unit="$2"
  local args=()
  if [ "$scope" = "user" ]; then args+=("--user"); export XDG_RUNTIME_DIR="/run/user/$(id -u)"; fi
  local out
  out=$(systemctl "${args[@]}" show "$unit" \
    -p ActiveState,SubState,ActiveEnterTimestamp,NRestarts 2>/dev/null || echo "")
  local active sub ts nr
  active=$(echo "$out" | sed -n 's/^ActiveState=//p')
  sub=$(echo "$out" | sed -n 's/^SubState=//p')
  ts=$(echo "$out" | sed -n 's/^ActiveEnterTimestamp=//p')
  nr=$(echo "$out" | sed -n 's/^NRestarts=//p')
  local uptime=""
  if [ -n "$ts" ] && [ "$ts" != "0" ] && [ "$ts" != "n/a" ]; then
    local epoch
    epoch=$(date -d "$ts" +%s 2>/dev/null || echo "")
    if [ -n "$epoch" ]; then
      uptime=$(( $(date +%s) - epoch ))
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s' "${active:-unknown}" "${sub:-unknown}" "${ts:-}" "${nr:-0}" "${uptime:-}"
}

build_services_json() {
  local first=1
  local out="["
  for entry in "${SERVICES_LIST[@]}"; do
    local scope="${entry%%:*}"
    local unit="${entry#*:}"
    local slug
    slug=$(service_slug "$unit")
    IFS=$'\t' read -r active sub ts nr uptime < <(systemctl_show_safe "$scope" "$unit")
    if [ $first -eq 0 ]; then out+=","; fi
    first=0
    local uptime_json="null"
    [ -n "$uptime" ] && [[ "$uptime" =~ ^[0-9]+$ ]] && uptime_json="$uptime"
    local nr_json="0"
    [[ "$nr" =~ ^[0-9]+$ ]] && nr_json="$nr"
    out+="{\"name\":\"${slug}\",\"status\":\"${active}\",\"subState\":\"${sub}\",\"uptimeSec\":${uptime_json},\"nRestarts\":${nr_json},\"activeEnterTs\":\"${ts}\"}"
  done
  out+="]"
  printf '%s' "$out"
}

# Send heartbeat + receive any operator-triggered action to perform.
# Possible actions:
#   restart_worker — restart the arem-worker-local systemd unit
#   reboot_host    — reboot the box
# Both require passwordless sudo for jordan via /etc/sudoers.d/arem-worker.
RESPONSE=$(curl -s -X POST "${DASHBOARD_URL}/api/internal/heartbeat" \
  -H "Content-Type: application/json" \
  -H "x-worker-token: ${WORKER_TOKEN:-}" \
  -d "{\"workerId\":\"${WORKER_ID}\",\"metadata\":${METADATA},\"services\":$(build_services_json)}" \
  --max-time 10 || true)

# Parse pendingAction without jq (pure bash + python fallback).
PENDING=$(printf '%s' "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    p = d.get('pendingAction')
    print(p if p else '')
except Exception:
    print('')
" 2>/dev/null || echo "")

# Also handle pending actions on sibling workers on this host. The
# dashboard returns these in `siblingActions` whenever a host-*
# workerId calls heartbeat. Without this block, restart_worker (which
# the action API routes to the local-3090-* row, not the host row)
# would sit pending forever — the host has sudo to restart the worker
# unit but only sees its own pendingAction otherwise.
SIBLING_ACTIONS=$(printf '%s' "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    for s in (d.get('siblingActions') or []):
        wid = s.get('workerId',''); action = s.get('action','')
        if wid and action:
            print(wid + '\t' + action)
except Exception:
    pass
" 2>/dev/null || echo "")

if [ -n "$SIBLING_ACTIONS" ]; then
  EXEC_JSON="["
  FIRST=1
  while IFS=$'\t' read -r SIB_ID SIB_ACTION; do
    [ -z "$SIB_ID" ] && continue
    echo "[heartbeat] sibling action: $SIB_ID -> $SIB_ACTION"
    case "$SIB_ACTION" in
      restart_worker)
        sudo /bin/systemctl restart arem-worker-local.service
        ;;
      restart_service)
        # SIB_ID looks like service-{slug}-{hostsuffix}. Extract slug.
        SVC_SLUG=$(echo "$SIB_ID" | sed -E "s/^service-(.*)-$(hostname -s)$/\1/")
        case "$SVC_SLUG" in
          sky-swap)        sudo /bin/systemctl restart arem-sky-swap.service ;;
          enhancement)     sudo /bin/systemctl restart arem-enhancement-local.service ;;
          inpaint-sandbox) sudo /bin/systemctl restart arem-inpaint-sandbox.service ;;
          pretagger)       sudo /bin/systemctl restart arem-pretagger.service ;;
          worker-local)    sudo /bin/systemctl restart arem-worker-local.service ;;
          video-render)    XDG_RUNTIME_DIR="/run/user/$(id -u)" /bin/systemctl --user restart arem-video-render.service ;;
          *) echo "[heartbeat] unknown service slug: $SVC_SLUG (skipping)"; continue ;;
        esac
        ;;
      *)
        echo "[heartbeat] unknown sibling action: $SIB_ACTION (skipping)"
        continue
        ;;
    esac
    if [ $FIRST -eq 0 ]; then EXEC_JSON+=","; fi
    EXEC_JSON+="{\"workerId\":\"${SIB_ID}\",\"action\":\"${SIB_ACTION}\"}"
    FIRST=0
  done <<< "$SIBLING_ACTIONS"
  EXEC_JSON+="]"
  if [ $FIRST -eq 0 ]; then
    curl -s -X POST "${DASHBOARD_URL}/api/internal/heartbeat" \
      -H "Content-Type: application/json" \
      -H "x-worker-token: ${WORKER_TOKEN:-}" \
      -d "{\"workerId\":\"${WORKER_ID}\",\"metadata\":${METADATA},\"siblingExecutedActions\":${EXEC_JSON}}" \
      --max-time 10 >/dev/null || true
  fi
fi

if [ -z "$PENDING" ]; then
  exit 0
fi

echo "[heartbeat] pending action: $PENDING"

case "$PENDING" in
  report_logs)
    # Tail the worker's journal and post back in a single combined call.
    LOGS=$(journalctl -u arem-worker-local --no-pager -n 100 2>/dev/null | tail -c 40000)
    LOGS_JSON=$(printf '%s' "$LOGS" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
    curl -s -X POST "${DASHBOARD_URL}/api/internal/heartbeat" \
      -H "Content-Type: application/json" \
      -H "x-worker-token: ${WORKER_TOKEN:-}" \
      -d "{\"workerId\":\"${WORKER_ID}\",\"metadata\":${METADATA},\"executedAction\":\"report_logs\",\"logs\":${LOGS_JSON}}" \
      --max-time 15 >/dev/null || true
    exit 0
    ;;
  restart_worker)
    sudo /bin/systemctl restart arem-worker-local.service
    EXECUTED="restart_worker"
    ;;
  restart_sky_swap)
    sudo /bin/systemctl restart arem-sky-swap.service
    EXECUTED="restart_sky_swap"
    ;;
  restart_enhancement)
    sudo /bin/systemctl restart arem-enhancement-local.service
    EXECUTED="restart_enhancement"
    ;;
  restart_inpaint_sandbox)
    sudo /bin/systemctl restart arem-inpaint-sandbox.service
    EXECUTED="restart_inpaint_sandbox"
    ;;
  restart_pretagger)
    sudo /bin/systemctl restart arem-pretagger.service
    EXECUTED="restart_pretagger"
    ;;
  restart_video_render)
    XDG_RUNTIME_DIR="/run/user/$(id -u)" /bin/systemctl --user restart arem-video-render.service
    EXECUTED="restart_video_render"
    ;;
  reboot_host)
    # Report the executed action first (we won't be alive after reboot
    # to do it later), then trigger the reboot in the background.
    curl -s -X POST "${DASHBOARD_URL}/api/internal/heartbeat" \
      -H "Content-Type: application/json" \
      -H "x-worker-token: ${WORKER_TOKEN:-}" \
      -d "{\"workerId\":\"${WORKER_ID}\",\"metadata\":${METADATA},\"executedAction\":\"reboot_host\"}" \
      --max-time 10 >/dev/null || true
    # Give the heartbeat a moment to land before triggering reboot
    sleep 2
    sudo /sbin/reboot
    exit 0
    ;;
  *)
    echo "[heartbeat] unknown action ignored: $PENDING"
    exit 0
    ;;
esac

# Confirm executed action so dashboard can clear the pending flag.
curl -s -X POST "${DASHBOARD_URL}/api/internal/heartbeat" \
  -H "Content-Type: application/json" \
  -H "x-worker-token: ${WORKER_TOKEN:-}" \
  -d "{\"workerId\":\"${WORKER_ID}\",\"metadata\":${METADATA},\"executedAction\":\"${EXECUTED}\"}" \
  --max-time 10 >/dev/null || true
