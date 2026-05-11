#!/bin/bash
# Host-level heartbeat. Pings the dashboard once per minute with a JSON
# blob of host health metrics. Independent of the worker process — if
# this script can run, the box is up, network is reachable, and Vercel
# is responding.
#
# Installed as a systemd timer: arem-host-heartbeat.timer fires once a
# minute and runs arem-host-heartbeat.service which executes this script.

set -euo pipefail

# Load creds (WORKER_TOKEN, DASHBOARD_URL)
if [ -f /tmp/vercel-prod.env ]; then
  set -a
  source /tmp/vercel-prod.env
  set +a
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

curl -sf -X POST "${DASHBOARD_URL}/api/internal/heartbeat" \
  -H "Content-Type: application/json" \
  -H "x-worker-token: ${WORKER_TOKEN:-}" \
  -d "{\"workerId\":\"${WORKER_ID}\",\"metadata\":${METADATA}}" \
  --max-time 10 >/dev/null
