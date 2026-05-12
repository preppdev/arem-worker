#!/usr/bin/env python3
"""
Local fleet/worker status TUI for the 3090 box.

Designed to be driven by `watch -n 2 -t -c` from a full-screen gnome-terminal
launched on login (see arem-fleet-tui.desktop). Prints one ANSI-colored
snapshot per invocation; `watch` handles the refresh + screen-clear.

Data sources, in priority order:
  1. Dashboard /api/workers — via x-api-token (EXTERNAL_API_TOKEN env var,
     same credential the production heartbeat uses). Gives the same
     current-job / throughput / heartbeat info that /fleet shows.
  2. Local fallbacks — systemctl is-active, nvidia-smi, journalctl,
     /proc/uptime. Used when the dashboard is unreachable so the display
     still shows something useful.

Usage:
  watch -n 2 -t -c python3 ~/arem-worker/arem-fleet-tui.py
  (one-shot for debugging:)  python3 ~/arem-worker/arem-fleet-tui.py
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
)


# ── env loading (same shape as host_heartbeat.sh / run_local.sh) ────────────

def load_env_file(path: str) -> dict[str, str]:
    if not os.path.isfile(path):
        return {}
    out: dict[str, str] = {}
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


# Prefer the persistent /etc/ location, fall back to legacy /tmp.
_env = load_env_file("/etc/arem-worker.env") or load_env_file("/tmp/vercel-prod.env")
EXTERNAL_API_TOKEN = (
    os.environ.get("EXTERNAL_API_TOKEN") or _env.get("EXTERNAL_API_TOKEN") or ""
)


# ── ANSI escapes ────────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[97m"


# ── data probes ─────────────────────────────────────────────────────────────

def fetch_workers() -> dict | None:
    """GET /api/workers with EXTERNAL_API_TOKEN; returns parsed JSON or None."""
    if not EXTERNAL_API_TOKEN:
        return None
    try:
        req = urllib.request.Request(
            f"{DASHBOARD_URL}/api/workers",
            headers={"x-api-token": EXTERNAL_API_TOKEN},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def gpu_info() -> dict | None:
    """Parse one row of nvidia-smi --query-gpu CSV."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split("\n")[0].split(",")]
        def maybe_int(s: str) -> int | None:
            return int(s) if s.isdigit() else None
        return {
            "name": parts[0],
            "temp": maybe_int(parts[1]),
            "util": maybe_int(parts[2]),
            "mem_used": maybe_int(parts[3]),
            "mem_total": maybe_int(parts[4]),
        }
    except Exception:
        return None


def systemd_active(unit: str) -> str:
    try:
        return subprocess.check_output(
            ["systemctl", "is-active", unit],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
    except Exception:
        return "unknown"


def journalctl_tail(unit: str, n: int = 8) -> list[str]:
    try:
        return subprocess.check_output(
            ["journalctl", "-u", unit, "--no-pager", "-n", str(n)],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().splitlines()
    except Exception:
        return []


def host_uptime_sec() -> int:
    try:
        with open("/proc/uptime") as f:
            return int(float(f.read().split()[0]))
    except Exception:
        return 0


# ── format helpers ──────────────────────────────────────────────────────────

def fmt_age(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def fmt_uptime(sec: int) -> str:
    if sec < 3600:
        return f"{sec // 60}m"
    if sec < 86400:
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"
    return f"{sec // 86400}d{(sec % 86400) // 3600:02d}h"


def term_width() -> int:
    try:
        return max(70, shutil.get_terminal_size().columns - 2)
    except Exception:
        return 100


# ── render ──────────────────────────────────────────────────────────────────

def render() -> None:
    hostname = socket.gethostname()
    worker_id_local = f"local-3090-{hostname.split('.')[0]}"
    width = term_width()
    rule = C.DIM + ("─" * width) + C.RESET
    now_str = datetime.now().strftime("%a %d %b %H:%M:%S")

    # Header
    pad = " " * max(1, width - len(hostname) - len(now_str))
    print(f"{C.BOLD}{C.WHITE}{hostname}{C.RESET}{pad}{C.DIM}{now_str}{C.RESET}")
    print(rule)

    # Worker service + dashboard reach
    svc = systemd_active("arem-worker-local")
    svc_color = C.GREEN if svc == "active" else C.RED
    uptime = host_uptime_sec()
    print(
        f"  {svc_color}●{C.RESET} {C.BOLD}WORKER:{C.RESET} {svc:<8}  "
        f"{C.DIM}(host uptime {fmt_uptime(uptime)}){C.RESET}"
    )

    workers_data = fetch_workers()
    dash_color = C.GREEN if workers_data else C.YELLOW
    dash_text = "reachable" if workers_data else "unreachable"
    print(f"  {dash_color}●{C.RESET} {C.BOLD}DASHBOARD:{C.RESET} {dash_text}")

    # GPU
    g = gpu_info()
    if g:
        temp = g["temp"]
        if temp is None:
            temp_color = C.YELLOW
            temp_str = "—"
        else:
            if temp > 80:
                temp_color = C.RED
            elif temp > 70:
                temp_color = C.YELLOW
            else:
                temp_color = C.GREEN
            temp_str = f"{temp}°C"
        util = f"{g['util']}%" if g["util"] is not None else "—"
        if g["mem_used"] is not None and g["mem_total"]:
            mem = f"{g['mem_used'] / 1024:.1f} / {g['mem_total'] / 1024:.0f} GB"
        else:
            mem = "—"
        print(
            f"  {temp_color}●{C.RESET} {C.BOLD}GPU:{C.RESET} {g['name']} · "
            f"{temp_color}{temp_str}{C.RESET} · {util} util · {mem}"
        )
    else:
        print(f"  {C.RED}●{C.RESET} {C.BOLD}GPU:{C.RESET} nvidia-smi failed")

    print()

    # Current job, throughput — from dashboard (no local equivalent)
    if workers_data:
        target = next(
            (
                w
                for w in workers_data.get("workers", [])
                if w.get("workerId") == worker_id_local
            ),
            None,
        )
        cj = target.get("currentJob") if target else None
        print(f"{C.BOLD}{C.CYAN}NOW{C.RESET}")
        if cj:
            start = cj.get("processingStartedAt")
            elapsed = "—"
            if start:
                try:
                    t = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    secs = int((datetime.now(timezone.utc) - t).total_seconds())
                    elapsed = fmt_age(max(0, secs))
                except Exception:
                    pass
            task = cj.get("taskNumber", "?")
            name = cj.get("jobFolderName") or (cj.get("id", "") or "")[:14]
            status = cj.get("status", "—")
            print(
                f"  {C.BOLD}{C.WHITE}T-{task}{C.RESET}  "
                f"⏱ {C.BOLD}{elapsed}{C.RESET}  {C.CYAN}{status}{C.RESET}"
            )
            print(f"  {C.DIM}{name}{C.RESET}")
        else:
            print(f"  {C.DIM}idle{C.RESET}")
        print()

        if target:
            hr = target.get("doneLastHour", 0)
            day = target.get("doneLastDay", 0)
            print(
                f"{C.BOLD}{C.CYAN}THROUGHPUT{C.RESET}  "
                f"{C.BOLD}{C.GREEN}{hr}{C.RESET}/hr  "
                f"{C.BOLD}{C.WHITE}{day}{C.RESET}/day"
            )
            print()
    else:
        print(f"{C.DIM}(dashboard unreachable — showing local-only data){C.RESET}")
        print()

    # Journal — always local
    print(f"{C.BOLD}{C.CYAN}WORKER JOURNAL{C.RESET}  {C.DIM}last 8 lines{C.RESET}")
    print(rule)
    lines = journalctl_tail("arem-worker-local", 8)
    if lines:
        for ln in lines:
            if len(ln) > width - 2:
                ln = ln[: width - 3] + "…"
            print(f"  {C.DIM}{ln}{C.RESET}")
    else:
        print(f"  {C.DIM}(no journal output — service may be in early start-up){C.RESET}")


if __name__ == "__main__":
    try:
        render()
    except KeyboardInterrupt:
        sys.exit(0)
