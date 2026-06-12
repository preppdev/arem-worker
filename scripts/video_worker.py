#!/usr/bin/env python3
"""Video-pipeline worker — SKELETON (2026-06-12).

AREM pipeline naming scheme (see AREM Editing docs/pipelines.md):
  editing | enhancement | video | drone

Polls /api/internal/video-jobs/claim on the dashboard (the contract mirrors
the enhancement pipeline exactly: claim -> progress deltas -> complete) and
will render the shoot's raw clips into finished walkthrough/social videos.

NOT IMPLEMENTED: every step is a logged no-op. The claim/progress/complete
plumbing is real, so pointing a systemd unit at this script exercises the
queue end-to-end; replace `run_step` with the renderer as it's built.

Usage / env: same shape as enhancement_worker.py (WORKER_TOKEN,
DASHBOARD_URL, WORKER_ID, IDLE_SLEEP).
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
WORKER_ID = os.environ.get("WORKER_ID", f"video-local-{socket.gethostname()}")
IDLE_SLEEP = int(os.environ.get("IDLE_SLEEP", "15"))


def log(msg: str) -> None:
    print(f"[{WORKER_ID}] {msg}", flush=True)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{DASHBOARD_URL}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} → HTTP {e.code}: {e.read().decode()[:300]}") from e


def run_step(step: dict, job: dict, parent: dict, options: dict | None) -> None:
    """TODO: the actual renderer. Per step key:
      render-walkthrough — discover raw clips (Dropbox 02-RAW-Video / R2),
        stabilize+grade, cut to template, music bed, brand cards, upload
        to R2 + Cloudflare Stream, report via complete.outputs.
      render-social — 9:16 cutdown of the walkthrough.
    """
    log(f"  STUB step {step['key']} for {parent.get('jobFolderName', job['parentJobId'])} "
        f"(options={options}) — no-op")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN required"); return 2

    log(f"polling {DASHBOARD_URL} every {IDLE_SLEEP}s (SKELETON — steps are no-ops)")
    while True:
        try:
            r = _request("POST", "/api/internal/video-jobs/claim", {"workerId": WORKER_ID})
        except Exception as e:
            log(f"claim error: {e}"); r = {"job": None}
        job = r.get("job")
        if job:
            jid = job["id"]
            try:
                for step in r.get("stepConfigs", []):
                    run_step(step, job, r.get("parent") or {}, r.get("options"))
                    _request("POST", f"/api/internal/video-jobs/{jid}/progress", {"dProcessed": 1})
                _request("POST", f"/api/internal/video-jobs/{jid}/complete",
                         {"ok": True, "outputs": []})
                log(f"job {jid} complete (stub)")
            except Exception as e:
                log(f"job {jid} FAILED: {e}")
                _request("POST", f"/api/internal/video-jobs/{jid}/complete",
                         {"ok": False, "errorMessage": str(e)[:500]})
            if args.once:
                return 0
        else:
            if args.once:
                log("queue empty (--once)"); return 0
            time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    sys.exit(main())
