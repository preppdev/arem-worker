"""RunPod serverless handler.

Each invocation claims one queued job from the dashboard, processes it
through the editing pipeline, uploads to Dropbox, and reports status
back to the dashboard.

Why this shape: we already have a working claim-based worker (worker.py).
Flipping to push-based dispatch requires schema changes on both sides.
A handler that does one claim+process per invocation:
- Plays nicely with RunPod's autoscaling (more queued jobs → more workers)
- Reuses the dashboard's atomic claim (FOR UPDATE SKIP LOCKED) — no race
- Lets us run local workers AND serverless workers against the same
  dashboard with no behavioral difference
"""
from __future__ import annotations

import os
import sys
import traceback

import runpod

# Reuse the local-mode worker logic. handler.py and worker.py live in
# the same package inside the container.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker import loop_once, log, WORKER_TOKEN, DASHBOARD_URL  # noqa: E402


def handler(event):
    """Process up to N queued jobs per invocation.

    Default: 1 job per invocation. RunPod scales workers up when more
    requests arrive, so 1 job per worker maximizes parallelism.

    Optionally drain the queue serially with `{"input": {"max_jobs": 5}}`
    or `{"input": {"drain": true}}` (the latter loops until empty).
    """
    inp = (event or {}).get("input") or {}
    max_jobs = int(inp.get("max_jobs", 1))
    drain = bool(inp.get("drain", False))

    log(f"handler invoked dashboard={DASHBOARD_URL} max_jobs={max_jobs} drain={drain}")
    if not WORKER_TOKEN:
        return {"error": "WORKER_TOKEN env var missing inside the container"}

    n_processed = 0
    n_empty = 0
    while True:
        try:
            processed = loop_once()
        except Exception as e:
            log(f"handler error: {e}\n{traceback.format_exc()}")
            return {"processed": n_processed, "error": str(e)}

        if processed:
            n_processed += 1
            n_empty = 0
            if not drain and n_processed >= max_jobs:
                break
        else:
            n_empty += 1
            if not drain or n_empty >= 2:
                break

    return {"processed": n_processed}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
