#!/usr/bin/env python3
"""arem-video-render worker — SCAFFOLD / STUB.

Polls the R2 queue `video-render/requests/` for render requests the dashboard
drops (one JSON per job), and — for now — just logs that rendering is not
implemented yet, moving each handled request to `video-render/handled/` so it
isn't reprocessed. The real transcode/render pipeline will replace handle().
"""
import os, json, time, subprocess, sys

R2 = os.environ.get("RCLONE_R2", "r2")
BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
REQ = f"{R2}:{BUCKET}/video-render/requests"
DONE = f"{R2}:{BUCKET}/video-render/handled"
INTERVAL = int(os.environ.get("INTERVAL", "15"))


def rclone(*args):
    return subprocess.run(["rclone", *args], capture_output=True, text=True)


def list_requests():
    r = rclone("lsf", REQ + "/", "--include", "*.json")
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def handle(name):
    r = rclone("cat", f"{REQ}/{name}")
    try:
        req = json.loads(r.stdout)
    except Exception:
        req = {"raw": r.stdout[:200]}
    print(f"[video-render] request {name}: job={req.get('jobId','?')} — "
          f"render pipeline NOT IMPLEMENTED yet; acknowledging.", flush=True)
    rclone("moveto", f"{REQ}/{name}", f"{DONE}/{name}")  # don't loop on it


def main():
    once = "--once" in sys.argv
    print(f"[video-render] stub worker up; polling {REQ} every {INTERVAL}s "
          f"(render not implemented)", flush=True)
    while True:
        try:
            for name in list_requests():
                handle(name)
        except Exception as e:
            print(f"[video-render] poll error: {e}", flush=True)
        if once:
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
