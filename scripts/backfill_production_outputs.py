"""Backfill historical production output JPGs into arem-production-edit-jobs.

For every completed Job in the dashboard's candidate list, copy the
Dropbox `08-Test-Edit/*.jpg` outputs to r2:<R2_OUTPUT_BUCKET>/jobs/<jobId>/
and POST each per-image productionR2Path back to
/api/internal/image-classification.

No GPU needed — pure rclone + HTTP. Run on any box with rclone configured
for both dropbox + r2 remotes.

Usage:
    python -m scripts.backfill_production_outputs --limit 50 --since 2025-01-01
    python -m scripts.backfill_production_outputs --job <jobId>          # single
    python -m scripts.backfill_production_outputs --dry-run --limit 5

Env:
    WORKER_TOKEN          dashboard auth (required)
    DASHBOARD_URL         default https://arem-editing-dashboard.vercel.app
    R2_OUTPUT_BUCKET      default arem-production-edit-jobs
    RCLONE_DROPBOX        default 'dropbox'
    RCLONE_R2             default 'r2'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")
RCLONE_DROPBOX = os.environ.get("RCLONE_DROPBOX", "dropbox")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")


def log(msg: str) -> None:
    print(msg, flush=True)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {body_text[:300]}") from e


def get_candidates(*, limit: int, offset: int, since: str | None,
                   job_id: str | None) -> dict:
    sp = {"limit": str(limit), "offset": str(offset)}
    if since:
        sp["since"] = since
    if job_id:
        sp["jobId"] = job_id
    return _request("GET",
                    f"/api/internal/backfill-jobs?{urllib.parse.urlencode(sp)}")


def post_production(payload: dict) -> dict:
    return _request("POST", "/api/internal/image-classification", payload)


def rclone(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args],
                          capture_output=True, text=True, timeout=timeout)


def resolve_dropbox(path: str) -> str:
    """Translate stored dropboxPath into an rclone remote path. Same
    logic as worker.resolve_dropbox_source — strips 'Jordan DiCaprio/'
    prefix if present and prepends RCLONE_DROPBOX:"""
    p = path.strip().rstrip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/"):]
    return f"{RCLONE_DROPBOX}:{p}"


def midstem_from_filename(name: str) -> str:
    """Strip _stage2-int / _stage2-ext suffixes from a worker-emitted
    JPG filename to recover the bare midStem that ImageReview uses."""
    stem = name.rsplit(".", 1)[0]
    for suffix in ("_stage2-int", "_stage2-ext"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def process_job(job: dict, *, dry_run: bool) -> dict:
    job_id = job["id"]
    out = job.get("outputLocation") or {}
    if out.get("kind") != "dropbox":
        # Manual-upload jobs already write to R2 (arem-editing-images);
        # they're not in scope for this backfill which targets the
        # Dropbox 08-Test-Edit -> arem-production-edit-jobs pipeline.
        log(f"  [{job_id}] non-dropbox output ({out.get('kind')}) — skip")
        return {"jobId": job_id, "skipped": "non-dropbox"}

    # outputLocation.path from the dashboard already includes the
    # 08-Test-Edit suffix — don't append it again.
    dropbox_src = resolve_dropbox(out["path"])
    r2_dst = f"{RCLONE_R2}:{R2_OUTPUT_BUCKET}/jobs/{job_id}"

    # List the Dropbox source first — empty source = nothing to do.
    ls = rclone(["lsf", dropbox_src, "--include", "*.jpg"], timeout=60)
    if ls.returncode != 0:
        log(f"  [{job_id}] dropbox lsf failed rc={ls.returncode}: "
            f"{ls.stderr[-200:].strip()}")
        return {"jobId": job_id, "skipped": "lsf-failed"}
    files = [f.strip() for f in ls.stdout.splitlines() if f.strip().endswith(".jpg")]
    if not files:
        log(f"  [{job_id}] no JPGs at {dropbox_src} — skip")
        return {"jobId": job_id, "skipped": "empty"}

    log(f"  [{job_id}] {len(files)} JPGs → {r2_dst}")
    if dry_run:
        for f in files[:3]:
            log(f"    [dry]   {f} -> jobs/{job_id}/{f}")
        if len(files) > 3:
            log(f"    [dry]   ... + {len(files) - 3} more")
        return {"jobId": job_id, "uploaded": 0, "posted": 0, "files": len(files),
                "dry_run": True}

    # Copy whole folder — rclone handles parallelism + idempotency.
    cp = rclone(["copy", dropbox_src, r2_dst,
                 "--include", "*.jpg",
                 "--transfers", "8", "--checkers", "8"],
                timeout=1800)
    if cp.returncode != 0:
        log(f"  [{job_id}] WARN copy rc={cp.returncode}: {cp.stderr[-200:].strip()}")
        return {"jobId": job_id, "uploaded": 0, "posted": 0, "files": len(files),
                "error": f"rclone rc={cp.returncode}"}

    # POST per-file productionR2Path. ImageReview keys are on midStem,
    # but a Job can emit BOTH <stem>_stage2-int.jpg AND <stem>_stage2-ext.jpg
    # for the same stem — last write wins, which is fine since each is
    # a complete final delivered JPG; the dashboard / virtual tour
    # platform doesn't care which variant is referenced.
    n_posted = 0
    for f in files:
        stem = midstem_from_filename(f)
        try:
            post_production({
                "jobId": job_id,
                "midStem": stem,
                "productionR2Path": f"jobs/{job_id}/{f}",
            })
            n_posted += 1
        except Exception as e:
            log(f"  [{job_id}] WARN POST {stem}: {str(e)[:200]}")

    log(f"  [{job_id}] uploaded {len(files)}  posted {n_posted}")
    return {"jobId": job_id, "uploaded": len(files), "posted": n_posted,
            "files": len(files)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--since", type=str, default=None,
                    help="ISO date; filter completedAt >= since")
    ap.add_argument("--job", type=str, default=None, help="single Job ID")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required")
        return 2
    if not R2_OUTPUT_BUCKET:
        log("ERROR: R2_OUTPUT_BUCKET env var is required")
        return 2

    log(f"production-output backfill: dashboard={DASHBOARD_URL} "
        f"bucket={R2_OUTPUT_BUCKET} dropbox_folder={DROPBOX_OUTPUT_FOLDER}")

    listing = get_candidates(limit=args.limit, offset=args.offset,
                             since=args.since, job_id=args.job)
    jobs = listing.get("jobs", [])
    log(f"  {len(jobs)} jobs to process (total candidates: {listing.get('total')})")
    if not jobs:
        return 0

    t0 = time.time()
    results: list[dict] = []
    for idx, job in enumerate(jobs, 1):
        log(f"\n=== [{idx}/{len(jobs)}] ===")
        try:
            results.append(process_job(job, dry_run=args.dry_run))
        except Exception as e:
            log(f"  ERROR job {job.get('id')}: {str(e)[:300]}")
            results.append({"jobId": job.get("id"), "error": str(e)[:200]})

    dt = time.time() - t0
    total_uploaded = sum(r.get("uploaded", 0) for r in results)
    total_posted = sum(r.get("posted", 0) for r in results)
    log(f"\n--- production-output backfill done in {dt:.1f}s ({dt/max(len(jobs),1):.1f}s/job avg) ---")
    log(f"  jobs: {len(jobs)}  uploaded: {total_uploaded}  posted: {total_posted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
