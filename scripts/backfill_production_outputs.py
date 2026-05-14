"""Backfill historical production output JPGs into arem-production-edit-jobs
+ Cloudflare Images.

For every completed Job in the dashboard's candidate list, this script:
  1. Pulls each *.jpg from the Dropbox 08-Test-Edit output folder.
  2. Re-uploads it to r2:<R2_OUTPUT_BUCKET>/tours/<jobId>/photos/<NNNN>-<stem>.jpg
     (R2-IA archive, multi-year retention).
  3. Uploads it to Cloudflare Images, capturing the returned cfImageId.
  4. POSTs (productionR2Path, sortOrder, cfImageId) per-image to
     /api/internal/image-classification.

sortOrder = 1-based ordinal in alphabetical midStem order within the job.
Stage-2 _stage2-{int|ext} suffix is stripped from the target filename so
keys read cleanly in the gallery.

No GPU needed — rclone + HTTP. Run on any box with rclone configured for
dropbox + r2 remotes.

Usage:
    python -m scripts.backfill_production_outputs --limit 50 --since 2025-01-01
    python -m scripts.backfill_production_outputs --job <jobId>          # single
    python -m scripts.backfill_production_outputs --dry-run --limit 5

Env:
    WORKER_TOKEN              dashboard auth (required)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs
    CLOUDFLARE_API_TOKEN      CF Images upload auth (required for CF step)
    CLOUDFLARE_ACCOUNT_ID     CF account for the Images API endpoint
    RCLONE_DROPBOX            default 'dropbox'
    RCLONE_R2                 default 'r2'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import urllib.error

import requests  # type: ignore  # for CF Images multipart upload

# Make repo root importable so we can `from pipeline import cc_ingest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import cc_ingest  # type: ignore  # noqa: E402

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")
RCLONE_DROPBOX = os.environ.get("RCLONE_DROPBOX", "dropbox")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")


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


def cf_images_upload(local_path: str, *, job_id: str, mid_stem: str,
                     sort_order: int) -> str | None:
    """POST a JPG to Cloudflare Images. Returns the cfImageId on success,
    None on any failure (logged). The R2 archive is already durable by
    the time we get here, so CF Images is recoverable via a later
    reconciliation pass."""
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        return None
    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{CLOUDFLARE_ACCOUNT_ID}/images/v1")
    meta = json.dumps({
        "src": "backfill_production_outputs",
        "jobId": job_id,
        "midStem": mid_stem,
        "sortOrder": sort_order,
    })
    try:
        with open(local_path, "rb") as f:
            files = {
                "file": (os.path.basename(local_path), f, "image/jpeg"),
                "metadata": (None, meta, "application/json"),
            }
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
                files=files,
                timeout=60,
            )
    except Exception as e:
        log(f"    WARN CF Images upload {mid_stem}: {str(e)[:200]}")
        return None
    if resp.status_code != 200:
        log(f"    WARN CF Images upload {mid_stem}: HTTP {resp.status_code} "
            f"{resp.text[:200]}")
        return None
    try:
        return resp.json()["result"]["id"]
    except Exception as e:
        log(f"    WARN CF Images parse {mid_stem}: {str(e)[:200]}")
        return None


def process_job(job: dict, *, dry_run: bool) -> dict:
    job_id = job["id"]
    out = job.get("outputLocation") or {}
    if out.get("kind") != "dropbox":
        log(f"  [{job_id}] non-dropbox output ({out.get('kind')}) — skip")
        return {"jobId": job_id, "skipped": "non-dropbox"}

    # outputLocation.path from the dashboard already includes the
    # 08-Test-Edit suffix — don't append it again.
    dropbox_src = resolve_dropbox(out["path"])

    # List the Dropbox source. Skip jobs whose 08-Test-Edit was deleted
    # or never existed.
    ls = rclone(["lsf", dropbox_src, "--include", "*.jpg"], timeout=60)
    if ls.returncode != 0:
        log(f"  [{job_id}] dropbox lsf failed rc={ls.returncode}: "
            f"{ls.stderr[-200:].strip()}")
        return {"jobId": job_id, "skipped": "lsf-failed"}
    files = sorted(
        f.strip() for f in ls.stdout.splitlines() if f.strip().endswith(".jpg")
    )
    # Sort by stripped-midStem so sortOrder is deterministic across re-runs.
    files = sorted(files, key=midstem_from_filename)
    if not files:
        log(f"  [{job_id}] no JPGs at {dropbox_src} — skip")
        return {"jobId": job_id, "skipped": "empty"}

    log(f"  [{job_id}] {len(files)} JPGs")
    if dry_run:
        for idx, f in enumerate(files[:3], 1):
            stem = midstem_from_filename(f)
            log(f"    [dry]   {f} -> tours/{job_id}/photos/{idx:04d}-{stem}.jpg")
        if len(files) > 3:
            log(f"    [dry]   ... + {len(files) - 3} more")
        return {"jobId": job_id, "uploaded": 0, "posted": 0, "files": len(files),
                "dry_run": True}

    # Stream-copy each file: Dropbox -> /tmp/<basename> -> r2 (renamed) +
    # CF Images. Per-file (not batch) because the target name encodes
    # sortOrder, and CF Images needs a local file handle.
    n_uploaded = 0
    n_cf = 0
    n_posted = 0
    cc_assets: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="arem-prod-backfill-") as scratch:
        for idx, src_name in enumerate(files, start=1):
            mid_stem = midstem_from_filename(src_name)
            local = os.path.join(scratch, src_name)
            # Pull from Dropbox to scratch
            cp_in = rclone(["copyto", f"{dropbox_src}/{src_name}", local,
                            "--progress=false"], timeout=180)
            if cp_in.returncode != 0:
                log(f"    WARN dropbox→scratch {src_name}: rc={cp_in.returncode} "
                    f"{cp_in.stderr[-160:].strip()}")
                continue
            # Push to R2 under the canonical key shape
            r2_key = f"tours/{job_id}/photos/{idx:04d}-{mid_stem}.jpg"
            cp_out = rclone(["copyto", local,
                             f"{RCLONE_R2}:{R2_OUTPUT_BUCKET}/{r2_key}",
                             "--progress=false"], timeout=180)
            if cp_out.returncode != 0:
                log(f"    WARN scratch→r2 {mid_stem}: rc={cp_out.returncode} "
                    f"{cp_out.stderr[-160:].strip()}")
                continue
            n_uploaded += 1
            # CF Images
            cf_id = cf_images_upload(local, job_id=job_id, mid_stem=mid_stem,
                                     sort_order=idx)
            if cf_id:
                n_cf += 1
            # Dashboard POST
            try:
                post_production({
                    "jobId": job_id,
                    "midStem": mid_stem,
                    "productionR2Path": r2_key,
                    "sortOrder": idx,
                    "cfImageId": cf_id,
                })
                n_posted += 1
            except Exception as e:
                log(f"    WARN POST {mid_stem}: {str(e)[:200]}")
            # Build CC asset record while local file is still on disk —
            # we need width/height/sizeBytes from it. Only include
            # assets that have both r2Key and cfImageId (CC's required
            # fields for kind=photo); the rest get a reconciliation pass.
            if cf_id:
                width, height = cc_ingest.jpeg_dims(local)
                try:
                    size_bytes = os.path.getsize(local)
                except OSError:
                    size_bytes = None
                cc_assets.append({
                    "kind": "photo",
                    "sortOrder": idx,
                    "r2Bucket": R2_OUTPUT_BUCKET,
                    "r2Key": r2_key,
                    "cfImageId": cf_id,
                    "mimeType": "image/jpeg",
                    "width": width,
                    "height": height,
                    "sizeBytes": size_bytes,
                    "room": None,            # populated by a later pass
                    "isHero": False,
                    "altText": None,
                    "caption": None,
                    "checksum": None,
                })
            os.remove(local)

    # POST the batch to CC. CC's Shoot ↔ our Job.id are separate id
    # spaces — always POST to /new with an ensureJob block; CC dedups
    # via dropboxPath, then address+state+date, else creates a stub.
    cc_accepted = cc_skipped = cc_errors = 0
    if cc_assets:
        ensure_job = cc_ingest.build_ensure_job(
            editor_job_id=job_id,
            dropbox_path=job.get("dropboxPath"),
            photographer=job.get("photographer"),
            completed_at=job.get("completedAt"),
        )
        result = cc_ingest.post_media(
            assets=cc_assets, shoot_key="new", ensure_job=ensure_job,
        )
        if "error" in result:
            log(f"  [{job_id}] CC ingest: {result['error']}")
        else:
            cc_accepted = len(result.get("accepted") or [])
            cc_skipped = len(result.get("skipped") or [])
            cc_errors = len(result.get("errors") or [])
            log(f"  [{job_id}] CC ingest: shootId={result.get('shootId')} "
                f"resolvedFrom={result.get('resolvedFrom')} "
                f"acc={cc_accepted} skip={cc_skipped} err={cc_errors}")

    log(f"  [{job_id}] uploaded={n_uploaded}  cf_images={n_cf}  posted={n_posted}  "
        f"cc={cc_accepted}/{cc_skipped}/{cc_errors} (acc/skip/err)")
    return {"jobId": job_id, "uploaded": n_uploaded, "cf_images": n_cf,
            "posted": n_posted, "files": len(files),
            "cc_accepted": cc_accepted, "cc_skipped": cc_skipped,
            "cc_errors": cc_errors}


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
        f"bucket={R2_OUTPUT_BUCKET}")

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
    total_cf = sum(r.get("cf_images", 0) for r in results)
    total_posted = sum(r.get("posted", 0) for r in results)
    total_cc_acc = sum(r.get("cc_accepted", 0) for r in results)
    total_cc_skip = sum(r.get("cc_skipped", 0) for r in results)
    total_cc_err = sum(r.get("cc_errors", 0) for r in results)
    log(f"\n--- production-output backfill done in {dt:.1f}s ({dt/max(len(jobs),1):.1f}s/job avg) ---")
    log(f"  jobs: {len(jobs)}  r2: {total_uploaded}  cf_images: {total_cf}  posted: {total_posted}")
    log(f"  cc_ingest: accepted={total_cc_acc}  skipped={total_cc_skip}  errors={total_cc_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
