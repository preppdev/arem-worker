"""Backfill Stage-1 NAFNet outputs for completed Jobs.

The fine-tune training loop (arem-photo-ai-V2/train_stage2.py) consumes
(Stage-1-output → vendor-JPG) pairs, but the live worker discards
Stage-1 intermediates with its scratch dir. This script re-runs Stage 1
only — lens-correct + demosaic + find_triplets + NAFNet forward pass —
for each candidate Job and uploads the resulting JPGs to R2 at
  training-stage1/<jobId>/<midStem>.jpg
then POSTs the paths back to the dashboard
(POST /api/internal/image-classification with stage1R2Path).

Stage 2, classifier, auto-upright, EXIF/branding are all skipped — Stage 1
is ~80% of the per-job time saved. Realistic throughput ~30–60 s per Job
including I/O.

Candidates default to Jobs completed on/after 2026-05-10 20:19 EDT (when
the may26 NAFNet w32 exterior ckpt deployed), so the resulting training
data aligns to the current production model.

Run from arem-worker repo root:
    python -m scripts.backfill_stage1 --limit 1 --dry-run
    python -m scripts.backfill_stage1 --limit 500          # full overnight run
    python -m scripts.backfill_stage1 --job <jobId>        # single
    python -m scripts.backfill_stage1 --since 2026-05-11   # custom cutoff

Depends on dashboard endpoints in arem-editing (the
/api/internal/backfill-stage1-jobs listing + the extended
/api/internal/image-classification accepting stage1R2Path).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import torch  # type: ignore
from PIL import Image  # type: ignore

# Reuse the live worker's Stage-1 pipeline. Importing run_pipeline pulls
# in lensfunpy / rawpy / torch — same deps the live worker uses.
from pipeline.run_pipeline import (  # type: ignore  # noqa: E402
    find_triplets,
    lens_correct_16,
    list_arws,
    LensExcluded,
    load_stage1,
    resize_max_mp,
    stage1_infer,
)


DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
RCLONE_DROPBOX = os.environ.get("RCLONE_DROPBOX", "dropbox")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/tmp/arem-stage1-backfill"))
CHECKPOINT_STAGE1 = Path(os.environ.get(
    "CHECKPOINT_STAGE1",
    str(Path.home() / "arem-worker" / "checkpoints" /
        "stage1_jxl_v1_best_lpips.pth"),
))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---- HTTP helpers ----


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
                   job_id: str | None, force: bool) -> dict:
    sp = {"limit": str(limit), "offset": str(offset)}
    if since:
        sp["since"] = since
    if job_id:
        sp["jobId"] = job_id
    if force:
        sp["force"] = "1"
    return _request("GET",
                    f"/api/internal/backfill-stage1-jobs?{urllib.parse.urlencode(sp)}")


def post_stage1(payload: dict) -> dict:
    return _request("POST", "/api/internal/image-classification", payload)


# ---- rclone helpers ----


def rclone(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args],
                          capture_output=True, text=True, timeout=timeout)


def resolve_dropbox(path: str) -> str:
    p = path.strip().rstrip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/"):]
    return f"{RCLONE_DROPBOX}:{p}"


def resolve_r2(key: str) -> str:
    return f"{RCLONE_R2}:{R2_BUCKET}/{key.lstrip('/')}"


# ARW + alternatives. Mirror what the live worker accepts so we don't
# silently miss anything.
RAW_EXTS = ["ARW", "arw", "NEF", "nef", "CR2", "cr2", "CR3", "cr3",
            "DNG", "dng", "RAF", "raf", "RW2", "rw2"]


def rclone_pull_raws(src: str, dst_dir: Path) -> int:
    """Pull RAW files only — skip JPG/sidecars/etc."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    include_flags: list[str] = []
    for ext in RAW_EXTS:
        include_flags += ["--include", f"*.{ext}"]
    proc = rclone(
        ["copy", src, str(dst_dir), *include_flags,
         "--transfers", "8", "--checkers", "8", "--ignore-existing"],
        timeout=1800,
    )
    if proc.returncode != 0:
        log(f"    rclone copy {src} rc={proc.returncode}: {proc.stderr[-160:].strip()}")
        return 0
    return sum(1 for p in dst_dir.iterdir() if p.is_file())


# ---- Stage-1 inference (lean wrapper around pipeline.run_pipeline) ----


def stage1_infer_dir(
    raw_dir: Path, out_dir: Path, model, device, lens_db,
) -> list[str]:
    """Run lens-correct + bracket grouping + Stage-1 NAFNet on every
    triplet in raw_dir. Returns the list of mid-stems for which Stage-1
    output landed in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raws = list_arws(raw_dir)
    triplets, group_failures = find_triplets(raws)
    if group_failures:
        log(f"    grouping: {len(group_failures)} bracket failures (continuing on valid triplets)")
    log(f"    {len(raws)} raws → {len(triplets)} triplets")

    # MAX_MP cap is the same constant the live worker uses (imported via
    # resize_max_mp). Reasonable inference resolution for the 3090.
    MAX_MP = 12

    successful: list[str] = []
    for ti, (under, mid, over) in enumerate(triplets, 1):
        stem = mid.stem
        try:
            u16, _ = lens_correct_16(under, lens_db)
            m16, _ = lens_correct_16(mid, lens_db)
            o16, _ = lens_correct_16(over, lens_db)
        except LensExcluded as e:
            log(f"    [{ti}/{len(triplets)}] {stem}: lens excluded — {e}")
            continue
        except Exception as e:
            log(f"    [{ti}/{len(triplets)}] {stem}: demosaic fail: {e}")
            continue

        u16 = resize_max_mp(u16, MAX_MP)
        m16 = resize_max_mp(m16, MAX_MP)
        o16 = resize_max_mp(o16, MAX_MP)

        try:
            s1_out = stage1_infer(model, u16, m16, o16, device)
        except Exception as e:
            log(f"    [{ti}/{len(triplets)}] {stem}: stage1 fail: {e}")
            continue

        # Save WITHOUT _stage1 suffix so the R2 key directly matches the
        # midStem convention used by other backfilled artifacts.
        Image.fromarray(s1_out).save(out_dir / f"{stem}.jpg", quality=95)
        successful.append(stem)

    return successful


# ---- Per-Job processing ----


def run_job(
    job: dict, *, model, device, lens_db, scratch_root: Path,
    dry_run: bool,
) -> dict:
    job_id: str = job["id"]
    task_no = job.get("taskNumber")
    src: dict = job["inputLocation"]
    already: set[str] = set(job.get("alreadyStage1MidStems") or [])

    log(f"  [T-{task_no}] {job_id}  src={src['kind']}:{src['path']}  "
        f"alreadyStage1={len(already)}")

    scratch = scratch_root / job_id
    scratch.mkdir(parents=True, exist_ok=True)
    raw_dir = scratch / "raws"
    stage1_dir = scratch / "stage1"

    # --- Pull RAWs ---
    src_remote = (
        resolve_dropbox(src["path"]) if src["kind"] == "dropbox"
        else resolve_r2(src["path"])
    )
    n_raws = rclone_pull_raws(src_remote, raw_dir)
    if n_raws == 0:
        log("    no RAW files at source — skip")
        return {"jobId": job_id, "raws": 0, "skipped": True, "scratch": scratch}

    # --- Run Stage 1 inference ---
    t0 = time.time()
    successful_stems = stage1_infer_dir(raw_dir, stage1_dir, model, device, lens_db)
    log(f"    stage1 inference {len(successful_stems)} stems in {time.time() - t0:.1f}s")

    # Skip already-done stems if we somehow re-processed them
    to_upload = [s for s in successful_stems if s not in already]
    if not to_upload:
        log("    no new stems to upload — skip")
        return {"jobId": job_id, "raws": n_raws, "uploaded": 0, "posted": 0,
                "scratch": scratch}

    # --- Upload to R2 ---
    n_uploaded = 0
    n_posted = 0
    if dry_run:
        log(f"    [dry] would upload {len(to_upload)} stage1 JPGs to "
            f"r2:{R2_BUCKET}/training-stage1/{job_id}/")
        for s in to_upload[:3]:
            log(f"    [dry]   {s}.jpg → training-stage1/{job_id}/{s}.jpg")
        if len(to_upload) > 3:
            log(f"    [dry]   ... + {len(to_upload) - 3} more")
    else:
        r = rclone(
            ["copy", str(stage1_dir),
             resolve_r2(f"training-stage1/{job_id}"),
             "--transfers", "8", "--checkers", "8"],
            timeout=1200,
        )
        if r.returncode != 0:
            log(f"    WARN stage1 upload rc={r.returncode}: {r.stderr[-160:].strip()}")
        else:
            n_uploaded = len(to_upload)

        # --- POST per-stem ---
        for stem in to_upload:
            try:
                post_stage1({
                    "jobId": job_id,
                    "midStem": stem,
                    "stage1R2Path": f"training-stage1/{job_id}/{stem}.jpg",
                })
                n_posted += 1
            except Exception as e:
                log(f"    WARN stage1 POST for {stem}: {str(e)[:200]}")

    log(f"  → raws={n_raws} stage1={len(successful_stems)} "
        f"uploaded={n_uploaded} posted={n_posted}"
        f"{'  [DRY]' if dry_run else ''}")
    return {
        "jobId": job_id,
        "raws": n_raws,
        "stage1_produced": len(successful_stems),
        "uploaded": n_uploaded,
        "posted": n_posted,
        "scratch": scratch,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="max Jobs per invocation")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--since", type=str, default=None,
                    help="ISO timestamp; default 2026-05-10T20:19:00-04:00 (may26 deploy)")
    ap.add_argument("--job", type=str, default=None, help="single Job ID")
    ap.add_argument("--dry-run", action="store_true",
                    help="run Stage-1 + log; skip uploads and dashboard POSTs")
    ap.add_argument("--force", action="store_true",
                    help="ignore alreadyStage1 filter; re-run every stem")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="don't delete /tmp dirs after each Job (debug)")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        print("ERROR: WORKER_TOKEN env var is required", file=sys.stderr)
        return 2
    if not CHECKPOINT_STAGE1.is_file():
        print(f"ERROR: Stage-1 ckpt not found at {CHECKPOINT_STAGE1}", file=sys.stderr)
        return 2

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    # Load the Stage-1 NAFNet + lensfunpy DB ONCE for the whole run.
    log(f"backfill: dashboard={DASHBOARD_URL}  stage1={CHECKPOINT_STAGE1}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device: {device}")
    model = load_stage1(CHECKPOINT_STAGE1, device)
    import lensfunpy  # type: ignore  # lazy — heavy DB
    lens_db = lensfunpy.Database()

    resp = get_candidates(
        limit=args.limit, offset=args.offset, since=args.since,
        job_id=args.job, force=args.force,
    )
    jobs = resp.get("jobs", [])
    log(f"  {len(jobs)} jobs to process (total candidates: {resp.get('total')})")
    if not jobs:
        return 0

    t0 = time.time()
    results: list[dict] = []
    for i, job in enumerate(jobs, 1):
        log(f"\n=== [{i}/{len(jobs)}] ===")
        try:
            r = run_job(
                job, model=model, device=device, lens_db=lens_db,
                scratch_root=WORK_ROOT, dry_run=args.dry_run,
            )
            results.append(r)
            if r.get("scratch") and not args.keep_scratch:
                shutil.rmtree(r["scratch"], ignore_errors=True)
        except Exception as e:
            log(f"  FAIL {job['id']}: {e}")
            results.append({"jobId": job["id"], "error": str(e)})

    elapsed = time.time() - t0
    total_raws = sum(r.get("raws", 0) for r in results)
    total_stage1 = sum(r.get("stage1_produced", 0) for r in results)
    total_posted = sum(r.get("posted", 0) for r in results)
    log(f"\n--- stage1 backfill done in {elapsed:.1f}s "
        f"({elapsed/max(1, len(results)):.1f}s/job avg) ---")
    log(f"  jobs: {len(results)}  raws: {total_raws}  "
        f"stage1: {total_stage1}  posted: {total_posted}")
    failed = [r for r in results if r.get("error")]
    if failed:
        log(f"  {len(failed)} job(s) errored:")
        for r in failed[:10]:
            log(f"    {r['jobId']}: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
