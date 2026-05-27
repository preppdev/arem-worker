"""Backfill classification + thumbnails + vendor mirrors for pre-PR-#8 Jobs.

Phase 3 of the worker-classification rollout. The live worker (PR #8) emits
isInteriorWorker/roomTypeWorker/thumbnailR2Path on every new Job. This script
retroactively does the same for already-completed Jobs by:

  1. Asking the dashboard which Jobs need backfill (GET /api/internal/backfill-jobs).
  2. For each Job:
       a. rclone copy the output JPGs (Dropbox 08-Test-Edit/ or R2 manual_uploads/.../outputs/)
          and the vendor reference JPGs (Dropbox 05-Finished-Photos/, if present) to local scratch.
       b. Pair ours ↔ vendor with the same fuzzy-match algorithm /review uses (PR #10).
       c. For each ours JPG, run interior + room classifiers, generate a 512px thumb.
          For matched vendor JPGs, generate a 512px thumb under ours' stem and rename
          the full-res copy to ours' stem so R2 keys line up.
       d. rclone copy thumbs + vendor mirrors to R2.
       e. POST one ImageReview upsert per image to /api/internal/image-classification.

The pre-PR-#8 output JPGs DON'T have EXIF UserComment classification embedded — only
JPGs the live worker produces after that PR do. The backfill is "read-only" on Dropbox:
no re-uploads, no EXIF rewrites. The classification snapshot lives in Postgres
(ImageReview) and R2 (thumbnails + vendor mirrors).

Run from the arem-worker repo root:
    python -m scripts.backfill_classifications --limit 1 --dry-run
    python -m scripts.backfill_classifications --limit 50
    python -m scripts.backfill_classifications --job <jobId>
    python -m scripts.backfill_classifications --since 2026-04-01

Env vars (inherited from worker.py defaults):
    WORKER_TOKEN, DASHBOARD_URL, RCLONE_DROPBOX, CLASSIFIER_PATH, CLASSIFIER_ROOM,
    WORK_ROOT
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import torch  # type: ignore

# Reuse the worker's classifier/thumbnail helpers. Importing run_pipeline pulls
# in the ML deps; that's fine — we'd be loading them anyway.
from pipeline.run_pipeline import (  # type: ignore  # noqa: E402
    classify,
    classify_room,
    load_classifier,
    load_room_classifier,
    make_thumbnail,
)

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
RCLONE_DROPBOX = os.environ.get("RCLONE_DROPBOX", "dropbox")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/tmp/arem-backfill"))
CLASSIFIER_PATH = Path(
    os.environ.get(
        "CLASSIFIER_PATH",
        str(Path(__file__).resolve().parent.parent / "pipeline" / "classifier_v2.pth"),
    )
)
CLASSIFIER_ROOM_ENV = os.environ.get("CLASSIFIER_ROOM")


def log(msg: str) -> None:
    print(msg, flush=True)


# ---- dashboard HTTP helpers ----


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "x-worker-token": WORKER_TOKEN,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {body_text[:300]}") from e


def get_backfill_jobs(
    *,
    limit: int,
    offset: int,
    order: str,
    since: str | None,
    job_id: str | None,
    force: bool,
    reclassify_v1: bool = False,
) -> dict:
    sp = {"limit": str(limit), "offset": str(offset), "order": order}
    if since:
        sp["since"] = since
    if job_id:
        sp["jobId"] = job_id
    if force:
        sp["force"] = "1"
    if reclassify_v1:
        sp["reclassifyV1"] = "1"
    qs = urllib.parse.urlencode(sp)
    return _request("GET", f"/api/internal/backfill-jobs?{qs}")


def post_classification(payload: dict) -> dict:
    return _request("POST", "/api/internal/image-classification", payload)


# ---- rclone helpers ----


def rclone(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    cmd = ["rclone", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def resolve_dropbox(path_under_root: str) -> str:
    """Strip the 'Jordan DiCaprio/' prefix (personal dropbox root) and
    prepend the rclone remote, matching worker.py's resolution rules."""
    p = path_under_root.strip().rstrip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/"):]
    return f"{RCLONE_DROPBOX}:{p}"


def resolve_r2_key(key: str) -> str:
    return f"{RCLONE_R2}:{R2_BUCKET}/{key.lstrip('/')}"


def rclone_copy_jpgs(src: str, dst_dir: Path) -> int:
    """Copy only .jpg/.jpeg files from src to dst_dir. Returns the number
    of files now in dst_dir, or 0 if the source doesn't exist."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    proc = rclone(
        [
            "copy", src, str(dst_dir),
            "--include", "*.jpg", "--include", "*.jpeg",
            "--include", "*.JPG", "--include", "*.JPEG",
            "--transfers", "8", "--checkers", "8",
            "--ignore-existing",
        ],
        timeout=900,
    )
    if proc.returncode != 0:
        # Most common cause: source folder doesn't exist. Treat as empty.
        log(f"    rclone copy {src} rc={proc.returncode}: {proc.stderr[-160:].strip()}")
        return 0
    return sum(1 for p in dst_dir.iterdir() if p.is_file())


# ---- fuzzy match (Python port of app/api/review/[jobId]/route.ts, PR #10) ----


def parse_stem_num(stem: str) -> int | None:
    """Return the trailing integer of a JPG stem (e.g. 'DSC_0613' → 613).
    None when there's no numeric tail (legacy hand-named outputs)."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def build_pairs(ours_stems: list[str], vendor_files: list[Path]) -> dict[str, Path]:
    """For each ours stem, find the best vendor file. First tries exact
    stem match (and underscore-stripped variant), then falls back to the
    offset-histogram + tie-break-by-smallest-|offset| algorithm from PR
    #10. Returns {ours_stem: vendor_path}; ours stems without a match are
    absent from the dict."""
    HISTOGRAM_WIDE = 10
    PAIR_TIGHT = 1

    # Exact / underscore-stripped lookup table
    vendor_by_key: dict[str, Path] = {}
    vendor_with_num: list[tuple[str, int, Path]] = []
    for vp in vendor_files:
        stem = vp.stem.lower()
        vendor_by_key.setdefault(stem, vp)
        vendor_by_key.setdefault(stem.replace("_", ""), vp)
        n = parse_stem_num(stem)
        if n is not None:
            vendor_with_num.append((stem, n, vp))

    pairs: dict[str, Path] = {}
    leftover_ours: list[tuple[str, int]] = []
    for s in ours_stems:
        lower = s.lower()
        hit = vendor_by_key.get(lower) or vendor_by_key.get(lower.replace("_", ""))
        if hit is not None:
            pairs[s] = hit
            continue
        n = parse_stem_num(s)
        if n is not None:
            leftover_ours.append((s, n))

    if not leftover_ours or not vendor_with_num:
        return pairs

    # Step 1+2: histogram of ours-num − vendor-num across all in-window pairs.
    offset_counts: dict[int, int] = {}
    for _, o_num in leftover_ours:
        for _, v_num, _ in vendor_with_num:
            d = o_num - v_num
            if abs(d) <= HISTOGRAM_WIDE:
                offset_counts[d] = offset_counts.get(d, 0) + 1
    if not offset_counts:
        return pairs

    expected = 0
    best_votes = 0
    for d, c in offset_counts.items():
        # Tie-break by smallest |offset| (PR #10 — fixes T-775 mismatch).
        if c > best_votes or (c == best_votes and abs(d) < abs(expected)):
            best_votes = c
            expected = d

    # Step 3: pair ours → vendor at offset ± 1, each vendor consumed once.
    used: set[str] = set()
    # Iterate ours in numeric order so deterministic on ties.
    leftover_ours.sort(key=lambda x: x[1])
    for s, o_num in leftover_ours:
        target = o_num - expected
        best: tuple[str, int, Path] | None = None
        best_dist = float("inf")
        for v_stem, v_num, v_path in vendor_with_num:
            if v_stem in used:
                continue
            dist = abs(v_num - target)
            if dist <= PAIR_TIGHT and dist < best_dist:
                best = (v_stem, v_num, v_path)
                best_dist = dist
        if best is not None:
            pairs[s] = best[2]
            used.add(best[0])
    return pairs


# ---- per-Job processing ----


def run_job(
    job: dict,
    *,
    clf,
    clf_labels,
    clf_tfm,
    room_model,
    device,
    scratch_root: Path,
    skip_vendor: bool,
    dry_run: bool,
) -> dict:
    """Process one Job. Returns a result dict for the final summary."""
    job_id: str = job["id"]
    task_no = job.get("taskNumber")
    src: dict = job["outputLocation"]
    vendor_src: dict | None = job.get("vendorLocation")
    already: set[str] = set(job.get("alreadyClassifiedMidStems") or [])

    log(
        f"  [T-{task_no}] {job_id}  "
        f"src={src['kind']}:{src['path']}  "
        f"vendor={vendor_src['path'] if vendor_src else '—'}  "
        f"alreadyClassified={len(already)}"
    )

    scratch = scratch_root / job_id
    scratch.mkdir(parents=True, exist_ok=True)
    ours_dir = scratch / "ours"
    vendor_dir = scratch / "vendor"
    ours_thumbs = scratch / "ours-thumbs"
    vendor_thumbs = scratch / "vendor-thumbs"
    vendor_renamed = scratch / "vendor-renamed"
    for d in (ours_thumbs, vendor_thumbs, vendor_renamed):
        d.mkdir(parents=True, exist_ok=True)

    # --- Pull outputs ---
    ours_remote = (
        resolve_dropbox(src["path"]) if src["kind"] == "dropbox"
        else resolve_r2_key(src["path"])
    )
    if rclone_copy_jpgs(ours_remote, ours_dir) == 0:
        log("  no output JPGs at source — skip")
        return {"jobId": job_id, "imagesFound": 0, "skipped": True, "scratch": scratch}

    # --- Pull vendor (optional) ---
    if vendor_src and not skip_vendor:
        vendor_remote = resolve_dropbox(vendor_src["path"])
        rclone_copy_jpgs(vendor_remote, vendor_dir)

    ours_jpgs = sorted(
        p for p in ours_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
    )
    vendor_jpgs = (
        sorted(p for p in vendor_dir.iterdir()
               if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
        if vendor_dir.is_dir() else []
    )
    log(f"  ours={len(ours_jpgs)} vendor={len(vendor_jpgs)}")

    ours_stems = [p.stem for p in ours_jpgs]
    vendor_pairs = build_pairs(ours_stems, vendor_jpgs) if vendor_jpgs else {}
    if vendor_jpgs:
        log(f"  vendor-match: {len(vendor_pairs)}/{len(ours_stems)} ours stems paired")

    # --- Per-image: classify + stage thumbs/mirrors + build POST payload ---
    n_classified = 0
    n_thumbs_ours = 0
    n_thumbs_vendor = 0
    n_vendor_full = 0
    failures: list[tuple[str, str]] = []
    # Payloads collected for later POST (after R2 batch uploads complete, so
    # readers see the row only once the paths it references are live).
    payloads: list[dict] = []

    for jpg in ours_jpgs:
        stem = jpg.stem
        payload: dict = {"jobId": job_id, "midStem": stem}
        already_done = stem in already

        # Classifier (skip when live worker already covered this image)
        if not already_done:
            try:
                is_int, _conf, _ = classify(clf, clf_labels, clf_tfm, jpg, device)
                room = classify_room(room_model, jpg, device) if is_int else None
                payload.update({
                    "isInteriorWorker": bool(is_int),
                    "roomTypeWorker": (room or {}).get("roomType"),
                    "roomConfidenceWorker": (
                        round((room or {}).get("roomConfidence") or 0.0, 4)
                        if room else None
                    ),
                    "classifierModelVersions": {
                        "interior_classifier": CLASSIFIER_PATH.name,
                        "room_classifier": (room or {}).get("version"),
                        "backfill": True,
                    },
                })
                n_classified += 1
            except Exception as e:
                failures.append((stem, f"classify: {e}"))

        # Ours thumb — always attempt (older jobs may lack one even when classifiedAt is set)
        try:
            make_thumbnail(jpg, ours_thumbs / f"{stem}.jpg")
            n_thumbs_ours += 1
            payload["thumbnailR2Path"] = f"image-thumbnails/{job_id}/{stem}.jpg"
        except Exception as e:
            failures.append((stem, f"ours-thumb: {e}"))

        # Vendor mirror + thumb when fuzzy-match found a pair
        vp = vendor_pairs.get(stem)
        if vp is not None:
            try:
                shutil.copyfile(vp, vendor_renamed / f"{stem}.jpg")
                n_vendor_full += 1
                payload["vendorImageR2Path"] = f"vendor-mirrors/{job_id}/{stem}.jpg"
            except Exception as e:
                failures.append((stem, f"vendor-copy: {e}"))
            try:
                make_thumbnail(vp, vendor_thumbs / f"{stem}.jpg")
                n_thumbs_vendor += 1
                payload["vendorThumbnailR2Path"] = (
                    f"image-thumbnails/vendor/{job_id}/{stem}.jpg"
                )
            except Exception as e:
                failures.append((stem, f"vendor-thumb: {e}"))

        # Skip the POST when nothing meaningful changed (already classified +
        # already had thumbs uploaded from a prior backfill + no new vendor match)
        if not any(k in payload for k in (
            "isInteriorWorker", "thumbnailR2Path",
            "vendorImageR2Path", "vendorThumbnailR2Path",
        )):
            continue
        payloads.append(payload)

    # --- Upload all staged artifacts in 3 rclone batches ---
    if not dry_run:
        for label, local_dir, remote_key, count in (
            ("ours-thumb", ours_thumbs, f"image-thumbnails/{job_id}", n_thumbs_ours),
            ("vendor-thumb", vendor_thumbs, f"image-thumbnails/vendor/{job_id}", n_thumbs_vendor),
            ("vendor-mirror", vendor_renamed, f"vendor-mirrors/{job_id}", n_vendor_full),
        ):
            if count == 0:
                continue
            r = rclone(
                ["copy", str(local_dir), resolve_r2_key(remote_key),
                 "--transfers", "16", "--checkers", "16"],
                timeout=1200,
            )
            if r.returncode != 0:
                log(f"  WARN {label} upload rc={r.returncode}: {r.stderr[-160:].strip()}")

    # --- POST one record per staged image ---
    n_posted = 0
    if dry_run:
        for p in payloads[:3]:
            log(f"  [dry] {p['midStem']}: "
                f"{json.dumps({k: v for k, v in p.items() if k not in ('jobId', 'midStem')})}")
        if len(payloads) > 3:
            log(f"  [dry] ... + {len(payloads) - 3} more")
    else:
        for p in payloads:
            try:
                post_classification(p)
                n_posted += 1
            except Exception as e:
                failures.append((p["midStem"], f"post: {e}"))

    log(f"  → classified={n_classified} thumbsOurs={n_thumbs_ours} "
        f"thumbsVendor={n_thumbs_vendor} vendorFull={n_vendor_full} posted={n_posted}"
        f"{'  [DRY]' if dry_run else ''}")

    return {
        "jobId": job_id,
        "imagesFound": len(ours_jpgs),
        "classified": n_classified,
        "thumbsOurs": n_thumbs_ours,
        "thumbsVendor": n_thumbs_vendor,
        "vendorFull": n_vendor_full,
        "posted": n_posted,
        "failures": failures,
        "scratch": scratch,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="max jobs per invocation")
    ap.add_argument("--offset", type=int, default=0, help="pagination offset")
    ap.add_argument("--order", choices=["oldest", "newest"], default="oldest")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO date; only jobs with completedAt >= this")
    ap.add_argument("--job", type=str, default=None, help="single job ID")
    ap.add_argument("--dry-run", action="store_true",
                    help="classify + log, but skip R2 uploads and dashboard POSTs")
    ap.add_argument("--force", action="store_true",
                    help="re-run classifier on every image regardless of prior state")
    ap.add_argument("--reclassify-v1", action="store_true",
                    help="treat stage-1-only + v1-classified interior rows as needing "
                         "(re-)classification by the active room ckpt. Exteriors stay done.")
    ap.add_argument("--no-vendor", action="store_true",
                    help="skip 05-Finished-Photos mirroring (ours-only)")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="don't delete scratch dirs after each Job (debug)")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        print("ERROR: WORKER_TOKEN env var is required", file=sys.stderr)
        return 2

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    log(f"backfill: dashboard={DASHBOARD_URL}  classifier={CLASSIFIER_PATH}  "
        f"room_clf={CLASSIFIER_ROOM_ENV or '(none)'}")

    # Load classifiers ONCE for the whole run.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device: {device}")
    clf, clf_labels, clf_tfm = load_classifier(CLASSIFIER_PATH, device)
    room_model = load_room_classifier(
        Path(CLASSIFIER_ROOM_ENV) if CLASSIFIER_ROOM_ENV else None, device,
    )

    resp = get_backfill_jobs(
        limit=args.limit, offset=args.offset, order=args.order,
        since=args.since, job_id=args.job, force=args.force,
        reclassify_v1=args.reclassify_v1,
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
                job,
                clf=clf, clf_labels=clf_labels, clf_tfm=clf_tfm,
                room_model=room_model, device=device,
                scratch_root=WORK_ROOT,
                skip_vendor=args.no_vendor,
                dry_run=args.dry_run,
            )
            results.append(r)
            if r.get("scratch") and not args.keep_scratch:
                shutil.rmtree(r["scratch"], ignore_errors=True)
        except Exception as e:
            log(f"  FAIL {job['id']}: {e}")
            results.append({"jobId": job["id"], "error": str(e)})

    # --- Summary ---
    elapsed = time.time() - t0
    total_imgs = sum(r.get("imagesFound", 0) for r in results)
    total_cls = sum(r.get("classified", 0) for r in results)
    total_post = sum(r.get("posted", 0) for r in results)
    log(f"\n--- backfill done in {elapsed:.1f}s ---")
    log(f"  jobs: {len(results)}  images: {total_imgs}  classified: {total_cls}  posted: {total_post}")
    fail_jobs = [r for r in results if r.get("failures") or r.get("error")]
    if fail_jobs:
        log(f"  {len(fail_jobs)} job(s) had issues:")
        for r in fail_jobs:
            if r.get("error"):
                log(f"    {r['jobId']}: {r['error']}")
            for stem, msg in (r.get("failures") or [])[:5]:
                log(f"    {r['jobId']}/{stem}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
