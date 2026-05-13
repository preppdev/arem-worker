"""Produce histogram-matched derivatives of ours JPGs aligned to vendor references.

Companion to scripts/backfill_classifications.py. The backfill mirrors
vendor JPGs to R2 (vendor-mirrors/<jobId>/<midStem>.jpg) and populates
ImageReview.vendorImageR2Path. This script consumes those pairs and
produces two pixel variants per ImageReview row:

  matchedRgbR2Path        skimage match_histograms per-channel RGB CDF
                          → r2:arem-training-data/image-matched/rgb/<jobId>/<stem>.jpg

  matchedReinhardR2Path   Reinhard LAB mean+std transfer
                          → r2:arem-training-data/image-matched/reinhard/<jobId>/<stem>.jpg

The original ours JPG (in Dropbox 08-Test-Edit or R2 manual_uploads/.../outputs)
is untouched — these are derivatives. The matched JPGs preserve the ours
resolution; histograms are computed on full-res pixels (~50ms/image).

Both algorithms run per-pair (one LUT per ours-vendor pair, not shared
across the Job). RGB-CDF tends to be more aggressive than Reinhard but
can occasionally introduce banding on smooth gradients; the side-by-side
output lets reviewers compare on a per-shoot basis during fine-tuning.

Run from arem-worker repo root:
    python -m scripts.match_histograms --limit 1 --dry-run
    python -m scripts.match_histograms --limit 50
    python -m scripts.match_histograms --job <jobId>

Depends on dashboard endpoints in arem-editing#18.
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

import cv2  # type: ignore
import numpy as np  # type: ignore
from skimage.exposure import match_histograms  # type: ignore


DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
RCLONE_DROPBOX = os.environ.get("RCLONE_DROPBOX", "dropbox")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/tmp/arem-match"))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---- HTTP helpers ----


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {body_text[:300]}") from e


def get_candidates(*, limit: int, offset: int, job_id: str | None, force: bool) -> dict:
    sp = {"limit": str(limit), "offset": str(offset)}
    if job_id:
        sp["jobId"] = job_id
    if force:
        sp["force"] = "1"
    return _request("GET", f"/api/internal/image-match-candidates?{urllib.parse.urlencode(sp)}")


def post_match(payload: dict) -> dict:
    return _request("POST", "/api/internal/image-match", payload)


# ---- rclone helpers ----


def rclone(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True, timeout=timeout)


def resolve_dropbox(path_under_root: str) -> str:
    p = path_under_root.strip().rstrip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/"):]
    return f"{RCLONE_DROPBOX}:{p}"


def resolve_r2(key: str) -> str:
    return f"{RCLONE_R2}:{R2_BUCKET}/{key.lstrip('/')}"


def rclone_copy_jpgs(src: str, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    r = rclone(
        ["copy", src, str(dst_dir),
         "--include", "*.jpg", "--include", "*.jpeg",
         "--include", "*.JPG", "--include", "*.JPEG",
         "--transfers", "8", "--checkers", "8",
         "--ignore-existing"],
        timeout=900,
    )
    if r.returncode != 0:
        log(f"    rclone copy {src} rc={r.returncode}: {r.stderr[-160:].strip()}")
        return 0
    return sum(1 for p in dst_dir.iterdir() if p.is_file())


# ---- Histogram-match algorithms ----


def match_rgb_cdf(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Per-channel CDF match via skimage. Returns BGR uint8 at src resolution.
    The reference is used only for its histogram — its resolution doesn't
    matter (we don't resample either image)."""
    # skimage operates on the array as-is; channel_axis=2 means per-channel.
    matched = match_histograms(src_bgr, ref_bgr, channel_axis=2)
    return np.clip(matched, 0, 255).astype(np.uint8)


def match_reinhard_lab(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """Reinhard mean+std transfer in LAB. Linear remap per channel:
        out = ((src - src_mean) / src_std) * ref_std + ref_mean
    Standard, fast, and less aggressive than full CDF — preserves chroma
    structure better."""
    # OpenCV's BGR→LAB uses 0–255 for L and 0–255 for a/b with offsets of 128
    # for a/b. Cast to float32 to avoid uint8 wraparound on the linear math.
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = np.empty_like(src_lab)
    for c in range(3):
        s_mean, s_std = src_lab[..., c].mean(), src_lab[..., c].std()
        r_mean, r_std = ref_lab[..., c].mean(), ref_lab[..., c].std()
        if s_std < 1e-6:
            # Flat channel — just shift to ref mean.
            out[..., c] = src_lab[..., c] - s_mean + r_mean
        else:
            out[..., c] = (src_lab[..., c] - s_mean) * (r_std / s_std) + r_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def save_jpg(bgr: np.ndarray, dst: Path, quality: int = 92) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {dst}")


# ---- per-Job processing ----


def run_job(
    job: dict,
    *,
    scratch_root: Path,
    dry_run: bool,
) -> dict:
    job_id: str = job["id"]
    task_no = job.get("taskNumber")
    src: dict = job["outputLocation"]
    items: list[dict] = job["items"]

    log(f"  [T-{task_no}] {job_id}  items={len(items)}  src={src['kind']}:{src['path']}")

    scratch = scratch_root / job_id
    scratch.mkdir(parents=True, exist_ok=True)
    ours_dir = scratch / "ours"
    vendor_dir = scratch / "vendor"
    rgb_dir = scratch / "rgb"
    rein_dir = scratch / "reinhard"
    for d in (ours_dir, vendor_dir, rgb_dir, rein_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Pull ours JPGs ---
    ours_remote = (
        resolve_dropbox(src["path"]) if src["kind"] == "dropbox"
        else resolve_r2(src["path"])
    )
    if rclone_copy_jpgs(ours_remote, ours_dir) == 0:
        log("  no ours JPGs at source — skip")
        return {"jobId": job_id, "items": 0, "skipped": True, "scratch": scratch}

    # --- Pull vendor JPGs (already on R2 from the classification backfill) ---
    # We pull the whole vendor-mirrors/<jobId> folder rather than each
    # midStem one-by-one — one rclone call vs many.
    rclone_copy_jpgs(resolve_r2(f"vendor-mirrors/{job_id}"), vendor_dir)

    # --- Per-item: load both, match, save ---
    n_matched = 0
    failures: list[tuple[str, str]] = []
    payloads: list[dict] = []

    for it in items:
        stem = it["midStem"]
        ours_path = ours_dir / f"{stem}.jpg"
        if not ours_path.is_file():
            # ours uses .jpeg sometimes, or different casing — try alternates
            alt = next((p for p in ours_dir.iterdir()
                        if p.is_file() and p.stem == stem), None)
            if alt is None:
                failures.append((stem, "ours JPG not in 08-Test-Edit"))
                continue
            ours_path = alt
        vendor_path = vendor_dir / f"{stem}.jpg"
        if not vendor_path.is_file():
            failures.append((stem, "vendor JPG missing from R2 mirror"))
            continue

        try:
            src_bgr = cv2.imread(str(ours_path), cv2.IMREAD_COLOR)
            ref_bgr = cv2.imread(str(vendor_path), cv2.IMREAD_COLOR)
            if src_bgr is None or ref_bgr is None:
                raise RuntimeError("cv2.imread returned None")
        except Exception as e:
            failures.append((stem, f"imread: {e}"))
            continue

        payload: dict = {"jobId": job_id, "midStem": stem}

        try:
            matched_rgb = match_rgb_cdf(src_bgr, ref_bgr)
            save_jpg(matched_rgb, rgb_dir / f"{stem}.jpg")
            payload["matchedRgbR2Path"] = f"image-matched/rgb/{job_id}/{stem}.jpg"
        except Exception as e:
            failures.append((stem, f"rgb-cdf: {e}"))

        try:
            matched_rein = match_reinhard_lab(src_bgr, ref_bgr)
            save_jpg(matched_rein, rein_dir / f"{stem}.jpg")
            payload["matchedReinhardR2Path"] = f"image-matched/reinhard/{job_id}/{stem}.jpg"
        except Exception as e:
            failures.append((stem, f"reinhard: {e}"))

        if "matchedRgbR2Path" in payload or "matchedReinhardR2Path" in payload:
            payloads.append(payload)
            n_matched += 1

    log(f"  matched {n_matched}/{len(items)} pairs")

    # --- Upload matched JPGs (two rclone copy batches) + POST ---
    n_posted = 0
    if not dry_run:
        for label, local_dir, remote_key in (
            ("rgb", rgb_dir, f"image-matched/rgb/{job_id}"),
            ("reinhard", rein_dir, f"image-matched/reinhard/{job_id}"),
        ):
            count = sum(1 for p in local_dir.iterdir() if p.is_file())
            if count == 0:
                continue
            r = rclone(
                ["copy", str(local_dir), resolve_r2(remote_key),
                 "--transfers", "8", "--checkers", "8"],
                timeout=1200,
            )
            if r.returncode != 0:
                log(f"  WARN {label} upload rc={r.returncode}: {r.stderr[-160:].strip()}")

        for p in payloads:
            try:
                post_match(p)
                n_posted += 1
            except Exception as e:
                failures.append((p["midStem"], f"post: {e}"))

        log(f"  posted {n_posted}/{len(payloads)}")
    else:
        for p in payloads[:3]:
            log(f"  [dry] {p['midStem']}: "
                f"{json.dumps({k: v for k, v in p.items() if k not in ('jobId', 'midStem')})}")
        if len(payloads) > 3:
            log(f"  [dry] ... + {len(payloads) - 3} more")

    return {
        "jobId": job_id,
        "items": len(items),
        "matched": n_matched,
        "posted": n_posted,
        "failures": failures,
        "scratch": scratch,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="max Jobs per invocation")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--job", type=str, default=None, help="single Job ID")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute matches + log; skip uploads and dashboard POSTs")
    ap.add_argument("--force", action="store_true",
                    help="include already-matched rows")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="don't delete /tmp dirs after each Job")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        print("ERROR: WORKER_TOKEN env var is required", file=sys.stderr)
        return 2

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    log(f"match: dashboard={DASHBOARD_URL}")
    resp = get_candidates(
        limit=args.limit, offset=args.offset, job_id=args.job, force=args.force,
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
            r = run_job(job, scratch_root=WORK_ROOT, dry_run=args.dry_run)
            results.append(r)
            if r.get("scratch") and not args.keep_scratch:
                shutil.rmtree(r["scratch"], ignore_errors=True)
        except Exception as e:
            log(f"  FAIL {job['id']}: {e}")
            results.append({"jobId": job["id"], "error": str(e)})

    elapsed = time.time() - t0
    total_items = sum(r.get("items", 0) for r in results)
    total_matched = sum(r.get("matched", 0) for r in results)
    total_posted = sum(r.get("posted", 0) for r in results)
    log(f"\n--- match done in {elapsed:.1f}s ---")
    log(f"  jobs: {len(results)}  items: {total_items}  matched: {total_matched}  posted: {total_posted}")
    bad = [r for r in results if r.get("failures") or r.get("error")]
    if bad:
        log(f"  {len(bad)} job(s) had issues:")
        for r in bad:
            if r.get("error"):
                log(f"    {r['jobId']}: {r['error']}")
            for stem, msg in (r.get("failures") or [])[:5]:
                log(f"    {r['jobId']}/{stem}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
