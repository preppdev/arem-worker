"""Evaluate the current auto-upright pipeline against the vendor's
05-Finished-Photos targets.

For a random sample of N paired (ours, vendor) images:

  1. Run the SAME Hough-based vertical-angle estimator on each.
     - vendor_angle ≈ what a downstream Hough would call "the residual
       rotation in vendor's already-finished image"
     - ours_angle   ≈ same on our pipeline's final output
  2. Feature-match ours → vendor via SIFT + RANSAC. Decompose the
     resulting homography to extract the *true* rotation between the
     two finished images. This is the rotation we'd need to apply to
     ours to align it with vendor — i.e., the residual error of our
     auto-upright.
  3. Aggregate: histogram of |residual|, per-room breakdown, worst-N
     outliers list. Write a CSV manifest + an HTML side-by-side report
     so the user can eyeball the failure cases.

Run on the 3090:
    DATABASE_URL=... \\
      /home/jordan/miniconda3/envs/arem-photo-ai/bin/python \\
      -m scripts.upright_eval --limit 50

Outputs land under /tmp/upright_eval/:
    /tmp/upright_eval/manifest.csv
    /tmp/upright_eval/report.html
    /tmp/upright_eval/img/<jobId>__<midStem>__{ours,vendor}.jpg
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore

OUT_ROOT = Path(os.environ.get("UPRIGHT_EVAL_ROOT", "/tmp/upright_eval"))
OUT_IMG = OUT_ROOT / "img"
RCLONE = ["rclone", "--config", str(Path.home() / ".config/rclone/rclone.conf")]


# ─── DB sample ──────────────────────────────────────────────────────────────

def fetch_sample(db_url: str, limit: int, seed: int) -> list[dict]:
    """Pick `limit` random paired ImageReview rows where we have both
    outputR2Path and vendorImageR2Path. Stratified by room so we don't
    get all-bedrooms by luck."""
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SET LOCAL seed = %s;
        SELECT ir.id, ir."jobId", ir."midStem",
               ir."outputR2Path", ir."vendorImageR2Path",
               COALESCE(ir."roomTypeCorrected", ir."roomTypeWorker") AS room,
               ir."isInteriorWorker" AS is_interior
        FROM "ImageReview" ir
        WHERE ir."outputR2Path" IS NOT NULL
          AND ir."vendorImageR2Path" IS NOT NULL
        ORDER BY random()
        LIMIT %s
    """, (seed / 1000.0, limit))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


# ─── image staging ─────────────────────────────────────────────────────────

def resolve_remote(path: str) -> str:
    p = path.lstrip("/")
    if p.startswith("AREM (Spiro Uploads)"):
        return f"dropbox:{p}"
    return f"r2:arem-training-data/{p}"


def stage_image(remote_path: str, local: Path) -> bool:
    if local.exists() and local.stat().st_size > 1000:
        return True
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = resolve_remote(remote_path)
    for _ in range(2):
        try:
            r = subprocess.run(
                [*RCLONE, "copyto", remote, str(local)],
                capture_output=True, timeout=60,
            )
            if r.returncode == 0 and local.exists() and local.stat().st_size > 1000:
                return True
        except subprocess.TimeoutExpired:
            pass
    return False


# ─── Hough vertical-angle estimator (mirrors auto_upright.py) ──────────────

def hough_vertical_angle(img: np.ndarray, min_len_frac: float = 0.05) -> tuple[float, int]:
    """Return (estimated_rotation_deg, n_vertical_segments). A positive
    angle means the image needs to be rotated CCW by that amount to make
    its dominant verticals truly vertical."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_len = int(min_len_frac * min(h, w))

    # LSD if available (faster + more accurate on architecture). Fall back to Hough.
    segs = []
    try:
        lsd = cv2.createLineSegmentDetector(0)
        out = lsd.detect(gray)[0]
        if out is not None:
            for x1, y1, x2, y2 in out.reshape(-1, 4):
                if math.hypot(x2 - x1, y2 - y1) >= min_len:
                    segs.append((x1, y1, x2, y2))
    except Exception:
        pass
    if not segs:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=min_len, maxLineGap=10)
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                segs.append((x1, y1, x2, y2))

    if not segs:
        return 0.0, 0

    # Bin near-vertical (|θ - 90°| < 25°). Take weighted mean of (θ - 90)
    # using segment length as weight; that residual = rotation needed.
    sum_w_angle = 0.0
    sum_w = 0.0
    n_vert = 0
    for x1, y1, x2, y2 in segs:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        # Angle from horizontal, 0..180
        theta_deg = math.degrees(math.atan2(dy, dx))
        # Fold to 0..180
        if theta_deg < 0:
            theta_deg += 180
        # vertical = ~90°
        delta = theta_deg - 90.0
        if abs(delta) < 25:
            sum_w_angle += delta * length
            sum_w += length
            n_vert += 1
    if sum_w == 0 or n_vert == 0:
        return 0.0, 0
    # The image's vertical lines are off-vertical by mean_delta.
    # Rotation needed = -mean_delta (CCW negative convention to match the
    # `imutils.rotate_bound`-style sign).
    mean_delta = sum_w_angle / sum_w
    return float(-mean_delta), n_vert


# ─── feature-match rotation extraction ─────────────────────────────────────

def estimate_relative_rotation(ours: np.ndarray, vendor: np.ndarray) -> float | None:
    """SIFT keypoints + RANSAC homography between ours and vendor. Decompose
    the resulting homography's upper-left 2x2 to pull out the rotation
    angle (degrees). Positive = ours needs to rotate CCW by this much to
    align with vendor.
    Returns None when feature matching fails (too few inliers etc.)."""
    sift = cv2.SIFT_create(nfeatures=4000)
    g1 = cv2.cvtColor(ours, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(vendor, cv2.COLOR_BGR2GRAY)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 50 or len(kp2) < 50:
        return None
    # FLANN matcher with Lowe's ratio test
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < 20:
        return None
    # Decompose: H ≈ scale * R + perspective. Pull rotation from the
    # upper-left 2x2 via SVD (Procrustes-style: H_2x2 = U S V^T, R = U V^T).
    a = H[:2, :2]
    U, _S, Vt = np.linalg.svd(a)
    R = U @ Vt
    # angle from R = [[cos, -sin], [sin, cos]]
    angle_rad = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(angle_rad)


# ─── per-image evaluation ──────────────────────────────────────────────────

def evaluate_one(row: dict) -> dict:
    stem = row["midStem"]
    job_id = row["jobId"]
    ours_local = OUT_IMG / f"{job_id}__{stem}__ours.jpg"
    vendor_local = OUT_IMG / f"{job_id}__{stem}__vendor.jpg"

    if not stage_image(row["outputR2Path"], ours_local):
        return {**row, "error": "stage ours failed"}
    if not stage_image(row["vendorImageR2Path"], vendor_local):
        return {**row, "error": "stage vendor failed"}

    ours = cv2.imread(str(ours_local))
    vendor = cv2.imread(str(vendor_local))
    if ours is None or vendor is None:
        return {**row, "error": "cv2 read failed"}

    ours_angle, ours_n = hough_vertical_angle(ours)
    vendor_angle, vendor_n = hough_vertical_angle(vendor)
    residual = estimate_relative_rotation(ours, vendor)

    return {
        **row,
        "ours_hough_deg": round(ours_angle, 3),
        "ours_n_vert": ours_n,
        "vendor_hough_deg": round(vendor_angle, 3),
        "vendor_n_vert": vendor_n,
        "ours_vs_vendor_residual_deg":
            round(residual, 3) if residual is not None else None,
        "error": None,
    }


# ─── report ────────────────────────────────────────────────────────────────

def write_outputs(results: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "manifest.csv"
    fields = [
        "jobId", "midStem", "room", "is_interior",
        "ours_hough_deg", "ours_n_vert",
        "vendor_hough_deg", "vendor_n_vert",
        "ours_vs_vendor_residual_deg",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    # Summary stats
    valid = [r for r in results
             if not r.get("error") and r.get("ours_vs_vendor_residual_deg") is not None]
    if valid:
        residuals = [abs(r["ours_vs_vendor_residual_deg"]) for r in valid]
        print(f"\n=== ours-vs-vendor rotation residual on n={len(valid)} ===")
        print(f"  mean |residual| = {np.mean(residuals):.3f} deg")
        print(f"  median |residual| = {np.median(residuals):.3f} deg")
        print(f"  p95 |residual| = {np.percentile(residuals, 95):.3f} deg")
        print(f"  max |residual| = {np.max(residuals):.3f} deg")

        buckets = {
            "0_lt_0.25": 0,
            "1_0.25-0.5": 0,
            "2_0.5-1.0": 0,
            "3_1.0-2.0": 0,
            "4_gt_2.0": 0,
        }
        for r in residuals:
            if r < 0.25:
                buckets["0_lt_0.25"] += 1
            elif r < 0.5:
                buckets["1_0.25-0.5"] += 1
            elif r < 1.0:
                buckets["2_0.5-1.0"] += 1
            elif r < 2.0:
                buckets["3_1.0-2.0"] += 1
            else:
                buckets["4_gt_2.0"] += 1
        print("  histogram:")
        for k, v in buckets.items():
            print(f"    {k:14s} {v:4d}")

    # HTML report — side-by-side, sortable by residual.
    html = OUT_ROOT / "report.html"
    rows_html = []
    sorted_results = sorted(
        results,
        key=lambda r: abs(r.get("ours_vs_vendor_residual_deg") or 0),
        reverse=True,
    )
    for r in sorted_results:
        if r.get("error"):
            continue
        ours_rel = f"img/{r['jobId']}__{r['midStem']}__ours.jpg"
        vendor_rel = f"img/{r['jobId']}__{r['midStem']}__vendor.jpg"
        residual = r.get("ours_vs_vendor_residual_deg")
        residual_str = f"{residual:+.3f}°" if residual is not None else "—"
        rows_html.append(f"""
        <tr>
            <td class="num">{residual_str}</td>
            <td>{r.get('room') or '(exterior)'}</td>
            <td class="num">{r.get('ours_hough_deg', 0):+.2f}°</td>
            <td class="num">{r.get('vendor_hough_deg', 0):+.2f}°</td>
            <td><img src="{ours_rel}" loading="lazy" /></td>
            <td><img src="{vendor_rel}" loading="lazy" /></td>
            <td class="meta">{r['jobId'][:8]}<br>{r['midStem']}</td>
        </tr>
        """)
    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Upright Eval</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #111; color: #ddd; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 6px 8px; vertical-align: top; }}
th {{ background: #222; text-align: left; position: sticky; top: 0; }}
td.num {{ font-family: SF Mono, Menlo, monospace; text-align: right; white-space: nowrap; }}
td.meta {{ font-family: SF Mono, Menlo, monospace; font-size: 11px; color: #888; }}
img {{ max-width: 480px; max-height: 320px; display: block; }}
h1 {{ font-size: 18px; margin-bottom: 4px; }}
h2 {{ font-size: 14px; color: #888; font-weight: normal; }}
</style></head><body>
<h1>Upright evaluation — ours vs vendor</h1>
<h2>Sorted by |ours→vendor residual rotation| desc. Larger numbers = worse straightening.</h2>
<table>
<thead><tr>
  <th>Residual</th><th>Room</th><th>Ours Hough</th><th>Vendor Hough</th>
  <th>Ours (our pipeline output)</th><th>Vendor (target)</th><th>ID / Stem</th>
</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</body></html>
"""
    html.write_text(html_content)
    print(f"wrote {html}")


# ─── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel rclone staging workers")
    args = ap.parse_args()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 2
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    sample = fetch_sample(db_url, args.limit, args.seed)
    print(f"sampled {len(sample)} paired rows")
    if not sample:
        return 0

    # Stage all images in parallel first (rclone-bound).
    print("staging images…")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs: list = []
        for r in sample:
            futs.append(pool.submit(
                stage_image, r["outputR2Path"],
                OUT_IMG / f"{r['jobId']}__{r['midStem']}__ours.jpg"
            ))
            futs.append(pool.submit(
                stage_image, r["vendorImageR2Path"],
                OUT_IMG / f"{r['jobId']}__{r['midStem']}__vendor.jpg"
            ))
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception:
                pass
            if i % 20 == 0:
                print(f"  {i}/{len(futs)} staged", flush=True)

    print("\nevaluating…")
    results: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(sample, 1):
        results.append(evaluate_one(r))
        if i % 10 == 0:
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(sample)}  ({rate:.1f}/sec)", flush=True)

    write_outputs(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
