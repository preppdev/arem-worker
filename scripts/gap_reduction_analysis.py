"""Quantify how much the histogram matchers close the (ours, vendor) style gap.

For each ImageReview row where matchedAt IS NOT NULL, computes two
complementary distance metrics on three configurations:

  baseline   ours              vs vendor — the gap matching tries to close
  rgb        matched-RGB-CDF   vs vendor — residual after per-channel CDF
  reinhard   matched-Reinhard  vs vendor — residual after LAB mean+std

Metrics:
  hist_L2    L2 distance over an 8-vector of histogram stats
             (luma mean/std, per-channel means, R/G + B/G WB ratios,
             saturation). Strong on global color/WB drift, weak on
             texture / sharpness / structural differences.
  lpips      Learned Perceptual Image Patch Similarity — pretrained
             VGG feature-space distance. Strong on texture / structure /
             sharpness — the "looks different to a human" axis that
             histograms miss. Weaker on subtle global color casts.

Both reported because they're complementary: histograms catch the
gap that matching IS designed to close; LPIPS catches the gap that
remains AFTER matching closes it (where fine-tuning has to do the
work).

Reads from R2 only (uses the thumbnail paths populated by the
classification backfill + the matched paths populated by the match
script). Doesn't need Dropbox auth.

Run on the GPU box (has rclone + Python deps):

    set -a && . /etc/arem-worker.env && set +a
    python3 scripts/gap_reduction_analysis.py --limit 200

Args:
    --limit N        Sample N rows (default 200)
    --resize PX      Resize all images to this longest-edge before stats
                     (default 512 — matches the thumbnail resolution)
    --since DATE     Filter Job.completedAt >= DATE (ISO)
    --csv FILE       Write per-image stats CSV alongside the summary
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import torch  # type: ignore
from PIL import Image


RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
RCLONE_DROPBOX = os.environ.get("RCLONE_DROPBOX", "dropbox")

# LPIPS model is lazy-loaded on first use (heavy import; ~250 MB weights
# pulled the first time). Reused across all pairs in a run.
_LPIPS_MODEL: tuple[object, torch.device] | None = None


def _get_lpips() -> tuple[object, torch.device]:
    """Return (LPIPS model, device). Loads on first call."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    import lpips  # type: ignore  # lazy import; heavy weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # VGG backbone — more stable / better-correlated to human judgment
    # than AlexNet at moderate cost.
    model = lpips.LPIPS(net="vgg").to(device).eval()
    _LPIPS_MODEL = (model, device)
    return _LPIPS_MODEL


def _to_lpips_tensor_pair(
    a_path: Path, b_path: Path, size: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load two JPGs and resize BOTH to the same target shape, preserving
    each image's aspect ratio but ensuring the resulting tensors share
    exact (H, W). Previous version resized each image independently to
    `size` longest-edge, which produced off-by-one shape mismatches when
    the two source images had slightly different aspect ratios (170 vs
    171 px on a non-square axis) and broke LPIPS on those pairs.

    Strategy: compute the target (h, w) from the FIRST image's aspect at
    `size` longest-edge, then force the second image to the same shape
    via PIL resize. Aspect-ratio difference between ours and vendor is
    usually <1% so the second-image distortion is imperceptible at the
    image level (LPIPS isn't shift- or scale-sensitive at this magnitude).
    """
    a = Image.open(a_path).convert("RGB")
    aw, ah = a.size
    ratio = size / max(aw, ah)
    target_w = max(1, int(round(aw * ratio)))
    target_h = max(1, int(round(ah * ratio)))
    a = a.resize((target_w, target_h), Image.LANCZOS)
    b = Image.open(b_path).convert("RGB").resize((target_w, target_h), Image.LANCZOS)

    def to_t(im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return to_t(a), to_t(b)


def lpips_distance(a_path: Path, b_path: Path, size: int = 512) -> float:
    """Perceptual distance between two images. Returns a float ≥ 0,
    lower = more similar. Typical range: 0.0–0.7."""
    model, device = _get_lpips()
    with torch.no_grad():
        a, b = _to_lpips_tensor_pair(a_path, b_path, size, device)
        d = model(a, b)
    return float(d.item())


def rclone_get_r2(key: str, dst: Path, timeout: int = 120) -> bool:
    """Pull one R2 object to a local path. Returns True if file landed."""
    src = f"{RCLONE_R2}:{R2_BUCKET}/{key.lstrip('/')}"
    proc = subprocess.run(
        ["rclone", "copyto", src, str(dst), "--retries", "1"],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def rclone_get_dropbox(dropbox_path: str, dst: Path, timeout: int = 180) -> bool:
    """Pull one Dropbox object to a local path. Handles the Jordan DiCaprio/
    prefix stripping that the live worker also does (personal Dropbox
    rooted differently from the rclone remote)."""
    p = dropbox_path.strip().lstrip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/"):]
    src = f"{RCLONE_DROPBOX}:{p}"
    proc = subprocess.run(
        ["rclone", "copyto", src, str(dst), "--retries", "1"],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def stats(jpg_path: Path, resize_to: int) -> np.ndarray:
    """Style-stat vector for one image. Returns a numpy float64 array:

       [luma_mean, luma_std,
        r_mean, g_mean, b_mean,
        rg_ratio, bg_ratio,         # white-balance proxy
        sat_mean]                   # max(R,G,B) - min(R,G,B)

    All values normalized to [0,1] (8-bit pixels divided by 255). Resize
    to a common longest-edge so resolution differences between ours
    (512px thumb) and matched (full-res) don't bias the stats.
    """
    im = Image.open(jpg_path).convert("RGB")
    w, h = im.size
    if max(w, h) > resize_to:
        ratio = resize_to / max(w, h)
        im = im.resize((max(1, int(w * ratio)), max(1, int(h * ratio))),
                       Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = arr.max(axis=-1) - arr.min(axis=-1)
    g_safe = g.mean() if g.mean() > 1e-4 else 1e-4
    return np.array([
        luma.mean(), luma.std(),
        r.mean(), g.mean(), b.mean(),
        r.mean() / g_safe, b.mean() / g_safe,
        sat.mean(),
    ], dtype=np.float64)


METRIC_NAMES = [
    "luma_mean", "luma_std",
    "r_mean", "g_mean", "b_mean",
    "rg_ratio", "bg_ratio",
    "sat_mean",
]


def fetch_rows(limit: int, since: str | None, axis: str = "all") -> list[dict]:
    """Pull a sample of ImageReview rows ready for analysis."""
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    where = [
        '"matchedAt" IS NOT NULL',
        '"thumbnailR2Path" IS NOT NULL',
        '"vendorThumbnailR2Path" IS NOT NULL',
        '"matchedRgbR2Path" IS NOT NULL',
        '"matchedReinhardR2Path" IS NOT NULL',
    ]
    params: list = []
    if since:
        where.append('j."completedAt" >= %s')
        params.append(since)
    if axis == "interior":
        where.append('ir."isInteriorWorker" = true')
    elif axis == "exterior":
        where.append('ir."isInteriorWorker" = false')
    # Full-res sources only: ours from Dropbox (outputR2Path is the
    # Dropbox path of the upright JPG), vendor + matched from R2.
    # Thumbnails are for room-classifier review work, not image-quality
    # analysis — different precision needs.
    where.append('ir."outputR2Path" IS NOT NULL')
    where.append('ir."vendorImageR2Path" IS NOT NULL')
    sql = f"""
        SELECT ir.id, ir."jobId", ir."midStem",
               ir."outputR2Path",
               ir."vendorImageR2Path",
               ir."matchedRgbR2Path",
               ir."matchedReinhardR2Path",
               ir."isInteriorWorker",
               ir."roomTypeWorker",
               j.photographer
        FROM "ImageReview" ir
        JOIN "Job" j ON ir."jobId" = j.id
        WHERE {' AND '.join(where)}
        ORDER BY random()
        LIMIT %s
    """
    params.append(limit)
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--since", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--axis", choices=["all", "interior", "exterior"], default="all",
                    help="filter ImageReview rows by isInteriorWorker")
    ap.add_argument("--no-lpips", action="store_true",
                    help="skip LPIPS (only histogram stats)")
    ap.add_argument("--lpips-size", type=int, default=512,
                    help="resize longest-edge to this for LPIPS (default 512)")
    args = ap.parse_args()

    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 2

    rows = fetch_rows(args.limit, args.since, args.axis)
    print(f"sampled {len(rows)} matched ImageReview rows (axis={args.axis})")
    if not rows:
        return 0

    scratch = Path(tempfile.mkdtemp(prefix="gap-analysis-"))
    try:
        # Per-row stats. Histogram delta vector per config; scalar LPIPS per config.
        deltas: dict[str, list[np.ndarray]] = defaultdict(list)
        lpips_scores: dict[str, list[float]] = defaultdict(list)
        csv_rows: list[dict] = []
        n_ok = 0

        if not args.no_lpips:
            print("loading LPIPS (VGG backbone)...", flush=True)
            _get_lpips()
            print("  ready", flush=True)

        for i, r in enumerate(rows, 1):
            # Full-res sources. ours lives in Dropbox; everything else in R2.
            local = {
                "ours":     scratch / f"{r['id']}_ours.jpg",
                "vendor":   scratch / f"{r['id']}_vendor.jpg",
                "rgb":      scratch / f"{r['id']}_rgb.jpg",
                "reinhard": scratch / f"{r['id']}_reinhard.jpg",
            }
            ok = (
                rclone_get_dropbox(r["outputR2Path"], local["ours"])
                and rclone_get_r2(r["vendorImageR2Path"], local["vendor"])
                and rclone_get_r2(r["matchedRgbR2Path"], local["rgb"])
                and rclone_get_r2(r["matchedReinhardR2Path"], local["reinhard"])
            )
            if not ok:
                print(f"  [{i}/{len(rows)}] {r['midStem']}: skipping (missing file)", flush=True)
                for k in local.values():
                    try: k.unlink()
                    except FileNotFoundError: pass
                continue

            s = {k: stats(local[k], args.resize) for k in local}
            d_ours     = s["ours"]     - s["vendor"]
            d_rgb      = s["rgb"]      - s["vendor"]
            d_reinhard = s["reinhard"] - s["vendor"]
            deltas["ours"].append(d_ours)
            deltas["rgb"].append(d_rgb)
            deltas["reinhard"].append(d_reinhard)

            lp = {"ours": 0.0, "rgb": 0.0, "reinhard": 0.0}
            if not args.no_lpips:
                for cfg, k in (("ours", "ours"), ("rgb", "rgb"), ("reinhard", "reinhard")):
                    try:
                        lp[cfg] = lpips_distance(local[k], local["vendor"], size=args.lpips_size)
                    except Exception as e:
                        print(f"    LPIPS fail {r['midStem']}/{cfg}: {e}", flush=True)
                lpips_scores["ours"].append(lp["ours"])
                lpips_scores["rgb"].append(lp["rgb"])
                lpips_scores["reinhard"].append(lp["reinhard"])

            n_ok += 1
            if i % 50 == 0 or i == len(rows):
                print(f"  ...{i}/{len(rows)} done", flush=True)

            for cfg, d in (("ours", d_ours), ("rgb", d_rgb), ("reinhard", d_reinhard)):
                csv_rows.append({
                    "jobId": r["jobId"], "midStem": r["midStem"],
                    "config": cfg,
                    "isInterior": r["isInteriorWorker"],
                    "roomType": r["roomTypeWorker"] or "",
                    "photographer": r["photographer"] or "",
                    "l2_to_vendor": float(np.linalg.norm(d)),
                    "lpips_to_vendor": lp[cfg] if not args.no_lpips else None,
                    **{f"delta_{n}": float(v) for n, v in zip(METRIC_NAMES, d)},
                })

            for k in local.values():
                try: k.unlink()
                except FileNotFoundError: pass

        if n_ok == 0:
            print("no rows successfully analyzed")
            return 1

        print(f"\nAnalyzed {n_ok}/{len(rows)} pairs.\n")

        # Aggregate: mean absolute delta per metric per config, and the
        # L2 distance to vendor as a single composite "gap" number.
        print(f"{'metric':<14}  {'ours':>10}  {'rgb':>10}  {'reinhard':>10}   (mean |Δ| vs vendor — lower is better)")
        print("-" * 70)
        for i, name in enumerate(METRIC_NAMES):
            ours_d = np.mean([abs(d[i]) for d in deltas["ours"]])
            rgb_d  = np.mean([abs(d[i]) for d in deltas["rgb"]])
            rein_d = np.mean([abs(d[i]) for d in deltas["reinhard"]])
            print(f"{name:<14}  {ours_d:>10.4f}  {rgb_d:>10.4f}  {rein_d:>10.4f}")
        print("-" * 70)

        # Composite L2 distance per pair (treats all 8 metrics with equal
        # weight). Mean across pairs.
        print("\nHistogram L2 distance to vendor (lower = better):")
        for cfg in ("ours", "rgb", "reinhard"):
            l2s = [np.linalg.norm(d) for d in deltas[cfg]]
            print(f"  {cfg:>10}: mean {np.mean(l2s):.4f}   median {np.median(l2s):.4f}")

        if not args.no_lpips and lpips_scores["ours"]:
            print("\nLPIPS perceptual distance to vendor (lower = better):")
            for cfg in ("ours", "rgb", "reinhard"):
                ls = lpips_scores[cfg]
                print(f"  {cfg:>10}: mean {np.mean(ls):.4f}   median {np.median(ls):.4f}")

        # Win rate: per pair, which config has the smallest L2 to vendor?
        wins = {"ours": 0, "rgb": 0, "reinhard": 0}
        for i in range(n_ok):
            scores = {cfg: np.linalg.norm(deltas[cfg][i]) for cfg in wins}
            wins[min(scores, key=lambda k: scores[k])] += 1
        print("\nPer-pair winner by histogram L2:")
        for cfg, n in wins.items():
            pct = 100.0 * n / n_ok
            print(f"  {cfg:>10}: {n:4d}  ({pct:5.1f}%)")

        if not args.no_lpips and lpips_scores["ours"]:
            wins_lp = {"ours": 0, "rgb": 0, "reinhard": 0}
            for i in range(n_ok):
                scores = {cfg: lpips_scores[cfg][i] for cfg in wins_lp}
                wins_lp[min(scores, key=lambda k: scores[k])] += 1
            print("\nPer-pair winner by LPIPS:")
            for cfg, n in wins_lp.items():
                pct = 100.0 * n / n_ok
                print(f"  {cfg:>10}: {n:4d}  ({pct:5.1f}%)")

        # Interior vs exterior breakdown
        int_idx = [i for i, r in enumerate(rows[:n_ok]) if r["isInteriorWorker"]]
        ext_idx = [i for i, r in enumerate(rows[:n_ok]) if r["isInteriorWorker"] is False]
        if int_idx and ext_idx:
            print("\nMean L2 by interior/exterior:")
            print(f"  {'config':<10}  {'interior':>10}  {'exterior':>10}")
            for cfg in ("ours", "rgb", "reinhard"):
                ints = np.mean([np.linalg.norm(deltas[cfg][i]) for i in int_idx])
                exts = np.mean([np.linalg.norm(deltas[cfg][i]) for i in ext_idx])
                print(f"  {cfg:<10}  {ints:>10.4f}  {exts:>10.4f}")

        if args.csv:
            with open(args.csv, "w") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\nwrote {len(csv_rows)} rows → {args.csv}")
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
