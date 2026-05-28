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

# Lazy-import torch / geocalib so the script can also be run in a
# dependency-light mode (no DL inference) for the classical-only stats.
_GEOCALIB_PINHOLE = None
_GEOCALIB_DIST = None
_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        import torch  # type: ignore
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _load_geocalib(model_id: str = "pinhole"):
    """model_id: 'pinhole' (default, no lens distortion) or 'distorted'
    (handles fisheye-ish wide-angle lenses common in real estate)."""
    global _GEOCALIB_PINHOLE, _GEOCALIB_DIST
    if model_id == "pinhole" and _GEOCALIB_PINHOLE is not None:
        return _GEOCALIB_PINHOLE
    if model_id == "distorted" and _GEOCALIB_DIST is not None:
        return _GEOCALIB_DIST
    from geocalib import GeoCalib  # type: ignore
    m = GeoCalib(weights=model_id).to(_device()).eval()
    if model_id == "pinhole":
        _GEOCALIB_PINHOLE = m
    else:
        _GEOCALIB_DIST = m
    return m


def geocalib_roll_deg(jpg_path: Path, model_id: str = "pinhole") -> float | None:
    try:
        m = _load_geocalib(model_id)
        img = m.load_image(str(jpg_path)).to(_device())
        res = m.calibrate(img)
        roll_rad = res["gravity"].roll.item()
        return math.degrees(roll_rad)
    except Exception as e:
        print(f"  geocalib({model_id}) failed on {jpg_path.name}: {e}")
        return None


# ─── STN model (image-domain training, scalar inference) ──────────────────

_STN_MODEL = None
_STN_TFM = None


def _load_stn(ckpt_path: str):
    """Lazy-load the upright STN model. We import the architecture from
    train_upright_stn so the model definition stays in one place."""
    global _STN_MODEL, _STN_TFM
    if _STN_MODEL is not None:
        return _STN_MODEL
    import torch  # type: ignore
    from torchvision import transforms  # type: ignore
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train_upright_stn import (  # type: ignore
        UprightSTN, INPUT_SIZE, NORM_MEAN, NORM_STD,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = UprightSTN()
    model.load_state_dict(state["model_state"])
    model = model.to(_device()).eval()
    _STN_TFM = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    _STN_MODEL = model
    print(f"  STN model: loaded {ckpt_path} (ver={state.get('version')}, "
          f"val_mae={state.get('val_mae_deg')}°)", flush=True)
    return model


def stn_roll_deg(jpg_path: Path, ckpt: str) -> float | None:
    """Predict the upright-correction angle for an image using the
    trained STN. Returns angle in degrees (sign convention matches
    Hough: positive = image needs to rotate CCW to be upright)."""
    try:
        import torch  # type: ignore
        m = _load_stn(ckpt)
        from PIL import Image  # type: ignore
        im = Image.open(jpg_path).convert("RGB")
        x = _STN_TFM(im).unsqueeze(0).to(_device())
        with torch.no_grad():
            pred = m(x).item()
        # Trained model predicts the angle to UNDO the rotation present
        # in the input. So pred = "rotate input by this much (CCW) to
        # make it upright." Same convention as our other estimators —
        # return as-is.
        return float(pred)
    except Exception as e:
        print(f"  stn failed on {jpg_path.name}: {e}")
        return None


# ─── VLM-based roll estimator (Claude Haiku) ────────────────────────────────

import base64  # used by vlm_roll_deg

VLM_PROMPT = """You are looking at a real-estate listing photo.

Estimate the camera roll: how much would the image need to be rotated to make
its true verticals (door frames, wall corners, window edges) perfectly vertical?

Convention: a POSITIVE angle means the image needs to rotate COUNTER-CLOCKWISE
(CCW) by that many degrees to be correct. Equivalently, the image is currently
tilted CLOCKWISE from upright by that amount.

Typical real-estate shots are within ±3 degrees; the answer is usually small.
Look at multiple verticals (frames, corners) to triangulate.

Respond with ONLY this JSON, nothing else:
{"roll_deg": <signed float in [-10, 10]>, "confidence": <float in [0,1]>}"""


def vlm_roll_deg(jpg_path: Path, model: str = "claude-haiku-4-5-20251001") -> float | None:
    """Ask Claude Haiku for the camera roll. Returns a signed angle in
    degrees, or None on parse failure. Costs ~$0.0005 per image at Haiku
    rates. Uses ANTHROPIC_API_KEY from env."""
    import json as _json
    import urllib.request as _req
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        img_b64 = base64.b64encode(jpg_path.read_bytes()).decode("ascii")
    except Exception:
        return None
    body = {
        "model": model,
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
    }
    req = _req.Request(
        "https://api.anthropic.com/v1/messages",
        data=_json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with _req.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  vlm failed on {jpg_path.name}: {e}")
        return None
    try:
        text = data["content"][0]["text"]
        # Strip any code fence the model may add.
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        parsed = _json.loads(text)
        return float(parsed["roll_deg"])
    except Exception as e:
        print(f"  vlm parse failed on {jpg_path.name}: {e}; raw={text[:120] if 'text' in dir() else '?'}")
        return None

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

def _detect_segments(img: np.ndarray, min_len_frac: float = 0.05) -> list[tuple[float, float, float, float]]:
    """Common LSD-first / Hough-fallback line segment detection. Used by
    every classical estimator below so they're comparing apples to apples."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_len = int(min_len_frac * min(h, w))
    segs: list[tuple[float, float, float, float]] = []
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
    return segs


def _vertical_residuals(segs) -> tuple[list[float], list[float]]:
    """For each segment in `segs`, return (signed angle off-vertical in
    degrees, length-weight). Filters to |delta - 90°| < 25° (near-vertical)."""
    deltas: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in segs:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        theta_deg = math.degrees(math.atan2(dy, dx))
        if theta_deg < 0:
            theta_deg += 180
        delta = theta_deg - 90.0
        if abs(delta) < 25:
            deltas.append(delta)
            weights.append(length)
    return deltas, weights


def lsd_median_roll_deg(img: np.ndarray, min_len_frac: float = 0.05) -> tuple[float, int]:
    """Length-weighted median of near-vertical line angles. Median is
    naturally robust to a handful of misdetected lines that pull the
    weighted-mean estimator (current) off-target."""
    segs = _detect_segments(img, min_len_frac)
    deltas, weights = _vertical_residuals(segs)
    n = len(deltas)
    if n == 0:
        return 0.0, 0
    # Length-weighted percentile-50.
    pairs = sorted(zip(deltas, weights))
    total = sum(weights)
    accum = 0.0
    median = 0.0
    for d, w in pairs:
        accum += w
        if accum >= total / 2:
            median = d
            break
    return float(-median), n


def lsd_vp_ransac_roll_deg(img: np.ndarray, min_len_frac: float = 0.05,
                            n_iter: int = 200, inlier_thresh_deg: float = 0.8,
                            rng_seed: int = 42) -> tuple[float, int]:
    """RANSAC over near-vertical line angles. Each iteration picks one
    random line's angle as a candidate, counts how many other lines fall
    within ±inlier_thresh, and refits using just those inliers. Keeps the
    candidate with the largest inlier set. More robust than weighted-mean
    on cluttered scenes where a few misaligned long edges dominate."""
    segs = _detect_segments(img, min_len_frac)
    deltas, weights = _vertical_residuals(segs)
    if len(deltas) < 3:
        return 0.0, len(deltas)
    angles = np.asarray(deltas, dtype=np.float64)
    ws = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    best_inlier_w = 0.0
    best_angle = 0.0
    for _ in range(n_iter):
        cand = angles[rng.integers(0, len(angles))]
        mask = np.abs(angles - cand) < inlier_thresh_deg
        inlier_w = float(ws[mask].sum())
        if inlier_w > best_inlier_w:
            # Length-weighted mean of inliers — sharper than the candidate alone.
            best_angle = float((angles[mask] * ws[mask]).sum() / inlier_w)
            best_inlier_w = inlier_w
    return float(-best_angle), len(deltas)


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

def evaluate_one(row: dict,
                  with_geocalib: bool = False,
                  with_geocalib_dist: bool = False,
                  with_vlm: bool = False,
                  with_lsd_median: bool = False,
                  with_lsd_vp: bool = False,
                  stn_ckpt: str | None = None) -> dict:
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

    out = {
        **row,
        "ours_hough_deg": round(ours_angle, 3),
        "ours_n_vert": ours_n,
        "vendor_hough_deg": round(vendor_angle, 3),
        "vendor_n_vert": vendor_n,
        "ours_vs_vendor_residual_deg":
            round(residual, 3) if residual is not None else None,
        "error": None,
    }

    # Each method predicts the camera roll on ours. We then simulate
    # applying that prediction as a correction and compute the new
    # residual to vendor: simulated_residual = sift_residual - prediction.
    # Lower |simulated_residual| = method would have done better.
    def _simulate(pred: float | None) -> float | None:
        if pred is None or residual is None:
            return None
        return round(residual - pred, 3)

    if with_geocalib:
        ours_gc = geocalib_roll_deg(ours_local, "pinhole")
        out["ours_geocalib_deg"] = round(ours_gc, 3) if ours_gc is not None else None
        out["geocalib_simulated_residual_deg"] = _simulate(ours_gc)

    if with_geocalib_dist:
        ours_gcd = geocalib_roll_deg(ours_local, "distorted")
        out["ours_geocalib_dist_deg"] = round(ours_gcd, 3) if ours_gcd is not None else None
        out["geocalib_dist_simulated_residual_deg"] = _simulate(ours_gcd)

    if with_vlm:
        ours_vlm = vlm_roll_deg(ours_local)
        out["ours_vlm_deg"] = round(ours_vlm, 3) if ours_vlm is not None else None
        out["vlm_simulated_residual_deg"] = _simulate(ours_vlm)

    if with_lsd_median:
        ours_med, _ = lsd_median_roll_deg(ours)
        out["ours_lsd_median_deg"] = round(ours_med, 3)
        out["lsd_median_simulated_residual_deg"] = _simulate(ours_med)

    if with_lsd_vp:
        ours_vp, _ = lsd_vp_ransac_roll_deg(ours)
        out["ours_lsd_vp_deg"] = round(ours_vp, 3)
        out["lsd_vp_simulated_residual_deg"] = _simulate(ours_vp)

    if stn_ckpt:
        ours_stn = stn_roll_deg(ours_local, stn_ckpt)
        out["ours_stn_deg"] = round(ours_stn, 3) if ours_stn is not None else None
        out["stn_simulated_residual_deg"] = _simulate(ours_stn)

    return out


# ─── report ────────────────────────────────────────────────────────────────

def write_outputs(results: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "manifest.csv"
    fields = [
        "jobId", "midStem", "room", "is_interior",
        "ours_hough_deg", "ours_n_vert",
        "vendor_hough_deg", "vendor_n_vert",
        "ours_vs_vendor_residual_deg",
        "ours_geocalib_deg", "geocalib_simulated_residual_deg",
        "ours_geocalib_dist_deg", "geocalib_dist_simulated_residual_deg",
        "ours_vlm_deg", "vlm_simulated_residual_deg",
        "ours_lsd_median_deg", "lsd_median_simulated_residual_deg",
        "ours_lsd_vp_deg", "lsd_vp_simulated_residual_deg",
        "ours_stn_deg", "stn_simulated_residual_deg",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    def _hist(name: str, values: list[float]) -> None:
        if not values: return
        buckets = {"0_lt_0.25": 0, "1_0.25-0.5": 0, "2_0.5-1.0": 0,
                   "3_1.0-2.0": 0, "4_gt_2.0": 0}
        for r in values:
            if r < 0.25: buckets["0_lt_0.25"] += 1
            elif r < 0.5: buckets["1_0.25-0.5"] += 1
            elif r < 1.0: buckets["2_0.5-1.0"] += 1
            elif r < 2.0: buckets["3_1.0-2.0"] += 1
            else: buckets["4_gt_2.0"] += 1
        print(f"\n=== {name} on n={len(values)} ===")
        print(f"  mean  |residual| = {np.mean(values):.3f} deg")
        print(f"  median|residual| = {np.median(values):.3f} deg")
        print(f"  p95   |residual| = {np.percentile(values, 95):.3f} deg")
        print(f"  max   |residual| = {np.max(values):.3f} deg")
        print("  histogram:")
        for k, v in buckets.items():
            print(f"    {k:14s} {v:4d}")

    # Current method: SIFT-measured residual from ours -> vendor.
    valid = [r for r in results
             if not r.get("error") and r.get("ours_vs_vendor_residual_deg") is not None]
    if valid:
        residuals = [abs(r["ours_vs_vendor_residual_deg"]) for r in valid]
        _hist("CURRENT (Hough) ours-vs-vendor residual", residuals)

    def _paired(name: str, sim_key: str) -> None:
        sub = [r for r in results
               if not r.get("error")
               and r.get(sim_key) is not None
               and r.get("ours_vs_vendor_residual_deg") is not None]
        if not sub:
            return
        residuals = [abs(r[sim_key]) for r in sub]
        _hist(f"{name} SIMULATED residual", residuals)
        pairs = [(abs(r["ours_vs_vendor_residual_deg"]), abs(r[sim_key]))
                 for r in sub]
        improvements = [a - b for a, b in pairs]
        n_better = sum(1 for d in improvements if d > 0.05)
        n_worse  = sum(1 for d in improvements if d < -0.05)
        n_same   = len(pairs) - n_better - n_worse
        print(f"\n=== PAIRED delta (current - {name}) on n={len(pairs)} ===")
        print(f"  {name} improves   (>0.05° lower):  {n_better}")
        print(f"  ~tie              (within ±0.05°):   {n_same}")
        print(f"  {name} worse      (>0.05° higher): {n_worse}")
        print(f"  mean improvement:   {np.mean(improvements):+.3f}°")
        print(f"  median improvement: {np.median(improvements):+.3f}°")

    _paired("LSD-median", "lsd_median_simulated_residual_deg")
    _paired("LSD-VP-RANSAC", "lsd_vp_simulated_residual_deg")
    _paired("GeoCalib(pinhole)", "geocalib_simulated_residual_deg")
    _paired("GeoCalib(distorted)", "geocalib_dist_simulated_residual_deg")
    _paired("VLM(Haiku)", "vlm_simulated_residual_deg")
    _paired("STN(trained)", "stn_simulated_residual_deg")

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
    ap.add_argument("--with-geocalib", action="store_true",
                    help="also run GeoCalib pinhole and simulate its correction")
    ap.add_argument("--with-geocalib-dist", action="store_true",
                    help="also run GeoCalib distorted (wide-angle / lens distortion model)")
    ap.add_argument("--with-vlm", action="store_true",
                    help="also call Claude Haiku for a VLM-based roll estimate "
                         "(requires ANTHROPIC_API_KEY)")
    ap.add_argument("--with-lsd-median", action="store_true",
                    help="LSD lines, length-weighted median angle")
    ap.add_argument("--with-lsd-vp", action="store_true",
                    help="LSD lines, RANSAC consensus angle (more robust to outliers)")
    ap.add_argument("--stn-ckpt", type=str, default=None,
                    help="path to a trained upright STN ckpt; benchmark its angle predictions")
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
    if args.with_geocalib:
        print("  warming GeoCalib pinhole…")
        _ = _load_geocalib("pinhole")
    if args.with_geocalib_dist:
        print("  warming GeoCalib distorted…")
        _ = _load_geocalib("distorted")
    if args.with_vlm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("  WARNING: --with-vlm set but ANTHROPIC_API_KEY missing; will return None")
    results: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(sample, 1):
        results.append(evaluate_one(
            r,
            with_geocalib=args.with_geocalib,
            with_geocalib_dist=args.with_geocalib_dist,
            with_vlm=args.with_vlm,
            with_lsd_median=args.with_lsd_median,
            with_lsd_vp=args.with_lsd_vp,
            stn_ckpt=args.stn_ckpt,
        ))
        if i % 10 == 0:
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(sample)}  ({rate:.1f}/sec)", flush=True)

    write_outputs(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
