"""Classical auto-upright for AREM Stage 2 outputs.

Reference behavior: Lightroom "Auto" (rotation + mild perspective), NOT "Full".

Algorithm:
  1. Detect line segments (HoughLinesP on Canny edges, fallback to LSD if available).
  2. Filter segments by length (≥ 5% of min image dim) and angle range.
  3. Bin into near-vertical (|θ-90°| < 25°) and near-horizontal (|θ| < 25°).
  4. For the dominant near-vertical direction, compute weighted-mean angle
     using line length as weight.
  5. Rotate around image center to make verticals vertical.
  6. Optional perspective correction: if vertical lines have residual horizontal
     vanishing-point skew, apply a mild vertical-perspective homography (capped
     so we don't over-correct).
  7. Crop to interior rectangle (largest axis-aligned rect inside the rotated/
     warped canvas — simple inscribed-rect via geometric formula for rotation,
     numerical search for full warp).

Output JPG carries EXIF + sRGB ICC profile (per project_post_stage2_pipeline_plan.md):
  - Copyright + Artist + Software (American Real Estate Media branding)
  - DateTimeOriginal, Make, Model, LensModel, FocalLength, FNumber, ISO,
    ExposureTime — preserved from mid-frame ARW
  - sRGB ICC profile embedded
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import exifread
import numpy as np
import piexif
from PIL import Image, ImageCms

# ---- branding constants (per Jordan, locked 2026-05-02) ----
ARTIST = "Jordan DiCaprio - American Real Estate Media"
SOFTWARE = "American Real Estate Media Photo Processor"
COPYRIGHT_TEMPLATE = ("\u00a9 {year} American Real Estate Media. "
                       "https://AmericanRealEstateMedia.com - All Rights Reserved.")

# Default ARW root used to look up mid-frame source for EXIF preservation.
# Override at runtime via the AREM_ARW_ROOT env var (used by the production
# worker which downloads ARWs to a per-job temp dir).
ARW_ROOT = Path(os.environ.get("AREM_ARW_ROOT",
                                "/home/jordan/Desktop/raw-files"))
# Optional: if AREM_ARW_FLAT_LAYOUT=1, look up ARWs as ARW_ROOT/<stem>.ARW
# instead of ARW_ROOT/<shoot>/01-RAW-Photos/<stem>.ARW.
ARW_FLAT = os.environ.get("AREM_ARW_FLAT_LAYOUT", "") in ("1", "true", "yes")

# sRGB ICC profile bytes (built once from PIL).
_SRGB_PROFILE_BYTES: bytes | None = None


def _srgb_profile_bytes() -> bytes:
    global _SRGB_PROFILE_BYTES
    if _SRGB_PROFILE_BYTES is None:
        prof = ImageCms.createProfile("sRGB")
        buf = io.BytesIO()
        ImageCms.ImageCmsProfile(prof).tobytes()  # ensure built
        _SRGB_PROFILE_BYTES = ImageCms.ImageCmsProfile(prof).tobytes()
    return _SRGB_PROFILE_BYTES


# ---- line detection ----

def detect_segments(gray: np.ndarray, min_len_frac: float = 0.05) -> np.ndarray:
    """Return Nx4 array of [x1, y1, x2, y2] line segments."""
    h, w = gray.shape
    min_len = int(min_len_frac * min(h, w))

    # LSD if available (faster, more accurate). If not, Canny + HoughLinesP.
    try:
        lsd = cv2.createLineSegmentDetector(0)
        lines, *_ = lsd.detect(gray)
        if lines is None:
            lines = np.empty((0, 1, 4), dtype=np.float32)
        segs = lines.reshape(-1, 4)
    except Exception:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                                 minLineLength=min_len, maxLineGap=10)
        if lines is None:
            return np.empty((0, 4), dtype=np.float32)
        segs = lines.reshape(-1, 4).astype(np.float32)

    # length filter
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    length = np.sqrt(dx * dx + dy * dy)
    return segs[length >= min_len]


def angle_deg(seg: np.ndarray) -> np.ndarray:
    """Per-segment angle in degrees ∈ [-90, 90]. Vertical lines → ±90."""
    dx = seg[:, 2] - seg[:, 0]
    dy = seg[:, 3] - seg[:, 1]
    a = np.degrees(np.arctan2(dy, dx))
    # fold to [-90, 90]
    a = np.where(a > 90, a - 180, a)
    a = np.where(a < -90, a + 180, a)
    return a


def length(seg: np.ndarray) -> np.ndarray:
    return np.sqrt((seg[:, 2] - seg[:, 0]) ** 2 + (seg[:, 3] - seg[:, 1]) ** 2)


def estimate_rotation(segs: np.ndarray, vert_window: float = 25.0) -> tuple[float, int, float]:
    """Return (rotation_deg, n_vert_used, residual_std).

    rotation_deg is what we should rotate the image (counter-clockwise positive)
    to make the dominant near-vertical direction true vertical.
    """
    if len(segs) == 0:
        return 0.0, 0, 0.0
    a = angle_deg(segs)
    L = length(segs)
    # Vertical segments: angles near ±90. Map to a "tilt from vertical" signed value.
    # Tilt = 90 - |a| with sign of a (so a=88 → tilt=+2, a=-87 → tilt=-3, etc.)
    is_vert = np.abs(np.abs(a) - 90) <= vert_window
    if not is_vert.any():
        return 0.0, 0, 0.0
    a_v = a[is_vert]
    L_v = L[is_vert]
    # Convert to "deviation from true vertical": positive means tilted clockwise.
    # If a=+89 → tilt = -1 (need to rotate +1° CCW).
    # If a=-89 → tilt = +1 (need to rotate -1° CCW).
    # So tilt_deg = 90 - |a| with sign opposite to sign of a.
    tilt = -np.sign(a_v) * (90 - np.abs(a_v))
    # weighted median (more robust than mean)
    order = np.argsort(tilt)
    tilt_sorted = tilt[order]
    L_sorted = L_v[order]
    cum = np.cumsum(L_sorted)
    half = cum[-1] / 2.0
    med = float(tilt_sorted[np.searchsorted(cum, half)])
    residual = float(np.std(tilt_sorted))
    return med, int(is_vert.sum()), residual


def rotate_and_crop(img: np.ndarray, angle_deg: float) -> tuple[np.ndarray, float]:
    """Rotate around center, then crop to largest axis-aligned interior rect.

    Returns (cropped_img, area_ratio). area_ratio = cropped_pixels / original.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)
    # Largest interior rect for a centered rotation by θ of an HxW image
    # (formula from Andrew M. Day, valid for |θ| < 90°):
    a = abs(math.radians(angle_deg))
    if a < 1e-6:
        return rot, 1.0
    side_long = max(w, h)
    side_short = min(w, h)
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    if side_short <= 2 * sin_a * cos_a * side_long:
        x = 0.5 * side_short
        wc = x / sin_a
        hc = x / cos_a
    else:
        denom = cos_a * cos_a - sin_a * sin_a
        wc = (w * cos_a - h * sin_a) / denom
        hc = (h * cos_a - w * sin_a) / denom
    if w < h:
        wc, hc = hc, wc
    wc = max(int(wc), 1)
    hc = max(int(hc), 1)
    cx, cy = w // 2, h // 2
    x0 = max(cx - wc // 2, 0)
    y0 = max(cy - hc // 2, 0)
    cropped = rot[y0:y0 + hc, x0:x0 + wc]
    return cropped, (cropped.shape[0] * cropped.shape[1]) / (h * w)


# ---- EXIF helpers ----

def _ratio_to_float(r) -> float | None:
    try:
        return r.num / r.den
    except (ZeroDivisionError, AttributeError):
        try:
            return float(r)
        except Exception:
            return None


def read_arw_exif(arw_path: Path) -> dict:
    """Pull the fields we want to preserve from a Sony ARW. Best effort."""
    out = {}
    if not arw_path.is_file():
        return out
    try:
        with arw_path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        return out

    def _get(key: str) -> str | None:
        v = tags.get(key)
        return str(v) if v is not None else None

    def _get_ratio(key: str) -> tuple[int, int] | None:
        v = tags.get(key)
        if v is None or not v.values:
            return None
        r = v.values[0]
        try:
            return int(r.num), int(r.den)
        except AttributeError:
            try:
                return int(float(r) * 100), 100
            except Exception:
                return None

    out["DateTimeOriginal"] = _get("EXIF DateTimeOriginal") or _get("Image DateTime")
    out["Make"] = _get("Image Make")
    out["Model"] = _get("Image Model")
    out["LensModel"] = _get("EXIF LensModel")
    out["FNumber"] = _get_ratio("EXIF FNumber")
    out["ExposureTime"] = _get_ratio("EXIF ExposureTime")
    out["FocalLength"] = _get_ratio("EXIF FocalLength")
    out["FocalLengthIn35mmFilm"] = _get("EXIF FocalLengthIn35mmFilm")
    iso = tags.get("EXIF ISOSpeedRatings")
    out["ISO"] = int(iso.values[0]) if iso and iso.values else None
    return out


def _to_bytes(s: str) -> bytes:
    return s.encode("utf-8", errors="ignore")


def build_exif_dict(arw_meta: dict, year: int) -> bytes:
    """Construct an EXIF byte blob with our branding + preserved camera info."""
    zeroth = {
        piexif.ImageIFD.Make:           _to_bytes(arw_meta.get("Make") or "Sony"),
        piexif.ImageIFD.Model:          _to_bytes(arw_meta.get("Model") or ""),
        piexif.ImageIFD.Software:       _to_bytes(SOFTWARE),
        piexif.ImageIFD.Artist:         _to_bytes(ARTIST),
        piexif.ImageIFD.Copyright:      _to_bytes(COPYRIGHT_TEMPLATE.format(year=year)),
        piexif.ImageIFD.XResolution:    (300, 1),
        piexif.ImageIFD.YResolution:    (300, 1),
        piexif.ImageIFD.ResolutionUnit: 2,  # inches
    }
    dt = arw_meta.get("DateTimeOriginal")
    if dt:
        zeroth[piexif.ImageIFD.DateTime] = _to_bytes(dt)

    exif = {}
    if dt:
        exif[piexif.ExifIFD.DateTimeOriginal] = _to_bytes(dt)
        exif[piexif.ExifIFD.DateTimeDigitized] = _to_bytes(dt)
    if arw_meta.get("FNumber"):
        exif[piexif.ExifIFD.FNumber] = arw_meta["FNumber"]
    if arw_meta.get("ExposureTime"):
        exif[piexif.ExifIFD.ExposureTime] = arw_meta["ExposureTime"]
    if arw_meta.get("FocalLength"):
        exif[piexif.ExifIFD.FocalLength] = arw_meta["FocalLength"]
    if arw_meta.get("FocalLengthIn35mmFilm"):
        try:
            exif[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(arw_meta["FocalLengthIn35mmFilm"])
        except Exception:
            pass
    if arw_meta.get("ISO"):
        exif[piexif.ExifIFD.ISOSpeedRatings] = arw_meta["ISO"]
    if arw_meta.get("LensModel"):
        exif[piexif.ExifIFD.LensModel] = _to_bytes(arw_meta["LensModel"])
    exif[piexif.ExifIFD.ColorSpace] = 1  # sRGB

    return piexif.dump({"0th": zeroth, "Exif": exif, "GPS": {},
                         "Interop": {}, "1st": {}, "thumbnail": None})


def upright_one(img_path: Path, out_path: Path,
                max_rotation_deg: float = 2.0,
                min_vert_segs: int = 8,
                max_residual_std_deg: float = 1.5,
                arw_root: Path = ARW_ROOT) -> dict:
    """Process one image: detect, rotate, crop. Returns metrics dict.

    Rotation is applied only when ALL of:
      |estimated| <= max_rotation_deg  (real-estate handheld is rarely >2°)
      n_vert     >= min_vert_segs      (enough verticals to trust the estimate)
      residual   <= max_residual_std_deg  (verticals agree with each other)

    Any failure → applied=0 (image kept straight) with `clamp_reason` set.
    """
    t0 = time.time()
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"path": str(img_path), "error": "imread_failed"}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    segs = detect_segments(gray)
    rot_deg, n_vert, residual = estimate_rotation(segs)

    clamp_reason: str | None = None
    if abs(rot_deg) > max_rotation_deg:
        clamp_reason = "magnitude"
    elif n_vert < min_vert_segs:
        clamp_reason = "low_confidence_n_vert"
    elif residual > max_residual_std_deg:
        clamp_reason = "low_confidence_residual"

    if clamp_reason is not None:
        applied = 0.0
        clamped = True
    else:
        applied = rot_deg
        clamped = False

    out_img, area_ratio = rotate_and_crop(img, applied)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

    # Look up source file for EXIF passthrough. RAW preferred (any vendor),
    # but accept JPG/TIFF/JXL too — exifread reads EXIF identically across
    # formats and shoots may come in as standard images now.
    mid_stem = out_path.stem
    shoot = out_path.parent.name
    arw_path = None
    for ext in ("ARW", "arw", "NEF", "nef", "CR2", "cr2", "CR3", "cr3",
                "DNG", "dng", "RAF", "raf", "RW2", "rw2",
                "JPG", "jpg", "JPEG", "jpeg",
                "TIF", "tif", "TIFF", "tiff", "JXL", "jxl"):
        if ARW_FLAT:
            cand = arw_root / f"{mid_stem}.{ext}"
        else:
            cand = arw_root / shoot / "01-RAW-Photos" / f"{mid_stem}.{ext}"
        if cand.is_file():
            arw_path = cand
            break
    arw_meta = read_arw_exif(arw_path) if arw_path is not None else {}

    # Year for copyright: from DateTimeOriginal, fallback to current.
    year = datetime.now().year
    dt = arw_meta.get("DateTimeOriginal")
    if dt:
        try:
            year = int(dt.split(":")[0])
        except (IndexError, ValueError):
            pass

    exif_bytes = build_exif_dict(arw_meta, year)
    Image.fromarray(out_rgb).save(out_path, quality=92,
                                   exif=exif_bytes,
                                   icc_profile=_srgb_profile_bytes())

    return {
        "path": str(img_path.name),
        "size_in": [h, w],
        "size_out": list(out_img.shape[:2]),
        "rotation_estimated_deg": round(rot_deg, 3),
        "rotation_applied_deg": round(applied, 3),
        "rotation_clamped": clamped,
        "clamp_reason": clamp_reason,
        "n_vertical_segs": n_vert,
        "residual_std_deg": round(residual, 3),
        "crop_area_ratio": round(area_ratio, 4),
        "exif_source_arw": arw_path.name if arw_meta else None,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }


# ---- pool worker ----

def _mid_stem_from_input(name: str) -> tuple[str, str]:
    """Strip stage2 route suffix to get the mid-stem; return (mid_stem, route).

    Input naming follows old convention: <DSC####>_stage2-{int|ext}.jpg
    Going forward all post-Stage-2 outputs use bare <mid_stem>.jpg with the
    route info stored in metadata.
    """
    base = name.rsplit(".", 1)[0]
    for suf, route in (("_stage2-int", "int"), ("_stage2-ext", "ext")):
        if base.endswith(suf):
            return base[: -len(suf)], route
    return base, "unknown"


def process_shoot(shoot_dir: Path, out_root: Path) -> dict:
    out_dir = out_root / shoot_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    # Support both layouts: flat (shoot/*.jpg) and nested (shoot/stage2/*.jpg)
    candidate_dirs = [shoot_dir / "stage2", shoot_dir]
    src_dir = next((d for d in candidate_dirs if d.is_dir() and any(d.glob("*.jpg"))), shoot_dir)
    jpgs = sorted(src_dir.glob("*.jpg"))
    for jp in jpgs:
        mid_stem, route = _mid_stem_from_input(jp.name)
        out_path = out_dir / f"{mid_stem}.jpg"
        m = upright_one(jp, out_path)
        m["mid_stem"] = mid_stem
        m["stage2_route"] = route
        metrics.append(m)
    (out_dir / "_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"shoot": shoot_dir.name, "n": len(metrics)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("/home/jordan/upright_test/stage2_input"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/jordan/upright_test/auto_uprighted"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    shoots = sorted([d for d in args.in_dir.iterdir() if d.is_dir()])
    print(f"shoots: {len(shoots)}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_shoot, s, args.out_dir) for s in shoots]
        for f in as_completed(futures):
            r = f.result()
            print(f"  done {r['n']:3d}  {r['shoot']}", flush=True)

    dt = time.time() - t0
    print(f"DONE in {dt:.1f}s ({dt / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
