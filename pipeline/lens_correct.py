"""Lens-aware demosaic for the Stage 1 learned-replacement POC.

Reads an ARW file, demosaics via rawpy with the exact same settings used
to generate the existing Photomatix Interior2 targets, then applies
lensfun distortion / TCA / vignetting correction based on EXIF lens
metadata.

Returns the corrected image as an `np.ndarray` (H, W, 3) uint8 in sRGB.

Sony FE 12-24mm f/2.8 GM is excluded from this POC (no upstream lensfun
profile, only the f/4 G version exists). The function raises
LensExcluded for those frames; the caller decides what to do (skip the
pair, fall back to uncorrected, etc).

Usage:
    from prep.lens_correct import lens_correct_bracket, LensExcluded

    try:
        rgb = lens_correct_bracket(Path("DSC01173.ARW"))
    except LensExcluded as e:
        # excluded lens — skip this pair
        ...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rawpy

logger = logging.getLogger(__name__)

# File extensions that go through the rawpy + lensfun path. Anything else
# (JPG/JPEG/TIFF/PNG/JXL/WEBP) goes through decode_standard_image() — no
# demosaic, no lens correction, just decode + upscale to 16-bit so the
# rest of the pipeline (NAFNet 9-ch input, Restormer Stage 2) can treat
# it identically.
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2"}

# Lenses we explicitly exclude from the Stage 1 POC because they have no
# usable upstream lensfun profile. Match by the LensModel substring.
EXCLUDED_LENS_SUBSTRINGS = (
    "12-24mm F2.8 GM",     # Sony FE 12-24mm f/2.8 GM
    "12-24 F2.8 GM",
    "FE 12-24mm GM",
)

# Map "what EXIF says" → ("lensfun maker", "lensfun model substring").
# lensfun's Database.find_lenses takes a model search string; the second
# element of each tuple is matched against Lens.model with substring.
_LENS_HINTS = (
    ("9mm F2.8",                     ("Viltrox", "Viltrox AF 9mm")),
    ("AF 9mm",                       ("Viltrox", "Viltrox AF 9mm")),
    ("Viltrox AF 9",                 ("Viltrox", "Viltrox AF 9mm")),
    # Laowa 12mm Zero-D (full-frame original) -- exact profile.
    ("12mm F2.8 Zero-D",             ("Venus",   "Laowa 12mm")),
    ("Laowa 12mm",                   ("Venus",   "Laowa 12mm")),
    # Laowa FFII 12mm "C-Dreamer" — closely related optical formula to
    # the Zero-D; fall back to that profile in absence of a dedicated
    # FFII entry in lensfun.
    ("FFII 12mm F2.8",               ("Venus",   "Laowa 12mm")),
    ("FFII 12mm",                    ("Venus",   "Laowa 12mm")),
    ("LAOWA FFII 12",                ("Venus",   "Laowa 12mm")),
    ("14-30mm F4",                   ("Nikon",   "14-30mm")),
    ("Z 14-30",                      ("Nikon",   "14-30mm")),
)


class LensExcluded(Exception):
    """Raised when the lens model is on the POC exclusion list."""


def _read_lens_exif(arw_path: Path) -> dict:
    """Return EXIF lens fields. Best-effort; missing fields are empty strings."""
    import exifread
    with arw_path.open("rb") as f:
        tags = exifread.process_file(f, details=False)

    def s(key, default=""):
        v = tags.get(key)
        return str(v) if v is not None else default

    return {
        "make": s("EXIF LensMake") or s("Image Make"),
        "model": s("EXIF LensModel"),
        "focal": s("EXIF FocalLength"),
        "focal_35mm": s("EXIF FocalLengthIn35mmFilm"),
        "fnumber": s("EXIF FNumber"),
        "camera_make": s("Image Make"),
        "camera_model": s("Image Model"),
    }


def _resolve_lensfun(db, exif: dict):
    """Return (Camera, Lens) or (Camera, None). Both may be None on no-match.

    Strategy:
      1. Match camera by "<make> <model>" string.
      2. Iterate _LENS_HINTS; for the first whose key is found inside
         exif["model"] (case-insensitive), search lensfun for the
         matching maker/model substring. Pick the first match.
    """
    cam = None
    cam_make = exif.get("camera_make", "").strip()
    cam_model = exif.get("camera_model", "").strip()
    if cam_make and cam_model:
        try:
            cam_results = db.find_cameras(cam_make, cam_model, loose_search=True)
            if cam_results:
                cam = cam_results[0]
        except Exception:
            cam = None

    lens_model = (exif["model"] or "").upper()
    chosen = None
    for hint, (maker, model_sub) in _LENS_HINTS:
        if hint.upper() in lens_model:
            cands = [
                L for L in db.lenses
                if maker.lower() in (L.maker or "").lower()
                and model_sub.lower() in (L.model or "").lower()
            ]
            if cands:
                chosen = cands[0]
                break
    return cam, chosen


def lens_correct_bracket(
    arw_path: Path,
    *,
    out_bps: int = 8,
    use_camera_wb: bool = False,
    use_auto_wb: bool = True,
    no_auto_bright: bool = True,
    db=None,
) -> Tuple[np.ndarray, dict]:
    """Demosaic + lens-correct an ARW.

    Returns (rgb_uint8, info_dict). info_dict has:
        lens_model_exif: str         (raw EXIF LensModel)
        lensfun_maker:   Optional[str]
        lensfun_model:   Optional[str]
        applied:         bool        (True iff lensfun correction succeeded)
        skip_reason:     Optional[str]

    Raises LensExcluded if the lens is on the POC exclusion list.

    rawpy settings default to what the existing Photomatix prep pipeline
    used (`use_auto_wb=True`, `no_auto_bright=True`, sRGB) — this keeps
    the model's input distribution aligned with the targets we trained
    against.
    """
    arw_path = Path(arw_path)

    exif = _read_lens_exif(arw_path)
    lens_model_exif = exif.get("model", "")
    # Lenses on the exclusion list have no upstream lensfun profile — we
    # used to raise LensExcluded and skip the whole bracket. As of
    # 2026-05-04, instead bypass lens correction and continue: rawpy
    # demosaic still produces a usable image, just without distortion /
    # vignetting / TCA correction. Mark this in info["skip_reason"] so
    # the caller knows correction was bypassed.
    excluded_lens = any(
        s.upper() in lens_model_exif.upper() for s in EXCLUDED_LENS_SUBSTRINGS
    )

    with rawpy.imread(str(arw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,
            use_auto_wb=use_auto_wb,
            no_auto_bright=no_auto_bright,
            output_bps=16 if out_bps == 16 else 8,
            output_color=rawpy.ColorSpace.sRGB,
            user_flip=0,
        )
    if out_bps == 8 and rgb.dtype != np.uint8:
        rgb = (rgb >> 8).astype(np.uint8) if rgb.dtype == np.uint16 else rgb

    info = {
        "lens_model_exif": lens_model_exif,
        "lensfun_maker": None,
        "lensfun_model": None,
        "applied": False,
        "skip_reason": None,
    }

    # Excluded lens — bypass lensfun lookup entirely.
    if excluded_lens:
        info["skip_reason"] = (
            f"lens {lens_model_exif!r} on POC exclusion list — "
            f"continuing without lens correction"
        )
        return rgb, info

    if db is None:
        import lensfunpy
        db = lensfunpy.Database()

    try:
        cam, lens = _resolve_lensfun(db, exif)
    except Exception as e:
        info["skip_reason"] = f"lensfun resolve error: {e}"
        return rgb, info

    if lens is None or cam is None:
        info["skip_reason"] = (
            f"no lensfun match (camera={cam!r}, lens_exif={lens_model_exif!r})"
        )
        return rgb, info

    info["lensfun_maker"] = lens.maker
    info["lensfun_model"] = lens.model

    try:
        focal = float(exif["focal"]) if exif["focal"] else lens.min_focal
    except ValueError:
        focal = lens.min_focal
    try:
        fnum = float(exif["fnumber"]) if exif["fnumber"] else 8.0
    except ValueError:
        fnum = 8.0

    h, w = rgb.shape[:2]
    try:
        import lensfunpy
        mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
        mod.initialize(focal=focal, aperture=fnum, distance=1000)
        # Geometry undistort: build map and remap
        undist_coords = mod.apply_geometry_distortion()
        if undist_coords is None:
            info["skip_reason"] = "lensfun.apply_geometry_distortion returned None"
            return rgb, info
        import cv2
        rgb_corr = cv2.remap(
            rgb, undist_coords, None, cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        info["applied"] = True
        return rgb_corr, info
    except Exception as e:
        info["skip_reason"] = f"lensfun apply error: {type(e).__name__}: {e}"
        logger.warning("lens_correct: %s on %s", info["skip_reason"], arw_path.name)
        return rgb, info


def decode_standard_image(path: Path, *, out_bps: int = 16) -> Tuple[np.ndarray, dict]:
    """Decode a JPG/JPEG/TIFF/PNG/JXL/WEBP into a uint16 numpy array.

    The pipeline expects 16-bit RGB after the demosaic step. For 8-bit
    inputs we left-shift by 8 to map [0, 255] → [0, 65535] (i.e. ×257
    via simple bit replication, equivalent to "pad zeros" in the
    low-byte). This is the standard 8→16 bit promotion.
    """
    from PIL import Image

    path = Path(path)
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if arr.dtype == np.uint8:
        if out_bps == 16:
            arr = (arr.astype(np.uint16) << 8) | arr.astype(np.uint16)
    elif arr.dtype == np.uint16:
        if out_bps == 8:
            arr = (arr >> 8).astype(np.uint8)
    info = {
        "lens_model_exif": "",
        "lensfun_maker": None,
        "lensfun_model": None,
        "applied": False,
        "skip_reason": f"non-raw input ({path.suffix.lower()}); lens correction skipped",
        "input_format": path.suffix.lower(),
    }
    return arr, info


def load_bracket_frame(
    path: Path, *, out_bps: int = 16, db=None,
) -> Tuple[np.ndarray, dict]:
    """Format-agnostic loader: route to lens_correct_bracket() for raw
    files, decode_standard_image() for everything else.
    """
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        return lens_correct_bracket(path, out_bps=out_bps, db=db)
    return decode_standard_image(path, out_bps=out_bps)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Lens-correct one ARW (smoke test)")
    ap.add_argument("arw", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Save corrected JPG here")
    args = ap.parse_args()

    try:
        rgb, info = lens_correct_bracket(args.arw)
    except LensExcluded as e:
        print(f"EXCLUDED: {e}")
        sys.exit(2)

    print(json.dumps(info, indent=2))
    print(f"shape: {rgb.shape}, dtype: {rgb.dtype}")
    if args.out:
        from PIL import Image
        Image.fromarray(rgb).save(args.out, quality=95)
        print(f"wrote {args.out}")
