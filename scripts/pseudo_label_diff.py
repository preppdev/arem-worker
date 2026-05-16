"""Pseudo-label generation from before/after vendor diffs.

For every ImageReview row tagged with a given condition where we have
both matchedRgbR2Path (our output color-aligned to vendor) and
vendorImageR2Path (AutoHDR's edited output), compute the local content
difference between the two and treat it as a free pseudo-label for that
condition.

Because matchedRgbR2Path is already tone-aligned to the vendor, the
remaining differences should be (with high confidence) the local edits
the vendor made: reflections removed, fingers cloned out, dead fixtures
relit, etc. The reviewer's per-condition tagging disambiguates *which*
edit a given diff blob corresponds to.

Pipeline per image:
  1. Fetch matchedRgb + vendor JPG.
  2. Resize both to a common (max-side) resolution.
  3. SSIM map (skimage, window 7) → "where structure changed."
  4. DINOv2 patch cosine-distance map → "where semantic content changed."
  5. Threshold both, take logical AND for precision.
  6. Morphological opening + closing + min-area filter.
  7. Save binary mask at the matchedRgb resolution.
  8. Upload to R2; POST to /api/internal/auto-mask-test as a new
     modelVersion so it renders side-by-side with prompt-based runs.

Usage:
    python -m scripts.pseudo_label_diff \\
        --condition reflection \\
        --ssim-threshold 0.20 \\
        --dino-threshold 0.15 \\
        --min-area 200 \\
        --tag v1

Env:
    WORKER_TOKEN              dashboard auth (required)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data (matched/vendor + mask uploads)
    RCLONE_R2                 default 'r2'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image  # type: ignore
from scipy import ndimage  # type: ignore
from skimage.metrics import structural_similarity  # type: ignore
from skimage.morphology import (  # type: ignore
    binary_closing, binary_opening, disk, remove_small_objects,
)
from transformers import AutoImageProcessor, AutoModel  # type: ignore

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
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
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {text[:300]}") from e


def rclone_fetch(r2_key: str, dest: Path) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}", str(dest),
         "--progress=false"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        log(f"  WARN fetch {r2_key}: rc={r.returncode} {r.stderr[-160:].strip()}")
        return False
    return True


def rclone_upload(local: Path, r2_key: str) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", str(local), f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}",
         "--progress=false"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        log(f"  WARN upload {r2_key}: rc={r.returncode} {r.stderr[-160:].strip()}")
        return False
    return True


def get_candidates(*, condition: str, model_version: str,
                   limit: int) -> list[dict]:
    sp = urllib.parse.urlencode({
        "condition": condition,
        "modelVersion": model_version,
        "limit": str(limit),
    })
    resp = _request("GET", f"/api/internal/diff-pair-candidates?{sp}")
    log(f"  candidates: {resp.get('todo')}/{resp.get('total')} todo "
        f"(already done: {resp.get('alreadyDone')})")
    return resp.get("items") or []


# ---- DINOv2 patch features ----

class PatchFeaturizer:
    """Same setup as few_shot_dinov2_mask: DINOv2 patch features at
    grid*14 input resolution. Returns L2-normalized (G, G, D)."""

    def __init__(self, model_id: str, device: str, grid: int = 32):
        log(f"  loading {model_id} on {device} ...")
        t0 = time.time()
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.device = device
        self.grid = grid
        self.patch = 14
        self.resize = grid * self.patch
        self.mean = torch.tensor(self.processor.image_mean).view(3, 1, 1).to(device)
        self.std = torch.tensor(self.processor.image_std).view(3, 1, 1).to(device)
        log(f"  loaded in {time.time() - t0:.1f}s  grid={grid} resize={self.resize}")

    @torch.no_grad()
    def features(self, image: Image.Image) -> torch.Tensor:
        im = image.convert("RGB").resize(
            (self.resize, self.resize), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        arr = arr.permute(2, 0, 1).to(self.device)
        arr = (arr - self.mean) / self.std
        out = self.model(pixel_values=arr.unsqueeze(0))
        h = out.last_hidden_state[:, 1:, :]
        h = F.normalize(h, dim=-1)
        return h.view(self.grid, self.grid, -1).contiguous()


def ssim_diff_mask(src_rgb: Image.Image, vendor_rgb: Image.Image,
                   threshold: float, target_w: int) -> np.ndarray:
    """Return a (H, W) bool mask of pixels where SSIM < (1 - threshold).
    Both images are first resized to (target_w, target_h) preserving the
    src aspect."""
    src_w, src_h = src_rgb.size
    target_h = round(src_h * (target_w / max(src_w, 1)))
    s = np.asarray(
        src_rgb.convert("L").resize((target_w, target_h), Image.BILINEAR),
        dtype=np.float32)
    v = np.asarray(
        vendor_rgb.convert("L").resize((target_w, target_h), Image.BILINEAR),
        dtype=np.float32)
    # data_range explicit since we passed float arrays
    _, ssim_map = structural_similarity(s, v, data_range=255.0, win_size=7,
                                        full=True)
    return (1 - ssim_map) > threshold


def dino_diff_mask(feat: PatchFeaturizer, src_rgb: Image.Image,
                   vendor_rgb: Image.Image, threshold: float,
                   target_shape: tuple[int, int]) -> np.ndarray:
    """Cosine-distance map between src and vendor patch features, then
    upsample (NEAREST) to target_shape (H, W) and threshold."""
    fs = feat.features(src_rgb)  # (G, G, D)
    fv = feat.features(vendor_rgb)
    cos = (fs * fv).sum(dim=-1)  # (G, G)
    diff = (1 - cos).cpu().numpy().astype(np.float32)
    binary = diff > threshold
    h, w = target_shape
    return np.asarray(
        Image.fromarray(binary.astype(np.uint8) * 255, mode="L").resize(
            (w, h), Image.NEAREST),
        dtype=np.uint8) > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--dinov2", default="facebook/dinov2-base")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--target-width", type=int, default=896,
                    help="SSIM is computed at this width preserving aspect; "
                         "DINOv2 always runs at grid*14.")
    ap.add_argument("--ssim-threshold", type=float, default=0.20,
                    help="threshold on (1 - SSIM); higher = require larger "
                         "structural change. 0.20 ≈ SSIM<0.80.")
    ap.add_argument("--dino-threshold", type=float, default=0.15,
                    help="threshold on (1 - cosine); higher = require larger "
                         "feature shift. 0.15 ≈ cosine<0.85.")
    ap.add_argument("--combine", choices=("and", "or"), default="and")
    ap.add_argument("--open-radius", type=int, default=3,
                    help="binary opening radius to remove salt noise")
    ap.add_argument("--close-radius", type=int, default=7,
                    help="binary closing radius to fill holes in regions")
    ap.add_argument("--min-area", type=int, default=200,
                    help="drop connected components smaller than this many "
                         "pixels (at target_width resolution)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required")
        return 2

    tag = args.tag or (
        f"ssim{int(args.ssim_threshold*100):02d}"
        f"_dino{int(args.dino_threshold*100):02d}"
        f"_{args.combine}"
    )
    model_version = f"diff__{tag}"
    log(f"pseudo-label diff: condition={args.condition} "
        f"model_version={model_version} device={args.device}")

    items = get_candidates(condition=args.condition,
                           model_version=model_version, limit=args.limit)
    if not items:
        log("nothing to do")
        return 0

    feat = PatchFeaturizer(args.dinov2, args.device, grid=args.grid)

    n_done = n_hit = n_err = 0
    t_start = time.time()
    open_se = disk(args.open_radius) if args.open_radius > 0 else None
    close_se = disk(args.close_radius) if args.close_radius > 0 else None

    with tempfile.TemporaryDirectory(prefix="arem-diff-") as scratch:
        sp = Path(scratch)
        for idx, it in enumerate(items, start=1):
            t0 = time.time()
            image_key = it["imageR2Path"]
            matched_local = sp / f"matched-{idx:05d}.jpg"
            vendor_local = sp / f"vendor-{idx:05d}.jpg"

            ok_m = rclone_fetch(it["matchedRgbR2Path"], matched_local)
            ok_v = rclone_fetch(it["vendorImageR2Path"], vendor_local)
            if not (ok_m and ok_v):
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/auto-mask-test", {
                            "condition": args.condition,
                            "imageR2Path": image_key,
                            "modelVersion": model_version,
                            "prompt": f"diff:{tag}",
                            "errorMessage": "fetch from R2 failed",
                        })
                    except Exception as e:
                        log(f"    WARN POST: {e}")
                n_err += 1
                continue
            try:
                with Image.open(matched_local) as im_m, \
                     Image.open(vendor_local) as im_v:
                    im_m = im_m.convert("RGB")
                    im_v = im_v.convert("RGB")
                    orig_w, orig_h = im_m.size

                # Compute SSIM at target_width
                src_w, src_h = im_m.size
                target_w = min(args.target_width, src_w)
                target_h = round(src_h * (target_w / max(src_w, 1)))
                ssim_mask = ssim_diff_mask(
                    im_m, im_v, args.ssim_threshold, target_w)
                dino_mask = dino_diff_mask(
                    feat, im_m, im_v, args.dino_threshold,
                    (target_h, target_w))

                if args.combine == "and":
                    combined = ssim_mask & dino_mask
                else:
                    combined = ssim_mask | dino_mask

                if open_se is not None:
                    combined = binary_opening(combined, open_se)
                if close_se is not None:
                    combined = binary_closing(combined, close_se)
                if args.min_area > 0:
                    combined = remove_small_objects(combined,
                                                    min_size=args.min_area)

                # Upsample mask to original resolution
                mask_img = Image.fromarray(
                    (combined.astype(np.uint8) * 255), mode="L"
                ).resize((orig_w, orig_h), Image.NEAREST)
                px = int(np.asarray(mask_img, dtype=np.uint8).sum() // 255)

                mask_path: str | None = None
                if px > 0:
                    safe_name = re.sub(r"[^a-z0-9._-]+", "_",
                                       image_key.split("/")[-1].lower())
                    mask_key = (
                        f"auto-masks/{args.condition}/{model_version}/"
                        f"{safe_name.rsplit('.', 1)[0]}.png"
                    )
                    mask_local = sp / f"mask-{idx:05d}.png"
                    mask_img.save(mask_local, format="PNG", optimize=True)
                    if not args.dry_run and rclone_upload(mask_local, mask_key):
                        mask_path = mask_key

                runtime_ms = int((time.time() - t0) * 1000)
                if not args.dry_run:
                    _request("POST", "/api/internal/auto-mask-test", {
                        "condition": args.condition,
                        "imageR2Path": image_key,
                        "modelVersion": model_version,
                        "prompt": f"diff:{tag}",
                        "autoMaskR2Path": mask_path,
                        "pixelCount": px,
                        "imageWidth": orig_w,
                        "imageHeight": orig_h,
                        "runtimeMs": runtime_ms,
                    })
                n_done += 1
                if px > 0:
                    n_hit += 1
                if idx % 5 == 0 or idx == len(items) or px > 0:
                    rate = idx / max(time.time() - t_start, 0.01)
                    log(f"  [{idx}/{len(items)}] {image_key}: "
                        f"{px} px {'(hit)' if px > 0 else '(empty)'} "
                        f"{runtime_ms}ms ({rate:.1f} img/s)")
            except Exception as e:
                log(f"  [{idx}/{len(items)}] {image_key}: ERROR {str(e)[:200]}")
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/auto-mask-test", {
                            "condition": args.condition,
                            "imageR2Path": image_key,
                            "modelVersion": model_version,
                            "prompt": f"diff:{tag}",
                            "errorMessage": str(e)[:500],
                            "runtimeMs": int((time.time() - t0) * 1000),
                        })
                    except Exception as e2:
                        log(f"    WARN POST: {e2}")
                n_err += 1
            finally:
                for f in (matched_local, vendor_local):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass

    dt = time.time() - t_start
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(items),1):.2f}s/img avg) ---")
    log(f"  processed: {n_done}  hits: {n_hit}  errors: {n_err}")
    log(f"  model_version: {model_version}")
    log(f"  next: python -m scripts.score_against_ground_truth "
        f"--condition {args.condition} --heldout 0 "
        f"--model-version {model_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
