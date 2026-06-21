#!/usr/bin/env python3
"""Prototype: learn the AREM drone 'finish' as a 3D LUT (tone/style only).

The drone finish was meant to be a tone/style transform (gentle_finish:
warmth/dehaze/contrast/vibrance + sharpening — no blur). The NAFNet that
replaced it regressed into spatial smearing. A 3D LUT is a pure per-color
mapping: it CANNOT blur or move anything — output sharpness == input sharpness
by construction. We fit it empirically from the real (developed DNG -> v1
finished) pairs so it captures the actual style.

  fit : accumulate input_rgb -> target_rgb over many pixels into a 33^3 LUT
  fill: identity for unseen cells, then light 3D smoothing (no banding)
  apply: trilinear interpolation (vectorized)

Usage:
  build_drone_lut.py fit  <pairs_dir> <lut_out.npy>     # pairs_dir has <stem>.DNG (or _dev.jpg) + <stem>.jpg target
  build_drone_lut.py apply <lut.npy> <in.jpg> <out.jpg>
"""
import sys, glob, os
import numpy as np, cv2, rawpy
from scipy.ndimage import map_coordinates, gaussian_filter

N = 33  # LUT grid per channel

def good_develop(dng):
    with rawpy.imread(dng) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0,
                              output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                              output_bps=8, user_flip=-1)
    return rgb[:, :, ::-1].copy()  # -> BGR uint8

def fit(pairs_dir, out_npy):
    sums = np.zeros((N, N, N, 3), np.float64)
    cnts = np.zeros((N, N, N), np.float64)
    targets = sorted(glob.glob(os.path.join(pairs_dir, "*.jpg")))
    print(f"{len(targets)} target jpgs", flush=True)
    used = 0
    for tj in targets:
        stem = os.path.splitext(os.path.basename(tj))[0]
        dng = None
        for ext in ("DNG", "dng"):
            p = os.path.join(pairs_dir, f"{stem}.{ext}")
            if os.path.isfile(p): dng = p; break
        if not dng:
            continue
        try:
            src = good_develop(dng)
            tgt = cv2.imread(tj)
        except Exception as e:
            print(f"  skip {stem}: {str(e)[:60]}", flush=True); continue
        if tgt is None: continue
        if tgt.shape[:2] != src.shape[:2]:
            tgt = cv2.resize(tgt, (src.shape[1], src.shape[0]))
        # downsample both equally for speed (preserves the color mapping)
        s = 900 / max(src.shape[:2])
        if s < 1:
            src = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            tgt = cv2.resize(tgt, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        si = np.round(src.astype(np.float32) / 255 * (N - 1)).astype(np.int32).reshape(-1, 3)
        tv = tgt.astype(np.float64).reshape(-1, 3)
        np.add.at(sums, (si[:, 0], si[:, 1], si[:, 2]), tv)
        np.add.at(cnts, (si[:, 0], si[:, 1], si[:, 2]), 1.0)
        used += 1
        if used % 20 == 0: print(f"  fit {used} pairs", flush=True)
    print(f"fit on {used} pairs; filled cells {int((cnts>0).sum())}/{N**3}", flush=True)
    # identity LUT (B,G,R cell -> its own color), then overwrite seen cells
    grid = np.linspace(0, 255, N)
    lut = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1)  # (N,N,N,3) = identity in BGR-index order
    seen = cnts > 0
    lut[seen] = sums[seen] / cnts[seen][..., None]
    # light count-aware smoothing to fill gaps + kill banding: blur values and a
    # weight mask, divide — so sparse cells borrow from dense neighbors.
    w = np.clip(cnts, 0, 50)
    for c in range(3):
        num = gaussian_filter(lut[..., c] * (w > 0), 1.0)
        den = gaussian_filter((w > 0).astype(float), 1.0)
        smoothed = num / np.maximum(den, 1e-6)
        lut[..., c] = np.where(seen, lut[..., c], smoothed)
    lut = gaussian_filter(lut, (0.6, 0.6, 0.6, 0))  # final gentle smooth
    np.save(out_npy, np.clip(lut, 0, 255).astype(np.float32))
    print(f"saved {out_npy}", flush=True)

def apply_lut(lut, bgr):
    coords = bgr.astype(np.float32) / 255 * (N - 1)  # (H,W,3) index space, BGR order
    H, W = bgr.shape[:2]
    cc = coords.reshape(-1, 3).T  # (3, HW)
    out = np.empty((H * W, 3), np.float32)
    for c in range(3):
        out[:, c] = map_coordinates(lut[..., c], cc, order=1, mode="nearest")
    return np.clip(out.reshape(H, W, 3), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "apply":
        lut = np.load(sys.argv[2])
        img = cv2.imread(sys.argv[3]) if not sys.argv[3].lower().endswith(("dng",)) else good_develop(sys.argv[3])
        cv2.imwrite(sys.argv[4], apply_lut(lut, img), [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"wrote {sys.argv[4]}", flush=True)
