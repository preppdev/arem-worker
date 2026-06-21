#!/usr/bin/env python3
"""Isolate which exterior stage paints blue on white objects.

For each Stage-1 frame: run exterior Stage-2 (may26_exterior) then Stage-3
(polish_exterior_v1), saving s1/s2/s3 side by side. Also prints, for the
brightest near-white pixels of s1 (likely white walls/roof/trim, NOT sky which
we exclude by taking the lower 70% of the frame), the mean B-R (blue minus red)
shift introduced by each stage — a positive jump = that stage is bluing whites.
"""
import os, sys, glob
import numpy as np, cv2, torch
REPO = os.path.expanduser("~/arem-worker"); sys.path.insert(0, REPO)
from pipeline.run_pipeline import load_stage2, load_stage3, stage2_infer, stage3_infer

S1_DIR, OUT, S2_CK, S3_CK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
s2 = load_stage2(__import__("pathlib").Path(S2_CK), DEV)
s3 = load_stage3(__import__("pathlib").Path(S3_CK), DEV)
from pathlib import Path

def white_lowerframe_mask(bgr):
    """near-white pixels in the LOWER 70% (exclude top sky band)."""
    h, w = bgr.shape[:2]
    b, g, r = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    mn = np.minimum(np.minimum(b, g), r); mx = np.maximum(np.maximum(b, g), r)
    white = (mn > 175) & ((mx - mn) < 28)         # bright + near-neutral
    band = np.zeros((h, w), bool); band[int(h * 0.30):, :] = True
    return white & band

def br_mean(bgr, m):
    if m.sum() < 50: return None
    return float((bgr[:, :, 0].astype(int) - bgr[:, :, 2].astype(int))[m].mean())

print(f"{'frame':<16} {'whitePx':>8} {'s1 B-R':>7} {'s2 B-R':>7} {'s3 B-R':>7}  {'d2':>5} {'d3':>5}")
for p in sorted(glob.glob(os.path.join(S1_DIR, "*.jpg")))[:12]:
    stem = Path(p).stem
    s1 = cv2.imread(p)
    s2p, s3p = Path(OUT) / f"{stem}_s2.jpg", Path(OUT) / f"{stem}_s3.jpg"
    stage2_infer(s2, Path(p), s2p, DEV)
    stage3_infer(s3, s2p, s3p, DEV)
    i2, i3 = cv2.imread(str(s2p)), cv2.imread(str(s3p))
    m = white_lowerframe_mask(s1)
    b1, b2, b3 = br_mean(s1, m), br_mean(i2, m), br_mean(i3, m)
    if b1 is None:
        print(f"{stem:<16} {int(m.sum()):>8}   (no white region)"); continue
    print(f"{stem:<16} {int(m.sum()):>8} {b1:>7.1f} {b2:>7.1f} {b3:>7.1f}  {b2-b1:>5.1f} {b3-b2:>5.1f}")
    # side-by-side sheet (downscaled)
    def tag(im, t):
        im = cv2.resize(im, (440, int(im.shape[0]*440/im.shape[1]))).copy()
        cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2); return im
    cv2.imwrite(str(Path(OUT) / f"sheet_{stem}.jpg"),
                np.hstack([tag(s1, "S1"), tag(i2, "S2"), tag(i3, "S3")]), [cv2.IMWRITE_JPEG_QUALITY, 88])
