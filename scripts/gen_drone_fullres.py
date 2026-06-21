#!/usr/bin/env python3
"""Render full-resolution drone-finish samples for review (NAFNet v2 + clarity).
Tiled inference so NAFNet runs at native res with no OOM and no seams."""
import sys, numpy as np, cv2, torch, rawpy
sys.path.insert(0, "/home/jordan/arem-worker/pipeline")
from models.nafnet import NAFNet

dev = "cuda"
na = NAFNet(in_channels=3, out_channels=3, width=32, middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
            use_residual=True, residual_start=0).to(dev).eval()
na.load_state_dict(torch.load("/home/jordan/drone_retrain/nafnet_v2.pt", map_location=dev)["model"])

def develop(dng):
    with rawpy.imread(dng) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0,
                              output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                              output_bps=8, user_flip=-1)
    return rgb[:, :, ::-1].copy()

def _run(patch):
    x = torch.from_numpy(patch[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad():
        o = na(x)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    return np.clip(o * 255, 0, 255).astype(np.float32)

def tiled(bgr, tile=1536, ov=256):
    H, W = bgr.shape[:2]
    if max(H, W) <= tile:
        return _run(bgr).astype(np.uint8)
    out = np.zeros((H, W, 3), np.float32); acc = np.zeros((H, W, 1), np.float32)
    win = (np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3)[..., None]
    step = tile - ov
    ys = sorted(set([min(y, H - tile) for y in range(0, H, step)] + [0, H - tile]))
    xs = sorted(set([min(x, W - tile) for x in range(0, W, step)] + [0, W - tile]))
    ys = [max(0, y) for y in ys]; xs = [max(0, x) for x in xs]
    for y in ys:
        for x in xs:
            o = _run(bgr[y:y + tile, x:x + tile])
            out[y:y + tile, x:x + tile] += o * win
            acc[y:y + tile, x:x + tile] += win
    return np.clip(out / np.maximum(acc, 1e-6), 0, 255).astype(np.uint8)

def clarity(bgr, clar=0.18, sharp=0.45):
    img = bgr.astype(np.float32) / 255.0
    img = np.clip(img + clar * (img - cv2.GaussianBlur(img, (0, 0), 4)), 0, 1)
    img = np.clip(img + sharp * (img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    return (img * 255).astype(np.uint8)

def to_12mp(bgr):
    """Delivery resolution = 12 MP. Downscale (preserving aspect) if larger;
    leave as-is if already <=12MP (T-2325 CAM frames are 4000x3000 = 12MP)."""
    H, W = bgr.shape[:2]
    mp = H * W
    if mp <= 12_000_000:
        return bgr
    s = (12_000_000 / mp) ** 0.5
    return cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)

# jobs: (label, source_kind, path)
jobs = [
    ("0280", "dng", "/tmp/dronefix/in/CAM_20260620124253_0280_W.DNG"),
    ("0281", "dng", "/tmp/dronefix/in/CAM_20260620124332_0281_W.DNG"),
    ("val_623", "dng", "/tmp/fr2/val.DNG"),
]
for label, _, p in jobs:
    dev_bgr = to_12mp(develop(p))              # deliver at 12 MP
    finished = clarity(_run(dev_bgr).astype(np.uint8))   # direct 12MP pass (no tiling)
    cv2.imwrite(f"/tmp/fr2/{label}_NAFNet_clarity.jpg", finished, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"/tmp/fr2/{label}_develop.jpg", dev_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{label}: {finished.shape[1]}x{finished.shape[0]} ({finished.shape[0]*finished.shape[1]/1e6:.1f}MP) done", flush=True)
