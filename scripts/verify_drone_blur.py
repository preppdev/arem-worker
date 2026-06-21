#!/usr/bin/env python3
"""One-off: verify the drone-grade blur fix on real DNGs.

For each DNG: develop, then apply the grade two ways —
  OLD: residual delta upscaled raw (current prod behavior)
  NEW: residual delta low-passed before upscale (the fix)
Reports variance-of-Laplacian sharpness for develop/old/new and writes all
three JPGs so they can be eyeballed. Higher sharpness = sharper.
"""
import os, sys, glob
import numpy as np, cv2, torch, rawpy

REPO = os.path.expanduser("~/arem-worker")
sys.path.insert(0, os.path.join(REPO, "pipeline"))
from models.nafnet import NAFNet

WORK_LONG = 2048
DRONE_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]
MODEL = sys.argv[3]
os.makedirs(OUT_DIR, exist_ok=True)

def develop(p):
    with rawpy.imread(p) as raw:
        dev = raw.postprocess(use_camera_wb=True, no_auto_bright=False,
                              output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                              output_bps=8, user_flip=-1)
    return np.ascontiguousarray(dev[:, :, ::-1])

def model_out_small(model, dev_bgr, device):
    H, W = dev_bgr.shape[:2]
    scale = WORK_LONG / max(H, W)
    sw, sh = max(16, int(round(W * scale))), max(16, int(round(H * scale)))
    small = cv2.resize(dev_bgr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    ph, pw = (16 - sh % 16) % 16, (16 - sw % 16) % 16
    padded = cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    t = torch.from_numpy(padded[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = torch.clamp(model(t), 0, 1)
    o = (out[0].permute(1, 2, 0).cpu().numpy() * 255)[:, :, ::-1]
    return o[:sh, :sw], small, (W, H)

def sharp(bgr):
    g = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(MODEL, map_location=device, weights_only=False)
model = NAFNet(in_channels=3, out_channels=3, width=int(ck.get("width", 32)),
               middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
               use_residual=True, residual_start=int(ck.get("residual_start", 0))).to(device).eval()
model.load_state_dict(ck["model"])
sigma = max(1.5, WORK_LONG / 512.0)
print(f"model val_l1={ck.get('val_l1')} sigma={sigma:.2f} device={device}\n")
print(f"{'stem':<22} {'develop':>9} {'OLD':>9} {'NEW':>9}  {'OLD/dev':>7} {'NEW/dev':>7}")

for p in sorted(glob.glob(os.path.join(DRONE_DIR, "*.DNG")) + glob.glob(os.path.join(DRONE_DIR, "*.dng"))):
    stem = os.path.splitext(os.path.basename(p))[0]
    dev = develop(p).astype(np.float32)
    o, small, (W, H) = model_out_small(model, dev.astype(np.uint8), device)
    delta_small = o - small
    # OLD
    delta_old = cv2.resize(delta_small, (W, H), interpolation=cv2.INTER_LINEAR)
    out_old = np.clip(dev + delta_old, 0, 255).astype(np.uint8)
    # NEW (low-pass the delta first)
    delta_lp = cv2.GaussianBlur(delta_small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    delta_new = cv2.resize(delta_lp, (W, H), interpolation=cv2.INTER_LINEAR)
    out_new = np.clip(dev + delta_new, 0, 255).astype(np.uint8)
    sd, so, sn = sharp(dev), sharp(out_old), sharp(out_new)
    print(f"{stem:<22} {sd:>9.0f} {so:>9.0f} {sn:>9.0f}  {so/sd:>7.2f} {sn/sd:>7.2f}")
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_dev.jpg"), dev.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_old.jpg"), out_old, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_new.jpg"), out_new, [cv2.IMWRITE_JPEG_QUALITY, 92])
