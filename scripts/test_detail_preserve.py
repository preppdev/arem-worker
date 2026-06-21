#!/usr/bin/env python3
"""Test a detail-preserving drone finish: apply only the NAFNet's LOW-FREQUENCY
color grade onto the SHARP develop (keeps full native detail), vs the current
direct-NAFNet (which smooths micro-texture). Crops at 100% for comparison."""
import sys, os, numpy as np, cv2, torch
sys.path.insert(0, os.path.expanduser("~/arem-worker"))
sys.path.insert(0, os.path.expanduser("~/arem-worker/pipeline"))
from pipeline.drone_finish import develop, to_12mp, clarity
from models.nafnet import NAFNet

DEV = "cuda"
ck = torch.load(os.path.expanduser("~/polish-cache/models/drone-finish/nafnet_v2.pt"), map_location=DEV, weights_only=False)
net = NAFNet(in_channels=3, out_channels=3, width=32, middle_blk_num=12,
             enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
             use_residual=True, residual_start=0).to(DEV).eval()
net.load_state_dict(ck["model"])

def nafnet_raw(bgr):
    x = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEV)
    with torch.inference_mode():
        o = torch.clamp(net(x), 0, 1)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    return np.clip(o * 255, 0, 255).astype(np.float32)

DNG = sys.argv[1]
base = to_12mp(develop(DNG)).astype(np.float32)
graded = nafnet_raw(base)

# CURRENT: direct NAFNet + clarity (smooths detail)
cur = clarity(np.clip(graded, 0, 255).astype(np.uint8))

# FIXED: low-freq grade transfer onto the sharp develop + clarity (keeps detail)
out = {}
for sigma in (2, 3):
    delta_lf = cv2.GaussianBlur(graded - base, (0, 0), sigmaX=sigma, sigmaY=sigma)
    fixed = clarity(np.clip(base + delta_lf, 0, 255).astype(np.uint8))
    out[sigma] = fixed

box = (1700, 1150, 2400, 1850)
def crop(img, name):
    x0, y0, x1, y1 = box
    cv2.imwrite(f"/tmp/dp_{name}.jpg", img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(f"/tmp/dpfull_{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
crop(np.clip(base, 0, 255).astype(np.uint8), "develop")
crop(cur, "current")
crop(out[2], "fixed_s2")
crop(out[3], "fixed_s3")
def lap(x): return cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
print("sharpness — develop:%.0f current:%.0f fixed_s2:%.0f fixed_s3:%.0f" % (
    lap(base.astype(np.uint8)), lap(cur), lap(out[2]), lap(out[3])), flush=True)
