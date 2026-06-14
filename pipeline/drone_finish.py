#!/usr/bin/env python3
"""Drone finishing path — the single-frame analogue of the interior pipeline.

For each DNG in --drone-dir: develop with rawpy (camera WB + auto-bright + sRGB
gamma, the recipe the model was trained against) → apply the learned grade with
the NAFNet w32 residual model → write a full-resolution JPG to --out-dir, with
GPS + orientation EXIF copied from the source DNG (drone shots carry geo).

Scale handling: DJI DNGs are ~20-48 MP, far larger than the 1024-crop training
scale, and running NAFNet at native res both OOMs and applies the grade at the
wrong spatial scale. The model is residual (output = net(input) + input), so the
grade is the smooth delta `net(input)`. We compute that delta at the TRAINING
resolution (2048 long-side), upscale the delta to native res, and add it to the
full develop — full delivery resolution, trained-scale grade.

Run as a subprocess from worker.py (isolated GPU), mirroring run_inference.
"""
import os, sys, glob, argparse, subprocess, shutil
import numpy as np, cv2, torch, rawpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nafnet import NAFNet

WORK_LONG = 2048  # model-pass long side (matches build_pairs working res)

def develop(dng_path):
    """rawpy develop -> BGR uint8, native sensor resolution."""
    with rawpy.imread(dng_path) as raw:
        dev = raw.postprocess(use_camera_wb=True, no_auto_bright=False,
                              output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                              output_bps=8, user_flip=-1)
    return np.ascontiguousarray(dev[:, :, ::-1])  # RGB -> BGR

def grade_delta(model, dev_bgr, device):
    """Run the model at WORK_LONG and return the residual grade delta (float BGR,
    -255..255) at the SAME size as dev_bgr (delta computed small, upscaled)."""
    H, W = dev_bgr.shape[:2]
    scale = WORK_LONG / max(H, W)
    sw, sh = max(16, int(round(W * scale))), max(16, int(round(H * scale)))
    small = cv2.resize(dev_bgr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    # pad to multiple of 16 (NAFNet has 4 downsampling levels)
    ph, pw = (16 - sh % 16) % 16, (16 - sw % 16) % 16
    padded = cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    t = torch.from_numpy(padded[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = torch.clamp(model(t), 0, 1)
    o = (out[0].permute(1, 2, 0).cpu().numpy() * 255)[:, :, ::-1]  # BGR float
    o = o[:sh, :sw]
    delta_small = o - small  # residual grade at small scale
    return cv2.resize(delta_small, (W, H), interpolation=cv2.INTER_LINEAR)

def copy_exif(dng, jpg):
    """Best-effort GPS + orientation passthrough from the DNG."""
    try:
        subprocess.run(["exiftool", "-tagsFromFile", dng, "-GPS:all", "-Orientation",
                        "-overwrite_original", jpg], capture_output=True, timeout=60)
    except Exception as e:
        print(f"  exif copy skip {os.path.basename(jpg)}: {str(e)[:80]}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.model, map_location=device, weights_only=False)
    model = NAFNet(in_channels=3, out_channels=3, width=int(ck.get("width", 32)),
                   middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
                   use_residual=True, residual_start=int(ck.get("residual_start", 0))).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"drone_finish: model val_l1={ck.get('val_l1')} device={device}", flush=True)

    dngs = sorted(set(glob.glob(os.path.join(a.drone_dir, "*.dng")) +
                      glob.glob(os.path.join(a.drone_dir, "*.DNG"))))
    print(f"drone_finish: {len(dngs)} DNG(s)", flush=True)
    ok = 0
    for p in dngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            dev = develop(p).astype(np.float32)
            delta = grade_delta(model, dev.astype(np.uint8), device)
            out = np.clip(dev + delta, 0, 255).astype(np.uint8)
            outp = os.path.join(a.out_dir, f"{stem}.jpg")
            cv2.imwrite(outp, out, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            copy_exif(p, outp)
            ok += 1
            print(f"  ok {stem} ({out.shape[1]}x{out.shape[0]})", flush=True)
        except Exception as e:
            print(f"  ERR {stem}: {str(e)[:120]}", flush=True)
    print(f"DRONE_FINISH_OUTPUTS {ok}/{len(dngs)}", flush=True)

if __name__ == "__main__":
    main()
