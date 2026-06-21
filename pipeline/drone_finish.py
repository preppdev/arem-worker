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
    # The grade is MEANT to be a smooth tonal/color delta (see module docstring),
    # but NAFNet is a restoration net — on texture-heavy frames (dense foliage)
    # it also denoises fine detail, which shows up as high-frequency content in
    # the delta. Computed at WORK_LONG and bilinearly upscaled ~2.5x to native,
    # that detail-removal map is misaligned against the sharp full-res develop and
    # subtracts smeared detail → a "tilt-shift" blur on the textured regions while
    # low-frequency subjects (roof, pool, pavement) survive. Low-pass the delta so
    # only the smooth grade is applied at native res and ALL fine detail comes from
    # `dev`. sigma is set in small-scale px; INTER_AREA downscale already removed
    # detail finer than ~1px, so this targets the 1–8px band the upscale smears.
    sigma = max(1.5, WORK_LONG / 512.0)  # ~4px at 2048
    delta_small = cv2.GaussianBlur(delta_small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.resize(delta_small, (W, H), interpolation=cv2.INTER_LINEAR)


def _sharpness(bgr):
    """Relative high-frequency energy (variance of Laplacian on luma). Cheap
    blur metric — higher = sharper."""
    g = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

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
    ap.add_argument("--model", default=None)  # required only when DNGs are present
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # Map stem -> path for each kind. Mixed-input rules (per shoot, per stem):
    #   - DNG present            → develop + grade it (a paired JPG is ignored)
    #   - JPG with no companion DNG → pass through unedited (copy as-is)
    #   - only JPGs              → all pass through
    # Only DNGs are ever edited; everything lands in --out-dir/<stem>.jpg.
    def gather(exts):
        out = {}
        for e in exts:
            for p in glob.glob(os.path.join(a.drone_dir, "*." + e)):
                out[os.path.splitext(os.path.basename(p))[0]] = p
        return out
    dngs = gather(["dng", "DNG"])
    jpgs = gather(["jpg", "JPG", "jpeg", "JPEG"])
    print(f"drone_finish: {len(dngs)} DNG(s), {len(jpgs)} JPG(s)", flush=True)

    # ── TEMPORARY model bypass (DRONE_BYPASS_MODEL=1) ─────────────────────────
    # The learned drone grade is producing blurred/over-processed output, so
    # while it's being fixed we ship the camera's own finished frame instead of
    # running the model. Per DNG: use the companion camera JPG if one sits beside
    # it; otherwise fall back to a clean rawpy develop (sharp, just ungraded).
    # Toggle off by unsetting the env — no code change needed.
    if os.environ.get("DRONE_BYPASS_MODEL", "").strip().lower() in ("1", "true", "yes", "on"):
        print("  DRONE_BYPASS_MODEL set — model DISABLED; using camera JPG beside each DNG "
              "(clean develop where none exists)", flush=True)
        jpg_pass = develop_pass = 0
        for stem, dng in sorted(dngs.items()):
            outp = os.path.join(a.out_dir, f"{stem}.jpg")
            if stem in jpgs:  # companion camera JPG → ship it as-is
                try:
                    shutil.copy2(jpgs[stem], outp); jpg_pass += 1
                    print(f"  jpg-beside {stem}", flush=True)
                except Exception as e:
                    print(f"  ERR jpg-beside {stem}: {str(e)[:120]}", flush=True)
            else:  # DNG-only → clean develop, no model grade
                try:
                    cv2.imwrite(outp, develop(dng), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                    copy_exif(dng, outp); develop_pass += 1
                    print(f"  clean-develop {stem}", flush=True)
                except Exception as e:
                    print(f"  ERR develop {stem}: {str(e)[:120]}", flush=True)
        # lone JPGs (no DNG) still pass through as before
        lone = 0
        for stem, p in sorted(jpgs.items()):
            if stem in dngs:
                continue
            try:
                shutil.copy2(p, os.path.join(a.out_dir, f"{stem}.jpg")); lone += 1
            except Exception as e:
                print(f"  ERR passthrough {stem}: {str(e)[:120]}", flush=True)
        print(f"DRONE_FINISH_OUTPUTS edited=0 passthrough={jpg_pass + develop_pass + lone} "
              f"(BYPASS: jpg-beside={jpg_pass} clean-develop={develop_pass} lone-jpg={lone}) "
              f"(dng={len(dngs)} jpg={len(jpgs)})", flush=True)
        return

    model = None
    if dngs:
        if not a.model or not os.path.isfile(a.model):
            print("  WARN DNGs present but no model — cannot develop; passing through JPGs only", flush=True)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ck = torch.load(a.model, map_location=device, weights_only=False)
            model = NAFNet(in_channels=3, out_channels=3, width=int(ck.get("width", 32)),
                           middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
                           use_residual=True, residual_start=int(ck.get("residual_start", 0))).to(device).eval()
            model.load_state_dict(ck["model"])
            print(f"  model val_l1={ck.get('val_l1')} device={device}", flush=True)

    edited = passed = 0
    if model:
        device = next(model.parameters()).device.type
        for stem, p in sorted(dngs.items()):
            try:
                dev = develop(p).astype(np.float32)
                delta = grade_delta(model, dev.astype(np.uint8), device)
                out = np.clip(dev + delta, 0, 255).astype(np.uint8)
                # Safety net: a finished frame should never be materially blurrier
                # than its own develop. If it is (a grade artifact slipped through),
                # ship the clean develop rather than a blurred edit. We'd rather
                # deliver an ungraded-but-sharp frame than a smeared one.
                dev_u8 = dev.astype(np.uint8)
                s_out, s_dev = _sharpness(out), _sharpness(dev_u8)
                if s_dev > 0 and s_out < 0.6 * s_dev:
                    print(f"  GUARD {stem}: edit blurrier than develop "
                          f"({s_out:.0f} < 0.6*{s_dev:.0f}) — shipping develop", flush=True)
                    out = dev_u8
                outp = os.path.join(a.out_dir, f"{stem}.jpg")
                cv2.imwrite(outp, out, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                copy_exif(p, outp)
                edited += 1
                print(f"  edit {stem} ({out.shape[1]}x{out.shape[0]})", flush=True)
            except Exception as e:
                print(f"  ERR {stem}: {str(e)[:120]}", flush=True)
    # pass-through: JPGs with no companion DNG, copied byte-for-byte (no re-encode)
    for stem, p in sorted(jpgs.items()):
        if stem in dngs:
            continue  # paired with a DNG → ignore the JPG, only the DNG is edited
        try:
            shutil.copy2(p, os.path.join(a.out_dir, f"{stem}.jpg"))
            passed += 1
            print(f"  pass {stem}", flush=True)
        except Exception as e:
            print(f"  ERR passthrough {stem}: {str(e)[:120]}", flush=True)
    print(f"DRONE_FINISH_OUTPUTS edited={edited} passthrough={passed} (dng={len(dngs)} jpg={len(jpgs)})", flush=True)

if __name__ == "__main__":
    main()
