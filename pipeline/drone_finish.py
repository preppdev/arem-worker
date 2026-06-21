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


def gentle_finish(bgr, dehaze=0.12, contrast=0.10, vibrance=0.22, warmth=0.012,
                  clarity=0.10, sharpen=0.30):
    """Deterministic tone/style finish — the AREM drone look as global tone +
    color + local-CONTRAST only. Warmth (R/B trim), dehaze (mild CLAHE blend on
    L), contrast, vibrance, then two unsharp passes (clarity @sigma4, sharpen
    @sigma1). Every spatial op here ADDS sharpness; nothing blurs. This is the
    original v1 target the NAFNet was trained to imitate — used directly so the
    finish can't smear. Input + output are BGR uint8 at native resolution."""
    img = bgr.astype(np.float32) / 255.0
    img[..., 2] = np.clip(img[..., 2] * (1 + warmth), 0, 1)
    img[..., 0] = np.clip(img[..., 0] * (1 - warmth), 0, 1)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.addWeighted(L, 1 - dehaze, cv2.createCLAHE(1.0 + dehaze * 3, (8, 8)).apply(L), dehaze, 0)
    img = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    img = np.clip((img - 0.5) * (1 + contrast) + 0.5, 0, 1)
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    hsv[..., 1] = np.clip((s + vibrance * (1 - s) * s) * 255, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    img = np.clip(img + clarity * (img - cv2.GaussianBlur(img, (0, 0), 4)), 0, 1)
    img = np.clip(img + sharpen * (img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    return (img * 255).astype(np.uint8)

def clarity(bgr, clar=0.18, sharp=0.45):
    """Unsharp clarity + sharpen pass — adds the dngoutput 'pop' after the NAFNet
    color/tone pass. Spatial but ONLY adds sharpness; cannot blur."""
    img = bgr.astype(np.float32) / 255.0
    img = np.clip(img + clar * (img - cv2.GaussianBlur(img, (0, 0), 4)), 0, 1)
    img = np.clip(img + sharp * (img - cv2.GaussianBlur(img, (0, 0), 1.0)), 0, 1)
    return (img * 255).astype(np.uint8)

def to_12mp(bgr):
    """Delivery resolution = 12 MP (matches every other AREM photo). Downscale
    (preserve aspect) if larger; leave as-is if already <=12 MP."""
    H, W = bgr.shape[:2]
    if H * W <= 12_000_000:
        return bgr
    s = (12_000_000 / (H * W)) ** 0.5
    return cv2.resize(bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)

def nafnet_finish(model, dev_bgr_12mp, device, grade_sigma=3.0):
    """Production drone finish (drone_finish_v2), DETAIL-PRESERVING.

    NAFNet is a dense conv net — applied directly it smooths fine micro-texture
    (shingle granules, foliage), losing detail vs the camera JPG even though the
    color grade is right. So we DON'T ship its pixels. We take only its smooth,
    spatially-adaptive COLOR GRADE — low-pass(NAFNet(develop) - develop) — and add
    that to the SHARP develop, then a deterministic clarity pass. Result: the
    develop's full native detail + the learned dngoutput grade + pop, with no
    smoothing. Output detail == develop detail by construction."""
    base = dev_bgr_12mp.astype(np.float32)
    x = torch.from_numpy(dev_bgr_12mp[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        graded = torch.clamp(model(x), 0, 1)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255.0
    grade_lf = cv2.GaussianBlur(graded - base, (0, 0), sigmaX=grade_sigma, sigmaY=grade_sigma)
    return clarity(np.clip(base + grade_lf, 0, 255).astype(np.uint8))

def grade_delta(model, dev_bgr, device):
    """LEGACY (smeared foliage; superseded by nafnet_finish). Run the model at
    WORK_LONG and return the residual grade delta upscaled to dev_bgr size."""
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

    # Finish mode. DEFAULT = gentle_finish (deterministic tone/style; can't blur).
    # The NAFNet model path is opt-in via DRONE_USE_NAFNET=1 — kept so a retrained,
    # better-constrained model can be A/B'd without a code change. The old model is
    # NOT the default because its residual-upscale smears foliage (see grade_delta).
    use_nafnet = os.environ.get("DRONE_USE_NAFNET", "").strip().lower() in ("1", "true", "yes", "on")
    model = None
    if use_nafnet and dngs:
        if not a.model or not os.path.isfile(a.model):
            print("  WARN DRONE_USE_NAFNET set but no model file — falling back to gentle_finish", flush=True)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ck = torch.load(a.model, map_location=device, weights_only=False)
            model = NAFNet(in_channels=3, out_channels=3, width=int(ck.get("width", 32)),
                           middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
                           use_residual=True, residual_start=int(ck.get("residual_start", 0))).to(device).eval()
            model.load_state_dict(ck["model"])
            print(f"  DRONE_USE_NAFNET: model val_l1={ck.get('val_l1')} device={device}", flush=True)
    if not model:
        print("  finish = gentle_finish (deterministic tone/style, no blur)", flush=True)

    edited = passed = 0
    device = next(model.parameters()).device.type if model else None
    for stem, p in sorted(dngs.items()):
        try:
            dev = develop(p)
            if model:
                base = to_12mp(dev)                       # deliver at 12 MP
                out = nafnet_finish(model, base, device)  # NAFNet v2 color + clarity, single pass
                # Safety net: never ship a frame materially blurrier than its
                # develop — fall back to the clean 12 MP develop if it regresses.
                if _sharpness(base) > 0 and _sharpness(out) < 0.6 * _sharpness(base):
                    print(f"  GUARD {stem}: edit blurrier than develop — shipping develop", flush=True)
                    out = base
            else:
                out = gentle_finish(dev.astype(np.uint8))
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
