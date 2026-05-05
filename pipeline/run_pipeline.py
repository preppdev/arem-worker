"""End-to-end Stage 1 + Stage 2 inference for one shoot.

RAW (3 ARWs per triplet) → rawpy demosaic → lensfun correction →
NAFNet Stage 1 (9-ch in, 3-ch out) → ResNet-18 classifier → Restormer
Stage 2 routed (interior or exterior) → JPG.

Adapted from /home/jordan/sam_reflection_tool/run_pipeline_test.py to
process one shoot directory at a time, driven via CLI flags. Used by
the production worker.

Usage:
    python run_pipeline.py \
        --raw-dir /path/to/01-RAW-Photos \
        --output-dir /path/to/predictions \
        --stage1-ckpt /workspace/checkpoints/stage1_jxl_v1_best_lpips.pth \
        --interior-ckpt /workspace/checkpoints/interior_full_v1_latest.pth \
        --exterior-ckpt /workspace/checkpoints/exterior_full_v1_latest.pth \
        --classifier /workspace/pipeline/classifier_v2.pth
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import lensfunpy
import numpy as np
import rawpy
import torch
import torch.nn as nn
import torchvision.models as tvmodels
from PIL import Image
from torchvision import transforms

# Ensure the worker's package layout is importable (this file lives next to
# models/ and lens_correct.py).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from models.nafnet import NAFNet
from models.restormer import Restormer
from lens_correct import lens_correct_bracket, LensExcluded, load_bracket_frame, RAW_EXTENSIONS

import exifread

# ---- bracket EV classification (inlined from prep/local_stage1_prep.py) ----
MID_TOL = 0.5
SIDE_MIN = 1.0

def classify_ev(ev):
    if ev is None: return None
    if abs(ev) < MID_TOL: return "mid"
    if ev <= -SIDE_MIN: return "under"
    if ev >= SIDE_MIN: return "over"
    return None

def parse_ev(s):
    if not s: return None
    s = s.strip().lstrip("+")
    try: return float(s)
    except ValueError: return None


# Accept Sony ARW + standard image formats (JPG/JPEG/TIFF/PNG/JXL/WEBP).
# All bracket frames within a single shoot must be the same format —
# we don't fuse e.g. one ARW with two JPGs. Format is inferred from the
# extension of the first matching file.
ARW_RE = re.compile(r"^DSC(\d{4,5})\.(ARW|JPG|JPEG|TIF|TIFF|PNG|JXL|WEBP)$",
                    re.IGNORECASE)
MAX_MP = 12.0


# -------- bracket grouping --------

def list_arws(shoot_raw_dir: Path) -> list[Path]:
    """Return image files in DSC# order. Accepts ARW + standard formats.

    If both raw and standard files exist with the same DSC#, prefer the
    raw (rawpy + lensfun gives better quality). If only standard files
    exist, use those.
    """
    by_dsc: dict[int, Path] = {}
    for p in shoot_raw_dir.iterdir():
        m = ARW_RE.match(p.name)
        if not m:
            continue
        dsc = int(m.group(1))
        ext = p.suffix.lower()
        existing = by_dsc.get(dsc)
        if existing is None:
            by_dsc[dsc] = p
        else:
            # Prefer raw over standard for same DSC#.
            existing_is_raw = existing.suffix.lower() in RAW_EXTENSIONS
            new_is_raw = ext in RAW_EXTENSIONS
            if new_is_raw and not existing_is_raw:
                by_dsc[dsc] = p
    return [by_dsc[k] for k in sorted(by_dsc)]


def read_ev_and_time(arw: Path):
    """Returns (ev_float | None, shutter_time_seconds_float | None).

    `shutter_time_seconds_float` is DateTimeOriginal converted to seconds since
    epoch (using SubSecTimeOriginal for sub-second precision), used as the
    bracket-grouping anchor — DSC# order is unreliable when the photographer
    has stray frames from earlier in the day with lower numbers.
    """
    import datetime as _dt
    with arw.open("rb") as fh:
        tags = exifread.process_file(fh, details=False, stop_tag="EXIF SubSecTimeOriginal")
    # EV
    ev = None
    v = tags.get("EXIF ExposureBiasValue")
    if v is not None and v.values:
        r = v.values[0]
        try:
            ev = r.num / r.den
        except (ZeroDivisionError, AttributeError):
            ev = parse_ev(str(v))
    # Shutter time
    t_sec = None
    dt_tag = tags.get("EXIF DateTimeOriginal")
    if dt_tag is not None:
        try:
            base = _dt.datetime.strptime(str(dt_tag), "%Y:%m:%d %H:%M:%S")
            sub = tags.get("EXIF SubSecTimeOriginal")
            sub_ms = 0.0
            if sub is not None:
                try:
                    sub_ms = float(f"0.{str(sub)}")
                except ValueError:
                    pass
            t_sec = base.timestamp() + sub_ms
        except (ValueError, TypeError):
            pass
    return ev, t_sec


def read_ev(arw: Path):
    """Back-compat shim — only used externally if anything else imports it."""
    return read_ev_and_time(arw)[0]


# Mid-frame anchor: an ARW whose EV bias is within ±1.0 stops of zero is
# treated as a candidate "mid" of a bracket. From there we check the
# immediate DSC# neighbors (anchor-1, anchor+1) — those must classify as
# under (negative EV) and over (positive EV) respectively for a valid
# triplet. No warmup frames, no fuzzy windows. If the immediate-neighbor
# rule doesn't hold, the anchor is flagged as a grouping failure for
# human review, never silently re-paired.
ANCHOR_EV_ABS_MAX = 1.0


def find_triplets(arws: list[Path]) -> tuple[list[tuple[Path, Path, Path]], list[dict]]:
    """Return (triplets, failures).

    triplets — list of (under, mid, over) tuples for valid brackets.
    failures — list of {"anchor", "reason"} dicts for mid candidates
                that couldn't be paired. Surfaced in _meta.json + the
                job's errorMessage so humans can fix data hygiene.
    """
    triplets: list[tuple[Path, Path, Path]] = []
    failures: list[dict] = []
    if not arws:
        return triplets, failures

    # Index ARWs by DSC number; read EV once per ARW.
    by_dsc: dict[int, tuple[Path, float | None]] = {}
    for a in arws:
        m = ARW_RE.match(a.name)
        if not m:
            continue
        dsc = int(m.group(1))
        ev, _ = read_ev_and_time(a)
        by_dsc[dsc] = (a, ev)

    for dsc in sorted(by_dsc):
        path, ev = by_dsc[dsc]
        # Anchor candidate: EV in [-ANCHOR_EV_ABS_MAX, +ANCHOR_EV_ABS_MAX]
        if ev is None or abs(ev) > ANCHOR_EV_ABS_MAX:
            continue

        prev = by_dsc.get(dsc - 1)
        nxt = by_dsc.get(dsc + 1)
        if prev is None and nxt is None:
            failures.append({"anchor": path.name, "ev": ev,
                             "reason": "no immediate DSC# neighbors present"})
            continue
        if prev is None:
            failures.append({"anchor": path.name, "ev": ev,
                             "reason": f"missing DSC{dsc-1} (under candidate)"})
            continue
        if nxt is None:
            failures.append({"anchor": path.name, "ev": ev,
                             "reason": f"missing DSC{dsc+1} (over candidate)"})
            continue

        prev_path, prev_ev = prev
        nxt_path, nxt_ev = nxt
        if prev_ev is None or nxt_ev is None:
            failures.append({"anchor": path.name, "ev": ev,
                             "reason": f"neighbor EV unreadable "
                                       f"(prev={prev_ev}, next={nxt_ev})"})
            continue
        if not (prev_ev < 0):
            failures.append({"anchor": path.name, "ev": ev,
                             "reason": f"DSC{dsc-1} not under (EV={prev_ev})"})
            continue
        if not (nxt_ev > 0):
            failures.append({"anchor": path.name, "ev": ev,
                             "reason": f"DSC{dsc+1} not over (EV={nxt_ev})"})
            continue

        triplets.append((prev_path, path, nxt_path))

    return triplets, failures


# -------- Stage 1 (NAFNet) --------

def load_stage1(ckpt: Path, device):
    model = NAFNet(in_channels=9, out_channels=3, width=16,
                   middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                   dec_blk_nums=[2, 2, 2, 2], use_residual=True,
                   residual_start=3).to(device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = ck.get("ema_state_dict")
    if isinstance(sd, dict) and "shadow" in sd:
        sd = sd["shadow"]
    if not sd:
        sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  Stage 1 loaded ({len(sd)} keys) ep={ck.get('epoch')} "
          f"lpips={ck.get('metrics',{}).get('lpips','?')}", flush=True)
    return model


def lens_correct_16(arw: Path, db) -> tuple[np.ndarray, dict]:
    """Decode + (if raw) lens-correct, returning uint16 RGB.

    Routes through load_bracket_frame() which handles both raw paths
    (rawpy + lensfun) and standard image paths (PIL decode + 8→16 bit
    promotion).
    """
    rgb, info = load_bracket_frame(arw, out_bps=16, db=db)
    if rgb.dtype != np.uint16:
        rgb = rgb.astype(np.uint16) << 8 if rgb.dtype == np.uint8 else rgb.astype(np.uint16)
    return rgb, info


def resize_max_mp(arr: np.ndarray, max_mp: float) -> np.ndarray:
    h, w = arr.shape[:2]
    mp = (h * w) / 1_000_000
    if mp > max_mp:
        scale = (max_mp / mp) ** 0.5
        nh, nw = int(h * scale), int(w * scale)
    else:
        nh, nw = h, w
    nh = (nh // 16) * 16
    nw = (nw // 16) * 16
    if (nh, nw) != arr.shape[:2]:
        arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    return arr


def stage1_infer(model, under16, mid16, over16, device) -> np.ndarray:
    h = min(under16.shape[0], mid16.shape[0], over16.shape[0])
    w = min(under16.shape[1], mid16.shape[1], over16.shape[1])
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    under16 = under16[:h16, :w16]
    mid16 = mid16[:h16, :w16]
    over16 = over16[:h16, :w16]
    arr = np.concatenate(
        [under16.astype(np.float32) / 65535.0,
         mid16.astype(np.float32) / 65535.0,
         over16.astype(np.float32) / 65535.0],
        axis=2,
    )
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(x)
    y = torch.clamp(y, 0, 1)
    out = (y.squeeze(0).float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# -------- Stage 2 (Restormer routed) --------

class RestormerStage2(Restormer):
    def forward(self, x):
        residual_input = x
        inp = self.patch_embed(x)
        e1 = self.encoder_1(inp)
        e2 = self.encoder_2(self.down_1(e1))
        e3 = self.encoder_3(self.down_2(e2))
        b = self.bottleneck(self.down_3(e3))
        d3 = self.decoder_3(self.reduce_3(torch.cat([self.up_3(b), e3], dim=1)))
        d2 = self.decoder_2(self.reduce_2(torch.cat([self.up_2(d3), e2], dim=1)))
        d1 = self.decoder_1(self.reduce_1(torch.cat([self.up_1(d2), e1], dim=1)))
        out = self.output(self.refinement(d1))
        return torch.clamp(residual_input + out, 0, 1)


def load_stage2(ckpt: Path, device):
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    model = RestormerStage2(in_channels=3, out_channels=3,
                             dim=48, num_blocks=[4, 6, 6, 8],
                             num_refinement_blocks=4, use_residual=False).to(device)
    if ck.get("ema_state_dict"):
        ema = ck["ema_state_dict"]
        sd = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
    else:
        sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  loaded {ckpt.name} ({len(sd)} keys) ep={ck.get('epoch')}", flush=True)
    return model


def load_classifier(clf_ckpt: Path, device):
    state = torch.load(str(clf_ckpt), map_location=device, weights_only=False)
    label_names = state["label_names"]
    model = tvmodels.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, state["num_classes"])
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"  Classifier labels: {label_names} val_acc={state.get('val_acc','?')}", flush=True)
    return model, label_names, tfm


def classify(clf, label_names, tfm, jpg_path, device) -> tuple[bool, float, str]:
    im = Image.open(jpg_path).convert("RGB")
    x = tfm(im).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(clf(x), dim=1).cpu().numpy()[0]
    idx = int(prob.argmax())
    return label_names[idx] == "interior", float(prob[idx]), label_names[idx]


def stage2_infer(model, jpg_path: Path, out_path: Path, device,
                 tile=1024, overlap=64, quality=92) -> tuple[int, int]:
    img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    img = img[:h16, :w16]

    # 16 MP single-pass: A6000 (48 GB) handles 12-16 MP comfortably with the
    # _safe_depthwise_3x3 wrapper in models/restormer.py spatially chunking
    # any depthwise tensor that would exceed PyTorch's int32 indexing limit.
    # The old 8 MP threshold was for the 24 GB 3090; on serverless A6000 it
    # was forcing every 12 MP shoot through tiling and producing seam-pattern
    # artifacts that look like the "same tiling" Jordan saw.
    if h16 * w16 <= 16 * 1024 * 1024:
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model(x)
        y = torch.clamp(y, 0, 1)
        out = (y.squeeze(0).float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(out).save(out_path, quality=quality)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return h16, w16

    out = np.zeros((h16, w16, 3), dtype=np.float32)
    weight = np.zeros((h16, w16), dtype=np.float32)
    step = tile - overlap
    feather = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
        feather[:overlap] = ramp
        feather[-overlap:] = ramp[::-1]
    win = (feather[:, None] * feather[None, :]).astype(np.float32)
    ys = list(range(0, max(h16 - tile, 0) + 1, step))
    if not ys or ys[-1] != h16 - tile:
        ys.append(max(h16 - tile, 0))
    xs = list(range(0, max(w16 - tile, 0) + 1, step))
    if not xs or xs[-1] != w16 - tile:
        xs.append(max(w16 - tile, 0))
    for y0 in ys:
        for x0 in xs:
            patch = img[y0:y0 + tile, x0:x0 + tile]
            xt = torch.from_numpy(patch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
            xt = xt.to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yt = model(xt)
            yt = torch.clamp(yt, 0, 1)
            pred = yt.squeeze(0).float().cpu().permute(1, 2, 0).numpy()
            out[y0:y0 + tile, x0:x0 + tile] += pred * win[:, :, None]
            weight[y0:y0 + tile, x0:x0 + tile] += win
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    out /= np.maximum(weight[:, :, None], 1e-8)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(out_path, quality=quality)
    return h16, w16


# -------- main: process one shoot --------

def process_shoot(raw_dir: Path, out_dir: Path,
                  stage1_ckpt: Path, int_ckpt: Path, ext_ckpt: Path,
                  clf_ckpt: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_dir = out_dir / "stage1"
    s2_dir = out_dir / "stage2"
    s1_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    print("Loading models...", flush=True)
    s1 = load_stage1(stage1_ckpt, device)
    s2_int = load_stage2(int_ckpt, device)
    s2_ext = load_stage2(ext_ckpt, device)
    clf, label_names, clf_tfm = load_classifier(clf_ckpt, device)
    if torch.cuda.is_available():
        print(f"GPU mem after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB",
              flush=True)
    db = lensfunpy.Database()

    arws = list_arws(raw_dir)
    triplets, group_failures = find_triplets(arws)
    print(f"\n{raw_dir}: {len(arws)} ARWs → {len(triplets)} triplets", flush=True)
    if group_failures:
        print(f"  ⚠ {len(group_failures)} grouping failures:", flush=True)
        for f in group_failures[:10]:
            print(f"    - {f['anchor']} (EV={f['ev']}): {f['reason']}", flush=True)
        if len(group_failures) > 10:
            print(f"    ... and {len(group_failures)-10} more", flush=True)

    meta = {"raw_dir": str(raw_dir), "n_arws": len(arws),
            "n_triplets": len(triplets), "triplets": [],
            "group_failures": group_failures}
    n_int = n_ext = 0
    grand_t0 = time.time()

    for ti, (under, mid, over) in enumerate(triplets, 1):
        stem = mid.stem
        t0 = time.time()
        try:
            u16, ui = lens_correct_16(under, db)
            m16, mi = lens_correct_16(mid,   db)
            o16, oi = lens_correct_16(over,  db)
        except LensExcluded as e:
            print(f"  [{ti}/{len(triplets)}] {stem}: lens excluded — {e}", flush=True)
            meta["triplets"].append({"stem": stem, "skip": "lens_excluded"})
            continue
        except Exception as e:
            print(f"  [{ti}/{len(triplets)}] {stem}: demosaic fail: {e}", flush=True)
            meta["triplets"].append({"stem": stem, "skip": f"demosaic_fail:{e}"})
            continue
        t_demos = time.time()

        u16 = resize_max_mp(u16, MAX_MP)
        m16 = resize_max_mp(m16, MAX_MP)
        o16 = resize_max_mp(o16, MAX_MP)

        try:
            s1_out = stage1_infer(s1, u16, m16, o16, device)
        except Exception as e:
            print(f"  [{ti}/{len(triplets)}] {stem}: stage1 fail: {e}", flush=True)
            traceback.print_exc()
            meta["triplets"].append({"stem": stem, "skip": f"stage1_fail:{e}"})
            continue
        t_s1 = time.time()

        s1_path = s1_dir / f"{stem}_stage1.jpg"
        Image.fromarray(s1_out).save(s1_path, quality=95)

        is_int, conf, label = classify(clf, label_names, clf_tfm, s1_path, device)
        t_clf = time.time()
        route = "int" if is_int else "ext"

        s2_path = s2_dir / f"{stem}_stage2-{route}.jpg"
        try:
            stage2_infer(s2_int if is_int else s2_ext, s1_path, s2_path, device)
        except Exception as e:
            print(f"  [{ti}/{len(triplets)}] {stem}: stage2 fail: {e}", flush=True)
            traceback.print_exc()
            meta["triplets"].append({"stem": stem, "skip": f"stage2_fail:{e}",
                                      "route": route, "conf": conf})
            continue
        t_s2 = time.time()

        if is_int: n_int += 1
        else: n_ext += 1

        meta["triplets"].append({
            "stem": stem,
            "under": under.name, "mid": mid.name, "over": over.name,
            "lens_applied": ui["applied"], "lens_skip": ui.get("skip_reason"),
            "route": route, "confidence": round(conf, 3),
            "stage1_path": str(s1_path.relative_to(out_dir)),
            "stage2_path": str(s2_path.relative_to(out_dir)),
            "stage1_size": list(s1_out.shape[:2]),
            "timing_s": {
                "demos": round(t_demos - t0, 2),
                "s1": round(t_s1 - t_demos, 2),
                "clf": round(t_clf - t_s1, 2),
                "s2": round(t_s2 - t_clf, 2),
                "total": round(t_s2 - t0, 2),
            },
        })
        print(f"  [{ti}/{len(triplets)}] {stem} {route} ({conf:.2f}) "
              f"demos={t_demos-t0:.1f}s s1={t_s1-t_demos:.1f}s "
              f"s2={t_s2-t_clf:.1f}s tot={t_s2-t0:.1f}s",
              flush=True)

    meta["wall_min"] = round((time.time() - grand_t0) / 60, 1)
    meta["n_int"] = n_int
    meta["n_ext"] = n_ext
    with open(out_dir / "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nshoot done: {n_int} int + {n_ext} ext  wall: {meta['wall_min']} min",
          flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stage1-ckpt", type=Path,
                     default=Path(os.environ.get("CHECKPOINT_STAGE1",
                                                  "/workspace/checkpoints/stage1_jxl_v1_best_lpips.pth")))
    ap.add_argument("--interior-ckpt", type=Path,
                     default=Path(os.environ.get("CHECKPOINT_INTERIOR",
                                                  "/workspace/checkpoints/interior_full_v1_latest.pth")))
    ap.add_argument("--exterior-ckpt", type=Path,
                     default=Path(os.environ.get("CHECKPOINT_EXTERIOR",
                                                  "/workspace/checkpoints/exterior_full_v1_latest.pth")))
    ap.add_argument("--classifier", type=Path,
                     default=Path(os.environ.get("CLASSIFIER_PATH",
                                                  "/workspace/pipeline/classifier_v2.pth")))
    args = ap.parse_args()

    process_shoot(args.raw_dir, args.output_dir,
                  args.stage1_ckpt, args.interior_ckpt, args.exterior_ckpt,
                  args.classifier)


if __name__ == "__main__":
    main()
