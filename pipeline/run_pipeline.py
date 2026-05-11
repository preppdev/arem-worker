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


# Accept Sony ARW, Nikon NEF, Canon CR2/CR3 (+ generic DNG/RAF/RW2) and
# standard image formats (JPG/JPEG/TIFF/PNG/JXL/WEBP). Filename prefix is
# permissive: matches DSC#####, DSC_####, IMG_####, _MG_####, etc — the
# bracket key is the trailing 4–5 digits, which all three vendors use as
# the camera-side frame counter. All bracket frames within a single shoot
# must share a format — we don't fuse e.g. one NEF with two JPGs.
ARW_RE = re.compile(
    r"^[A-Z_]*?(\d{4,5})\."
    r"(ARW|NEF|CR2|CR3|DNG|RAF|RW2|JPG|JPEG|TIF|TIFF|PNG|JXL|WEBP)$",
    re.IGNORECASE,
)
MAX_MP = float(os.environ.get("MAX_MP_OVERRIDE", "12.0"))


# -------- bracket grouping --------

# Recognized image extensions (raw + standard). Files outside this set are
# ignored at listing time.
ACCEPTED_EXTS = {".arw", ".nef", ".cr2", ".cr3", ".dng", ".raf", ".rw2",
                 ".jpg", ".jpeg", ".tif", ".tiff", ".png", ".jxl", ".webp"}


def list_arws(shoot_raw_dir: Path) -> list[Path]:
    """Return image files for triplet detection.

    Includes anything with a recognized image extension. When two files
    share the same DSC#-style numeric stem (e.g. "DSC02600.ARW" and
    "DSC02600.JPG"), we prefer the raw; otherwise both files pass
    through (find_triplets_exif handles deduplication by EXIF). Files
    with non-DSC filenames (e.g. T-739's "Photo May 04 2026, 11 28 25
    AM.arw" from the Lightroom Mobile export) used to be silently
    dropped by ARW_RE; they're now included.
    """
    by_dsc: dict[int, Path] = {}
    extras: list[Path] = []
    for p in shoot_raw_dir.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in ACCEPTED_EXTS:
            continue
        m = ARW_RE.match(p.name)
        if m:
            dsc = int(m.group(1))
            existing = by_dsc.get(dsc)
            if existing is None:
                by_dsc[dsc] = p
            else:
                existing_is_raw = existing.suffix.lower() in RAW_EXTENSIONS
                new_is_raw = ext in RAW_EXTENSIONS
                if new_is_raw and not existing_is_raw:
                    by_dsc[dsc] = p
        else:
            extras.append(p)
    return [by_dsc[k] for k in sorted(by_dsc)] + sorted(extras)


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


# Bracket-grouping rule: any 3 frames whose EVs form one mid (|EV| ≤
# ANCHOR_EV_ABS_MAX), one under (EV < 0), and one over (EV > 0) make a
# valid triplet, regardless of the in-camera shooting order. Sony AEB
# writes under→mid→over; Nikon AEB writes mid→under→over; we accept
# either. Frames once consumed by a triplet are not reused.
ANCHOR_EV_ABS_MAX = 1.0

# Two frames belong to the same bracket if their EXIF capture times are
# within this many seconds. Camera AEB fires its 3 or 5 frames within
# ~1 second; we leave headroom for the buffer-flush pause that some
# cameras insert between the 3rd and 4th of a 5-shot bracket.
MAX_BRACKET_GAP_SEC = 3.0


def find_triplets(arws: list[Path]) -> tuple[list[tuple[Path, Path, Path]], list[dict]]:
    """Detect bracket triplets. Dispatches between two algorithms:

    - 'exif' (default): EXIF-time-driven clustering with EV-role
      selection. Handles non-DSC filenames (Lightroom Mobile,
      Dropbox-renamed "(1)" files), 5-frame AEB (picks the central
      -2/0/+2 sub-triplet), and duplicate uploads by deduping on
      (capture_time_with_subsec, EV).

    - 'dsc' (legacy fallback): the original DSC#-sequenced 3-frame
      sliding-window detector. Available via FIND_TRIPLETS_MODE=dsc
      env var in case the exif version misbehaves.

    Both return (triplets, failures) with the same shape.
    """
    mode = os.environ.get("FIND_TRIPLETS_MODE", "exif").lower()
    if mode == "dsc":
        return find_triplets_dsc(arws)
    return find_triplets_exif(arws)


def find_triplets_dsc(arws: list[Path]) -> tuple[list[tuple[Path, Path, Path]], list[dict]]:
    """Legacy DSC#-driven detector. Kept for fallback via
    FIND_TRIPLETS_MODE=dsc. Behavior unchanged from pre-2026-05-11.
    """
    triplets: list[tuple[Path, Path, Path]] = []
    failures: list[dict] = []
    if not arws:
        return triplets, failures

    # Index frames by counter (DSC# / IMG#); read EV once per frame.
    by_dsc: dict[int, tuple[Path, float | None]] = {}
    for a in arws:
        m = ARW_RE.match(a.name)
        if not m:
            continue
        dsc = int(m.group(1))
        ev, _ = read_ev_and_time(a)
        by_dsc[dsc] = (a, ev)

    sorted_dscs = sorted(by_dsc)
    consumed: set[int] = set()
    i = 0
    while i + 2 < len(sorted_dscs):
        d0, d1, d2 = sorted_dscs[i], sorted_dscs[i + 1], sorted_dscs[i + 2]
        if d0 in consumed:
            i += 1
            continue
        # Three consecutive counter values are required — gaps mean
        # frames were deleted or this isn't a contiguous bracket.
        if d1 != d0 + 1 or d2 != d1 + 1:
            i += 1
            continue
        f0, f1, f2 = by_dsc[d0], by_dsc[d1], by_dsc[d2]
        evs = [f0[1], f1[1], f2[1]]
        window_label = f"{f0[0].name}–{f2[0].name}"

        if any(e is None for e in evs):
            failures.append({"window": window_label, "evs": evs,
                             "reason": "EV missing on one or more frames"})
            i += 1
            continue

        # Identify roles by EV: most negative is under, most positive is
        # over, the remaining frame must be the mid (closest to 0).
        idx_under = min(range(3), key=lambda k: evs[k])
        idx_over = max(range(3), key=lambda k: evs[k])
        if idx_under == idx_over:
            i += 1
            continue
        idx_mid_set = {0, 1, 2} - {idx_under, idx_over}
        if len(idx_mid_set) != 1:
            i += 1
            continue
        idx_mid = idx_mid_set.pop()

        if evs[idx_under] >= 0:
            failures.append({"window": window_label, "evs": evs,
                             "reason": "no under-exposed frame in window"})
            i += 1
            continue
        if evs[idx_over] <= 0:
            failures.append({"window": window_label, "evs": evs,
                             "reason": "no over-exposed frame in window"})
            i += 1
            continue
        if abs(evs[idx_mid]) > ANCHOR_EV_ABS_MAX:
            failures.append({"window": window_label, "evs": evs,
                             "reason": f"mid frame EV {evs[idx_mid]} outside "
                                       f"±{ANCHOR_EV_ABS_MAX}"})
            i += 1
            continue

        frames = [f0[0], f1[0], f2[0]]
        triplets.append((frames[idx_under], frames[idx_mid], frames[idx_over]))
        consumed.update([d0, d1, d2])
        i += 3

    return triplets, failures


def find_triplets_exif(arws: list[Path]) -> tuple[list[tuple[Path, Path, Path]], list[dict]]:
    """EXIF-driven triplet detection.

    Reads (capture_time_with_subsec, EV, file_size) per file, dedupes
    exact duplicates by (time, EV), clusters surviving frames into
    time-bounded brackets, then picks one valid (under, mid, over)
    triplet per cluster — preferring frames closest to the EV-role
    boundaries so 5-frame {-4,-2,0,+2,+4} brackets emit the central
    (-2, 0, +2) sub-triplet instead of the (+4, 0, -4) wrap-around.

    Failures list each cluster that couldn't form a triplet with a
    plain-English reason.
    """
    triplets: list[tuple[Path, Path, Path]] = []
    failures: list[dict] = []
    if not arws:
        return triplets, failures

    # Phase 1: per-file metadata
    items: list[dict] = []
    missing: list[dict] = []
    for a in arws:
        ev, t = read_ev_and_time(a)
        try:
            size = a.stat().st_size
        except OSError:
            size = 0
        if ev is None or t is None:
            missing.append({
                "window": a.name,
                "evs": [ev],
                "reason": f"EXIF {'EV' if ev is None else 'DateTimeOriginal'} missing",
            })
            continue
        items.append({"path": a, "ev": ev, "t": t, "size": size, "name": a.name})

    # Phase 2: dedupe — identical (time, EV) means same shutter press
    # stored twice (e.g. re-uploaded card). Prefer raw over standard
    # JPG, then prefer the file whose name doesn't carry a Dropbox-style
    # " (1)" rename suffix.
    def dup_score(it: dict) -> tuple[int, int, int]:
        is_raw = it["path"].suffix.lower() in RAW_EXTENSIONS
        has_dup_suffix = 1 if re.search(r"\s\(\d+\)\.[A-Za-z0-9]+$", it["name"]) else 0
        # Higher tuple wins; we want raw > non-raw, no-suffix > suffix, larger > smaller
        return (1 if is_raw else 0, 1 - has_dup_suffix, it["size"])

    by_key: dict[tuple[float, float], dict] = {}
    dups_dropped = 0
    for it in items:
        key = (round(it["t"] * 1000) / 1000.0, round(it["ev"], 3))
        existing = by_key.get(key)
        if existing is None or dup_score(it) > dup_score(existing):
            if existing is not None:
                dups_dropped += 1
            by_key[key] = it
        else:
            dups_dropped += 1
    deduped = sorted(by_key.values(), key=lambda x: x["t"])
    if dups_dropped:
        print(f"  find_triplets_exif: dropped {dups_dropped} exact duplicate(s)", flush=True)

    # Phase 3: cluster into brackets by time gap
    clusters: list[list[dict]] = []
    cur: list[dict] = []
    for it in deduped:
        if cur and (it["t"] - cur[-1]["t"]) > MAX_BRACKET_GAP_SEC:
            clusters.append(cur)
            cur = []
        cur.append(it)
    if cur:
        clusters.append(cur)

    # Phase 4: pick a valid (under, mid, over) triplet per cluster.
    for cluster in clusters:
        window_label = f"{cluster[0]['name']}…{cluster[-1]['name']}"
        evs = [c["ev"] for c in cluster]
        if len(cluster) < 3:
            failures.append({"window": window_label, "evs": evs,
                             "reason": f"only {len(cluster)} frame(s) in bracket window "
                                       f"(need ≥ 3)"})
            continue
        mids = [c for c in cluster if abs(c["ev"]) <= ANCHOR_EV_ABS_MAX]
        unders = [c for c in cluster if c["ev"] <= -SIDE_MIN]
        overs = [c for c in cluster if c["ev"] >= SIDE_MIN]
        if not mids:
            failures.append({"window": window_label, "evs": evs,
                             "reason": f"no mid frame (|EV|≤{ANCHOR_EV_ABS_MAX}) in bracket"})
            continue
        if not unders:
            failures.append({"window": window_label, "evs": evs,
                             "reason": "no under-exposed frame in bracket"})
            continue
        if not overs:
            failures.append({"window": window_label, "evs": evs,
                             "reason": "no over-exposed frame in bracket"})
            continue
        # Mid: closest to 0. Under: closest to 0 from below (least extreme).
        # Over: closest to 0 from above. This naturally picks (-2, 0, +2)
        # from a 5-frame ±4/±2/0 bracket rather than the more-extreme ±4.
        mid = min(mids, key=lambda c: abs(c["ev"]))
        under = max(unders, key=lambda c: c["ev"])
        over = min(overs, key=lambda c: c["ev"])
        # Belt-and-suspenders: ensure the three are distinct
        chosen = {id(mid), id(under), id(over)}
        if len(chosen) != 3:
            failures.append({"window": window_label, "evs": evs,
                             "reason": "could not separate roles (same frame for two slots)"})
            continue
        triplets.append((under["path"], mid["path"], over["path"]))

    failures.extend(missing)
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
    """Load a Stage 2 checkpoint. Dispatches by `arch` field:
       - "nafnet": new may26 NAFNet w32 — uses arch hyperparams from ckpt
       - "restormer" or missing: legacy Restormer dim=48 (default)

    NAFNet weights are cast to bf16 in place — the activation memory at 12 MP
    fp32 single-pass exceeds the 48 GB A40/A6000 budget once Stage 1, classifier,
    and both Stage 2 models are resident. NAFNet has no MDTA softmax to break
    under half precision, so bf16 weights + bf16 activations is safe.
    """
    # Load to CPU to avoid temporarily holding the full ckpt (model + EMA + extras)
    # in GPU memory; only the picked state dict moves to GPU via load_state_dict.
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    arch = (ck.get("arch") or "restormer").lower()

    if arch == "nafnet":
        from models.nafnet import NAFNet
        model = NAFNet(
            in_channels=ck.get("in_channels", 3),
            out_channels=ck.get("out_channels", 3),
            width=ck.get("width", 32),
            middle_blk_num=ck.get("middle_blk_num", 12),
            enc_blk_nums=ck.get("enc_blk_nums", [2, 2, 4, 8]),
            dec_blk_nums=ck.get("dec_blk_nums", [2, 2, 2, 2]),
            use_residual=ck.get("use_residual", True),
            residual_start=ck.get("residual_start", 0),
        )
    else:
        model = RestormerStage2(
            in_channels=3, out_channels=3,
            dim=48, num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4, use_residual=False,
        )

    # Prefer EMA weights if present
    if ck.get("ema_state_dict"):
        ema = ck["ema_state_dict"]
        sd = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
    else:
        sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    model._arch = arch

    # Free the on-CPU ckpt dict ASAP (incl. any optimizer state in legacy ckpts)
    del ck, sd

    if arch == "nafnet":
        model = model.to(device).bfloat16()
    else:
        model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  loaded {ckpt.name} arch={arch} dtype={next(model.parameters()).dtype}", flush=True)
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

    arch = getattr(model, "_arch", "restormer")
    # Single-pass only (no tiling). NAFNet weights + activations run in bf16
    # (model already cast in load_stage2). Restormer stays fp32 — its MDTA
    # channel-softmax saturates under half precision and produces an
    # FFT-confirmed horizontal-stripe periodic texture.
    # Cap derives from MAX_MP (single-pass only, no tiling). +5% slack accounts
    # for the 16-pixel alignment rounding pushing dims slightly past the cap.
    cap_px = int(MAX_MP * 1.05 * 1024 * 1024)
    if h16 * w16 <= cap_px:
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        if arch == "nafnet":
            x = x.to(device, dtype=torch.bfloat16)
        else:
            x = x.to(device)
        with torch.no_grad():
            y = model(x)
        y = torch.clamp(y.float(), 0, 1)
        out = (y.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(out).save(out_path, quality=quality)
        del x, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return h16, w16

    raise RuntimeError(
        f"stage2: image {h16}×{w16}={h16*w16/1e6:.1f}MP exceeds MAX_MP={MAX_MP} cap "
        f"and tiling is disabled. Raise MAX_MP_OVERRIDE or revisit the cap."
    )


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
            print(f"    - {f['window']} (EVs={f['evs']}): {f['reason']}", flush=True)
        if len(group_failures) > 10:
            print(f"    ... and {len(group_failures)-10} more", flush=True)

    meta = {"raw_dir": str(raw_dir), "n_arws": len(arws),
            "n_triplets": len(triplets), "triplets": [],
            "group_failures": group_failures}
    n_int = n_ext = 0
    grand_t0 = time.time()

    # Reset CUDA peak after model loads so we measure inference-time peak,
    # which is the relevant number for "could we infer at higher resolution?"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
    if torch.cuda.is_available():
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        try:
            free, total = torch.cuda.mem_get_info()
            meta["gpu_total_gb"] = round(total / 1e9, 2)
        except Exception:
            pass
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
