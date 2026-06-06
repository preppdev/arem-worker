"""End-to-end Stage 1 + Stage 2 + Stage 3 inference for one shoot.

RAW (3 ARWs per triplet) → rawpy demosaic → lensfun correction →
NAFNet Stage 1 (9-ch in, 3-ch out) → ResNet-18 classifier → NAFNet
Stage 2 routed (interior or exterior) → NAFNet Stage 3 polish (routed,
optional) → JPG.

Stage 3 is a style-application step: a small NAFNet fine-tune that
nudges Stage-2 output toward a chosen target style. The exterior and
interior lanes carry independent ckpts; either side can be null, in
which case Stage 3 is a no-op for that route and the Stage-2 output is
emitted as the final JPG. The variant id (ckpt-parent dirname) is
recorded on each ImageReview row via worker.py.

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
        --polish-exterior-ckpt /workspace/checkpoints/polish_exterior_v1.pth \
        --polish-interior-ckpt /workspace/checkpoints/polish_interior_v1.pth \
        --classifier /workspace/pipeline/classifier_v2.pth
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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

# Safety-net upper bound on between-frame gap *within* a bracket.
# We don't use this as the primary cluster boundary anymore — that's
# done by EV-pattern detection (see Phase 3 in find_triplets_exif).
# This constant only fires when the gap is so long there's no way the
# frames belong to the same shutter-press sequence (photographer walked
# to the next room, sat down, anything). 60s is comfortably longer than
# even a 5-frame bracket with 2-second +EV exposures and write delay.
#
# Pre-2026-05-26 this was 3.0 and was the primary boundary, which
# silently dropped any bracket whose +3 EV frame took >3s to capture
# (long exposures in low-light interiors). T-1476 saw 6 of 18 triplets
# split this way, repeated through 3 watchdog makeups producing the
# same 12 JPGs each time. The EV-pattern split below is robust to that.
MAX_BRACKET_GAP_SEC = 60.0


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

    # Phase 3: cluster by EV-monotonicity (was time-gap pre-2026-05-26).
    # Rationale: a long-exposure +3 EV frame in low light can take >3s
    # after the mid frame, which used to split a single bracket into a
    # (-3, 0) pair + an orphan (+3) — both fail the ≥3 check. T-1476
    # lost 6 triplets this way; the 3 watchdog make-ups all produced
    # the same 12 JPGs from 54 ARWs.
    #
    # Sony AEB (and Nikon, and our entire fleet today) writes brackets
    # in strictly ascending EV order: 3-frame is (-EV, 0, +EV);
    # 5-frame is (-EV_max, -EV_mid, 0, +EV_mid, +EV_max). A new
    # bracket therefore announces itself as the first frame whose EV
    # is LESS THAN the previous frame's EV — that's the monotonicity
    # break.
    #
    # This is tighter than time-gap clustering in both directions:
    #   - within bracket: any length gap is fine as long as EV keeps
    #     ascending → fixes T-1476's slow-+3 problem
    #   - between brackets: detected exactly at the transition, even
    #     if brackets fire back-to-back with no perceptible time pause
    #
    # MAX_BRACKET_GAP_SEC stays as a safety net for pathological cases
    # (e.g. EXIF timestamps lost / corrupted on a subset of frames).
    #
    # Constraint: this assumes ascending in-camera order. Sony Mark II/
    # III and Nikon both do. If we ever add a camera that shoots
    # mid-first (0, -EV, +EV) we'll need to detect that per-shoot and
    # branch. Punt until we see it.
    clusters: list[list[dict]] = []
    cur: list[dict] = []
    for it in deduped:
        if cur:
            prev_ev = cur[-1]["ev"]
            monotonicity_break = it["ev"] < prev_ev
            time_safety_break  = (it["t"] - cur[-1]["t"]) > MAX_BRACKET_GAP_SEC
            if monotonicity_break or time_safety_break:
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


# -------- Stage 2 (NAFNet routed) --------

def load_stage2(ckpt: Path, device):
    """Load a NAFNet Stage 2 checkpoint. Arch hyperparams are read from
    the checkpoint (may26 ckpts carry them inline); we default to the
    may26 w=32 config when a key is missing.

    Weights are cast to bf16 in place — the activation memory at 12 MP
    fp32 single-pass exceeds the 48 GB A40/A6000 budget once Stage 1, the
    classifier, and both Stage 2 models are resident. NAFNet has no
    softmax attention so bf16 weights + bf16 activations are safe.
    """
    # Load to CPU to avoid temporarily holding the full ckpt (model + EMA
    # + extras) in GPU memory; only the picked state dict moves to GPU
    # via load_state_dict.
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)

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

    # Free the on-CPU ckpt dict ASAP
    del ck, sd

    model = model.to(device).bfloat16()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  loaded {ckpt.name} (NAFNet) dtype={next(model.parameters()).dtype}", flush=True)
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


# ─── Room classifier (TwoHead ResNet-18 — occupancy + 9-class room) ─────────
# Same architecture as scripts/enrich_scenes.py in the dashboard repo; we
# re-implement here to keep the worker self-contained (no cross-repo Python
# imports). Falls back to silently skipping if the checkpoint path isn't set
# or the file doesn't exist — older worker deployments without the room
# checkpoint just emit isInteriorWorker without roomTypeWorker.

class _RoomTwoHead(nn.Module):
    def __init__(self, n_rooms: int, n_occ: int):
        super().__init__()
        self.backbone = tvmodels.resnet18(weights=None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.occ_head = nn.Linear(feat, n_occ)
        self.room_head = nn.Linear(feat, n_rooms)

    def forward(self, x):
        f = self.backbone(x)
        return self.occ_head(f), self.room_head(f)


def load_room_classifier(ckpt_path: Path | None, device):
    """Return (model, room_types, occupancies, tfm, version) or None."""
    if not ckpt_path or not Path(ckpt_path).is_file():
        print(f"  room classifier: skip (no ckpt at {ckpt_path})", flush=True)
        return None
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    room_types = state["room_types"]
    occupancies = state["occupancies"]
    model = _RoomTwoHead(n_rooms=len(room_types), n_occ=len(occupancies))
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    version = state.get("version") or "v1"
    print(
        f"  Room classifier: {len(room_types)} rooms ({room_types}), "
        f"{len(occupancies)} occupancies ({occupancies}), ver={version}",
        flush=True,
    )
    return model, room_types, occupancies, tfm, version


def classify_room(room_model, jpg_path: Path, device) -> dict | None:
    """Run the room/occupancy classifier on `jpg_path`. Returns a dict with
    roomType/confidence/occupancy or None when no model is loaded."""
    if room_model is None:
        return None
    model, room_types, occupancies, tfm, version = room_model
    im = Image.open(jpg_path).convert("RGB")
    x = tfm(im).unsqueeze(0).to(device)
    with torch.no_grad():
        occ_logits, room_logits = model(x)
        room_probs = torch.softmax(room_logits, dim=1).cpu().numpy()[0]
        occ_probs = torch.softmax(occ_logits, dim=1).cpu().numpy()[0]
    r_idx = int(room_probs.argmax())
    o_idx = int(occ_probs.argmax())
    return {
        "roomType": room_types[r_idx],
        "roomConfidence": float(room_probs[r_idx]),
        "occupancy": occupancies[o_idx],
        "occupancyConfidence": float(occ_probs[o_idx]),
        "roomDistribution": {n: float(p) for n, p in zip(room_types, room_probs)},
        "version": version,
    }


# ─── 512px thumbnail + EXIF classification embed ────────────────────────────

def make_thumbnail(src: Path, dst: Path, max_long_edge: int = 512, quality: int = 88):
    """Save a long-edge-512 thumbnail JPG. Preserves aspect ratio."""
    im = Image.open(src).convert("RGB")
    im.thumbnail((max_long_edge, max_long_edge))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=quality)


_EXIFTOOL_CFG = Path(__file__).resolve().parent / "exiftool_arem.cfg"


def _find_exiftool() -> str | None:
    """Locate the exiftool binary. PATH first; then a few well-known
    install locations on the 3090. Returns None if nothing is found —
    callers degrade to the JSON UserComment fallback."""
    import shutil
    p = shutil.which("exiftool")
    if p:
        return p
    for cand in (
        "/home/jordan/.local/bin/exiftool",  # 3090 default
        "/usr/local/bin/exiftool",
        "/opt/homebrew/bin/exiftool",  # Mac dev box
    ):
        if Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def embed_classification_exif(jpg_path: Path, payload: dict) -> bool:
    """Bake the worker's room/interior classification into the JPG so it
    survives Dropbox sync, vendor handoff, and any tool that opens the
    file directly.

    Writes three layers (any one alone is sufficient for a downstream
    consumer; together they cover legacy + modern + custom workflows):

      1. XMP-arem:* — custom namespace declared in exiftool_arem.cfg.
         Highest-fidelity (typed boolean, real, strings) for our own
         tooling. Requires the cfg to read back cleanly.
      2. XMP-dc:Subject + IPTC:Keywords — multi-value keywords like
         ["interior", "kitchen"]. Lightroom / Bridge / Finder show these.
      3. EXIF UserComment — JSON blob, legacy compat. Any reader can
         pick up the raw verdict without exiftool/XMP knowledge.

    `payload` shape (from the call site in process_shoot):
      { isInterior: bool, roomType: str|None, roomConfidence: float|None,
        interiorConfidence: float, modelVersions: { room_classifier: str, ... },
        stage3Variant: str|None, classifiedAt: ISO8601 str }

    Returns True if exiftool succeeded; False if exiftool is missing or
    errored. The JSON UserComment fallback writes regardless (via piexif)
    so a missing exiftool binary doesn't lose the verdict entirely.
    """
    ok_exiftool = _write_xmp_iptc(jpg_path, payload)
    _write_user_comment_json(jpg_path, payload)
    return ok_exiftool


def _write_xmp_iptc(jpg_path: Path, payload: dict) -> bool:
    """Run exiftool to write the XMP-arem namespace + XMP-dc:Subject +
    IPTC:Keywords in one subprocess. -overwrite_original because we
    don't want the .original sidecar files cluttering output dirs."""
    exiftool_bin = _find_exiftool()
    if not exiftool_bin or not _EXIFTOOL_CFG.is_file():
        return False
    is_int = payload.get("isInterior")
    room = payload.get("roomType")
    room_conf = payload.get("roomConfidence")
    int_conf = payload.get("interiorConfidence")
    classifier_v = (payload.get("modelVersions") or {}).get("room_classifier")
    stage3_v = payload.get("stage3Variant")
    classified_at = payload.get("classifiedAt")

    # Build the keywords list. Always include interior|exterior so
    # filtering can group at that axis even when the room model
    # declined to assign a class (exteriors, low-confidence interiors).
    keywords: list[str] = []
    if is_int is not None:
        keywords.append("interior" if is_int else "exterior")
    if room:
        keywords.append(room)
    if stage3_v:
        keywords.append(f"polish:{stage3_v}")

    cmd: list[str] = [
        exiftool_bin, "-config", str(_EXIFTOOL_CFG),
        "-overwrite_original",
        # XMP-arem custom namespace
        f"-XMP-arem:isInterior={'true' if is_int else 'false'}"
        if is_int is not None else "-XMP-arem:isInterior=",
        f"-XMP-arem:roomType={room or ''}",
        f"-XMP-arem:roomConfidence={room_conf}" if room_conf is not None
            else "-XMP-arem:roomConfidence=",
        f"-XMP-arem:interiorConfidence={int_conf}" if int_conf is not None
            else "-XMP-arem:interiorConfidence=",
        f"-XMP-arem:classifierVersion={classifier_v or ''}",
        f"-XMP-arem:classifiedAt={classified_at or ''}",
        f"-XMP-arem:stage3Variant={stage3_v or ''}",
    ]
    # Multi-value tags: exiftool wants one -tag=value per item, with the
    # first one clearing prior content via -=. We avoid -= because most of
    # our outputs are fresh writes, so previous keywords aren't a concern.
    for kw in keywords:
        cmd += [f"-XMP-dc:Subject+={kw}", f"-IPTC:Keywords+={kw}"]
    cmd.append(str(jpg_path))

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _write_user_comment_json(jpg_path: Path, payload: dict) -> bool:
    """Legacy JSON-in-UserComment write. Kept for backward compat with
    any tool that reads the EXIF UserComment slot directly (the
    pre-XMP integration relied on it). Best-effort — piexif missing or
    a malformed source JPG returns False without raising."""
    try:
        import piexif  # type: ignore
    except Exception:
        return False
    try:
        existing = piexif.load(str(jpg_path))
    except Exception:
        existing = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    blob = b"ASCII\x00\x00\x00" + json.dumps(payload, separators=(",", ":")).encode("utf-8")
    existing.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = blob
    try:
        piexif.insert(piexif.dump(existing), str(jpg_path))
        return True
    except Exception:
        return False


def load_stage3(ckpt: Path, device):
    """Load a NAFNet Stage 3 (polish) checkpoint.

    Identical loading semantics to `load_stage2` — Stage 3 is the same
    arch (NAFNet w=32 by default), just fine-tuned on a different
    objective (pipeline-output → vendor/Lightroom-style target). Kept as
    a separate symbol so call sites stay self-documenting; thin wrapper
    over load_stage2 so we don't drift if the loader changes.
    """
    return load_stage2(ckpt, device)


def stage3_infer(model, jpg_path: Path, out_path: Path, device,
                 quality=92) -> tuple[int, int]:
    """Apply Stage 3 polish to a Stage-2 JPG.

    Same single-pass logic as stage2_infer — operates on a JPG read from
    disk, dimensions snapped to multiple of 16, bf16 inference, single
    pass under MAX_MP. Stage-2 output already obeys MAX_MP so there's
    no additional resize needed.
    """
    return stage2_infer(model, jpg_path, out_path, device, quality=quality)


def stage2_infer(model, jpg_path: Path, out_path: Path, device,
                 tile=1024, overlap=64, quality=92) -> tuple[int, int]:
    img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    img = img[:h16, :w16]

    # Single-pass only (no tiling). NAFNet weights + activations run in
    # bf16 (model already cast in load_stage2). Cap derives from MAX_MP;
    # +5% slack accounts for 16-pixel alignment rounding pushing dims
    # slightly past the cap.
    cap_px = int(MAX_MP * 1.05 * 1024 * 1024)
    if h16 * w16 <= cap_px:
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device, dtype=torch.bfloat16)
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
                  clf_ckpt: Path,
                  polish_int_ckpt: Path | None = None,
                  polish_ext_ckpt: Path | None = None,
                  skip_mid_stems: set[str] | None = None,
                  single_stems: set[str] | None = None) -> dict:
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
    # Stage 3 polish (optional, per-route). A missing ckpt for a lane
    # means Stage 3 is a no-op for that route — the Stage-2 output is
    # the final image. variant ids are the ckpt parent dirname, e.g.
    # "polish_exterior_v1", and surface on ImageReview.stage3Variant.
    s3_int = load_stage3(polish_int_ckpt, device) if polish_int_ckpt else None
    s3_ext = load_stage3(polish_ext_ckpt, device) if polish_ext_ckpt else None
    s3_int_variant = polish_int_ckpt.parent.name if polish_int_ckpt else None
    s3_ext_variant = polish_ext_ckpt.parent.name if polish_ext_ckpt else None
    if s3_int or s3_ext:
        print(f"  Stage 3 polish: ext={s3_ext_variant or 'off'} "
              f"int={s3_int_variant or 'off'}", flush=True)
    clf, label_names, clf_tfm = load_classifier(clf_ckpt, device)
    # Optional — gated by env CLASSIFIER_ROOM. If unset/missing the
    # worker just emits isInteriorWorker without roomTypeWorker.
    room_ckpt = os.environ.get("CLASSIFIER_ROOM")
    room_model = load_room_classifier(Path(room_ckpt) if room_ckpt else None, device)
    if torch.cuda.is_available():
        print(f"GPU mem after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB",
              flush=True)
    db = lensfunpy.Database()

    arws = list_arws(raw_dir)
    single_stems = single_stems or set()
    orphan_frames: list[dict] = []
    if single_stems:
        # Single-frame recovery (alternative Stage 1): develop exactly the
        # requested frames as singles, no bracketing. Used by the per-frame
        # orphan-recovery action.
        by_stem = {a.stem: a for a in arws}
        singles = [by_stem[s] for s in sorted(single_stems) if s in by_stem]
        not_found = sorted(s for s in single_stems if s not in by_stem)
        triplets, group_failures = [], []
        print(f"\n{raw_dir}: single-frame recovery — {len(singles)}/{len(single_stems)} "
              f"frame(s){' (not found: ' + ', '.join(not_found) + ')' if not_found else ''}",
              flush=True)
    else:
        triplets, group_failures = find_triplets(arws)
        singles = []
        print(f"\n{raw_dir}: {len(arws)} ARWs → {len(triplets)} triplets", flush=True)
        # Orphan capture: RAWs not consumed by any triplet — the candidates
        # the dashboard surfaces for single-frame recovery.
        used = {f.stem for (u, m, o) in triplets for f in (u, m, o)}
        orphan_frames = [{"stem": a.stem, "name": a.name}
                         for a in arws if a.stem not in used]
        if orphan_frames:
            print(f"  {len(orphan_frames)} orphan frame(s) (not in any bracket): "
                  f"{', '.join(o['stem'] for o in orphan_frames[:12])}"
                  f"{' …' if len(orphan_frames) > 12 else ''}", flush=True)

    # Makeup-job optimization: the worker passes skip_mid_stems for the
    # midStems that the parent already has finished JPGs for. Drop them
    # before the per-triplet inference loop so we don't waste GPU time
    # redoing them. The per-image POST step on the worker side ALSO
    # filters on the same set (defensive), but doing it here is what
    # actually saves wall-clock + GPU. Recorded as skip="already_done"
    # in meta so the worker still sees them in the manifest for any
    # auditing it does.
    if skip_mid_stems:
        kept = []
        skipped_stems = []
        for triplet in triplets:
            stem = triplet[1].stem  # mid frame
            if stem in skip_mid_stems:
                skipped_stems.append(stem)
            else:
                kept.append(triplet)
        if skipped_stems:
            print(f"  skip_mid_stems: dropping {len(skipped_stems)} already-done "
                  f"triplet(s) before inference (kept {len(kept)} for processing)",
                  flush=True)
        triplets = kept

    if group_failures:
        print(f"  ⚠ {len(group_failures)} grouping failures:", flush=True)
        for f in group_failures[:10]:
            print(f"    - {f['window']} (EVs={f['evs']}): {f['reason']}", flush=True)
        if len(group_failures) > 10:
            print(f"    ... and {len(group_failures)-10} more", flush=True)

    meta = {"raw_dir": str(raw_dir), "n_arws": len(arws),
            "n_triplets": len(triplets), "triplets": [],
            "group_failures": group_failures,
            "orphanFrames": orphan_frames,
            "n_singles": len(singles),
            "single_recovery": bool(single_stems),
            "skipped_stems": sorted(skip_mid_stems) if skip_mid_stems else []}
    n_int = n_ext = 0
    grand_t0 = time.time()

    # Reset CUDA peak after model loads so we measure inference-time peak,
    # which is the relevant number for "could we infer at higher resolution?"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Shared per-image downstream: classify → Stage 2 → Stage 3 → room →
    # EXIF → thumbnail → meta. Called by BOTH the triplet path (s1_out from
    # the Stage-1 HDR net) and the single-frame path (s1_out = the developed
    # RAW). Identical logic for both — only how s1_out is produced differs.
    def finish_image(idx, total, stem, s1_out, t0, t_demos, t_s1,
                     under, mid, over, single, lens_info):
        nonlocal n_int, n_ext
        tag = f"[{'single ' if single else ''}{idx}/{total}]"
        s1_path = s1_dir / f"{stem}_stage1.jpg"
        Image.fromarray(s1_out).save(s1_path, quality=95)

        is_int, conf, label = classify(clf, label_names, clf_tfm, s1_path, device)
        t_clf = time.time()
        route = "int" if is_int else "ext"

        s2_path = s2_dir / f"{stem}_stage2-{route}.jpg"
        try:
            stage2_infer(s2_int if is_int else s2_ext, s1_path, s2_path, device)
        except Exception as e:
            print(f"  {tag} {stem}: stage2 fail: {e}", flush=True)
            traceback.print_exc()
            meta["triplets"].append({"stem": stem, "skip": f"stage2_fail:{e}",
                                      "route": route, "conf": conf, "single": single})
            return
        t_s2 = time.time()

        # ── Stage 3 (polish) — overwrites the Stage 2 file in place. ───────
        s3_model = s3_int if is_int else s3_ext
        s3_variant = s3_int_variant if is_int else s3_ext_variant
        applied_stage3_variant: str | None = None
        if s3_model is not None:
            s3_path = s2_dir / f"{stem}_stage3-{route}.jpg"
            try:
                stage3_infer(s3_model, s2_path, s3_path, device)
                s3_path.replace(s2_path)
                applied_stage3_variant = s3_variant
            except Exception as e:
                print(f"  {tag} {stem}: stage3 fail (continuing with stage2 output): {e}", flush=True)
                traceback.print_exc()
                s3_path.unlink(missing_ok=True)
        t_s3 = time.time()

        if is_int: n_int += 1
        else: n_ext += 1

        room = classify_room(room_model, s2_path, device)
        t_room = time.time()

        cls_payload = {
            "isInterior": bool(is_int),
            "interiorClassifier": "classifier_v2",
            "interiorConfidence": round(conf, 4),
            "roomType": (room or {}).get("roomType"),
            "roomConfidence": round((room or {}).get("roomConfidence") or 0.0, 4) if room else None,
            "occupancy": (room or {}).get("occupancy"),
            "stage3Variant": applied_stage3_variant,
            "modelVersions": {
                # Single frames skip the HDR net — record that for provenance.
                "stage1": "single-develop" if single else stage1_ckpt.name,
                "stage2": (int_ckpt if is_int else ext_ckpt).name,
                "stage3": (polish_int_ckpt.name if is_int else polish_ext_ckpt.name)
                           if applied_stage3_variant else None,
                "interior_classifier": clf_ckpt.name,
                "room_classifier": (room or {}).get("version"),
            },
            "classifiedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        embed_classification_exif(s2_path, cls_payload)

        thumb_dir = out_dir / "_thumbnails"
        thumb_path = thumb_dir / f"{stem}.jpg"
        try:
            make_thumbnail(s2_path, thumb_path)
        except Exception as e:
            print(f"  {tag} {stem}: thumbnail fail: {e}", flush=True)
            thumb_path = None  # type: ignore

        meta["triplets"].append({
            "stem": stem,
            "under": under.name, "mid": mid.name, "over": over.name,
            "single": single,
            "lens_applied": lens_info["applied"], "lens_skip": lens_info.get("skip_reason"),
            "route": route, "confidence": round(conf, 3),
            "stage1_path": str(s1_path.relative_to(out_dir)),
            "stage2_path": str(s2_path.relative_to(out_dir)),
            "stage1_size": list(s1_out.shape[:2]),
            "classification": cls_payload,
            "thumbnail_path": (str(thumb_path.relative_to(out_dir)) if thumb_path else None),
            "timing_s": {
                "demos": round(t_demos - t0, 2),
                "s1": round(t_s1 - t_demos, 2),
                "clf": round(t_clf - t_s1, 2),
                "s2": round(t_s2 - t_clf, 2),
                "s3": round(t_s3 - t_s2, 2),
                "room": round(t_room - t_s3, 2),
                "total": round(t_room - t0, 2),
            },
        })
        s3_log = f" s3={t_s3-t_s2:.1f}s" if applied_stage3_variant else ""
        print(f"  {tag} {stem} {route} ({conf:.2f}) "
              f"demos={t_demos-t0:.1f}s s1={t_s1-t_demos:.1f}s "
              f"s2={t_s2-t_clf:.1f}s{s3_log} tot={t_s3-t0:.1f}s", flush=True)

    # ── Triplet path: Stage-1 HDR fusion of (under, mid, over) ───────────
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
        finish_image(ti, len(triplets), stem, s1_out, t0, t_demos, t_s1,
                     under, mid, over, single=False, lens_info=ui)

    # ── Single-frame path (alternative Stage 1): develop the one RAW. ────
    # lens_correct_16 already does WB + lens correction + sRGB gamma
    # (no_auto_bright), so the only "Stage 1" left is dropping 16→8 bit.
    for si, frame in enumerate(singles, 1):
        stem = frame.stem
        t0 = time.time()
        try:
            m16, mi = lens_correct_16(frame, db)
        except LensExcluded as e:
            print(f"  [single {si}/{len(singles)}] {stem}: lens excluded — {e}", flush=True)
            meta["triplets"].append({"stem": stem, "skip": "lens_excluded", "single": True})
            continue
        except Exception as e:
            print(f"  [single {si}/{len(singles)}] {stem}: demosaic fail: {e}", flush=True)
            meta["triplets"].append({"stem": stem, "skip": f"demosaic_fail:{e}", "single": True})
            continue
        t_demos = time.time()
        m16 = resize_max_mp(m16, MAX_MP)
        s1_out = (m16 >> 8).astype(np.uint8)  # 16-bit sRGB → 8-bit
        t_s1 = time.time()
        finish_image(si, len(singles), stem, s1_out, t0, t_demos, t_s1,
                     frame, frame, frame, single=True, lens_info=mi)

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
    # Stage 3 polish ckpts. Optional per-route — empty string disables
    # Stage 3 for that lane. Env vars CHECKPOINT_POLISH_INTERIOR /
    # CHECKPOINT_POLISH_EXTERIOR provide the worker default; the worker
    # resolves them from AppSetting `polish.ckpts` keyed by Job.polishStyle.
    def _opt_path(s: str | None) -> Path | None:
        return Path(s) if s else None
    ap.add_argument("--polish-interior-ckpt", type=_opt_path, default=None,
                     help="Optional Stage 3 polish ckpt for interior route; "
                          "omit or pass empty string to skip Stage 3 for interior.")
    ap.add_argument("--polish-exterior-ckpt", type=_opt_path, default=None,
                     help="Optional Stage 3 polish ckpt for exterior route; "
                          "omit or pass empty string to skip Stage 3 for exterior.")
    ap.add_argument("--classifier", type=Path,
                     default=Path(os.environ.get("CLASSIFIER_PATH",
                                                  "/workspace/pipeline/classifier_v2.pth")))
    # Make-up optimization: the worker passes the midStems the parent
    # already has finished JPGs for, so we can skip them at the
    # triplet-loop level (where the expensive Stage 1/2/3 work happens).
    # Comma-separated to keep the CLI simple. An empty string is a
    # legal no-op (full reprocess).
    ap.add_argument("--skip-mid-stems", type=str, default="",
                     help="Comma-separated midStems to skip in the "
                          "inference loop (used by make-up jobs to avoid "
                          "redoing already-finished triplets).")
    ap.add_argument("--single-stems", type=str, default="",
                     help="Comma-separated stems to develop as SINGLE frames "
                          "(alternative Stage 1 — no bracketing). When set, "
                          "ONLY these frames are processed (single-frame "
                          "orphan recovery).")
    args = ap.parse_args()
    skip_set = {s.strip() for s in args.skip_mid_stems.split(",") if s.strip()}
    single_set = {s.strip() for s in args.single_stems.split(",") if s.strip()}

    # Env fallbacks for polish ckpts — used when worker.py passes via env
    # rather than via flags. Worker prefers flags but env is the legacy
    # fallback for ad-hoc CLI runs.
    polish_int = args.polish_interior_ckpt or _opt_path(
        os.environ.get("CHECKPOINT_POLISH_INTERIOR") or None)
    polish_ext = args.polish_exterior_ckpt or _opt_path(
        os.environ.get("CHECKPOINT_POLISH_EXTERIOR") or None)

    process_shoot(args.raw_dir, args.output_dir,
                  args.stage1_ckpt, args.interior_ckpt, args.exterior_ckpt,
                  args.classifier,
                  polish_int_ckpt=polish_int,
                  polish_ext_ckpt=polish_ext,
                  skip_mid_stems=skip_set,
                  single_stems=single_set)


if __name__ == "__main__":
    main()
