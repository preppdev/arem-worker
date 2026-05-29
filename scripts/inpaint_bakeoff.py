"""Inpainter bake-off — head-to-head comparison of LaMa, FLUX.1-Fill-dev,
OmniEraser, ObjectClear, FLUX-Kontext, and (optional) Adobe Firefly on the
same (image, mask) test set.

Test set source
---------------
We pull condition-tagged (image, vendor) pairs from the dashboard's
diff-pair-candidates endpoint, then look up an already-computed mask
from AutoMaskTest (best of `diff__v1` for reflection, `clipseg-rd64-
refined__unlit-fixture` for dead-fixture, etc.). The diff-derived mask
is what AutoHDR/vendor actually removed — exactly the right ground-truth
mask for "remove this object" evaluation.

Methods
-------
Each method module exposes ``run(image_bgr, mask_u8) -> result_bgr``.
The harness handles the rapidraw-inspired pad-and-crop trick:
- Crop a (mask + 1.5x padding, min 128 px) ROI around the mask
- Align to 64-pixel boundary
- Inpaint
- Paste back with a 4-px feather

Output
------
Per-frame composite JPG showing every method side by side, plus an HTML
index sorted by mask area descending.

Usage
-----
    python -m scripts.inpaint_bakeoff \\
        --methods lama,flux_fill,omni_eraser,object_clear \\
        --condition reflection \\
        --limit 50 \\
        --out /tmp/inpaint_bakeoff
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore


DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")


def log(msg: str) -> None:
    print(msg, flush=True)


# ─── Test set selection ────────────────────────────────────────────────────

# Best-quality mask source per condition (manually picked from the
# AutoMaskTest leaderboard — highest pixel count + visual sanity check).
MASK_MODEL_VERSION = {
    "reflection": "diff__v1",
    "dead-fixture": "clipseg-rd64-refined__unlit-fixture",
    "window": "owlv2-base+sam-vit-base",
}


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_candidates(condition: str, limit: int) -> list[dict]:
    sp = urllib.parse.urlencode({"condition": condition, "limit": str(limit * 4)})
    resp = _request("GET", f"/api/internal/diff-pair-candidates?{sp}")
    return resp.get("items") or []


def fetch_mask_index(condition: str, model_version: str) -> dict[str, dict]:
    """Returns imageR2Path -> {autoMaskR2Path, pixelCount} for the given
    (condition, modelVersion) — single batched call."""
    sp = urllib.parse.urlencode({
        "condition": condition,
        "modelVersion": model_version,
        "limit": "5000",
    })
    try:
        resp = _request("GET", f"/api/internal/auto-mask-rows?{sp}")
    except Exception as e:
        log(f"  mask index lookup error: {e}")
        return {}
    out: dict[str, dict] = {}
    for r in (resp.get("items") or []):
        ip = r.get("imageR2Path")
        if ip and r.get("autoMaskR2Path"):
            out[ip] = r
    return out


def rclone_fetch(r2_key: str, dest: Path, bucket: str = R2_BUCKET) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", f"{RCLONE_R2}:{bucket}/{r2_key}", str(dest),
         "--progress=false"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        log(f"  rclone fail: {r.stderr.strip()[:200]}")
        return False
    return True


def build_test_set(conditions: list[str], total: int,
                   stage_dir: Path) -> list[dict]:
    """Pull at most `total` (image, mask, vendor) triples across the given
    conditions. Round-robin among conditions for balance. Stages every
    file locally; returns rows with absolute paths."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    per_cond = max(1, total // max(1, len(conditions)))
    out: list[dict] = []
    for cond in conditions:
        mv = MASK_MODEL_VERSION.get(cond)
        if not mv:
            log(f"  no mask model registered for condition '{cond}', skipping")
            continue
        log(f"  pulling {cond} candidates …")
        cands = fetch_candidates(cond, per_cond * 4)
        mask_idx = fetch_mask_index(cond, mv)
        log(f"    mask index ({mv}): {len(mask_idx)} pre-computed masks")
        cond_count = 0
        for c in cands:
            if cond_count >= per_cond:
                break
            img_path = c["imageR2Path"]
            mask_row = mask_idx.get(img_path)
            if not mask_row:
                continue
            stem = Path(img_path).stem
            img_local = stage_dir / f"{cond}__{stem}__src.jpg"
            mask_local = stage_dir / f"{cond}__{stem}__mask.png"
            vendor_local = stage_dir / f"{cond}__{stem}__vendor.jpg"
            if not img_local.is_file():
                if not rclone_fetch(img_path, img_local,
                                    bucket="arem-production-edit-jobs"):
                    continue
            if not mask_local.is_file():
                if not rclone_fetch(mask_row["autoMaskR2Path"], mask_local,
                                    bucket="arem-training-data"):
                    continue
            if not vendor_local.is_file():
                if not rclone_fetch(c["vendorImageR2Path"], vendor_local,
                                    bucket="arem-training-data"):
                    pass  # vendor is optional for inference; reference for review
            row = {
                "condition": cond,
                "stem": stem,
                "src_path": img_local,
                "mask_path": mask_local,
                "vendor_path": vendor_local if vendor_local.is_file() else None,
                "mask_model": mv,
            }
            out.append(row)
            cond_count += 1
        log(f"    {cond}: collected {cond_count}")
        if len(out) >= total:
            break
    return out[:total]


# ─── ROI helpers (rapidraw-inspired) ──────────────────────────────────────

def align_64(x: int) -> int:
    return (x // 64) * 64


def ceil_64(x: int) -> int:
    return ((x + 63) // 64) * 64


def compute_roi(mask_u8: np.ndarray, pad_min_px: int = 128,
                pad_factor: float = 1.5) -> tuple[int, int, int, int]:
    """Bounding rect of mask, expanded by max(min_px, factor × mask_size),
    clamped to image bounds, then aligned to 64-px boundaries.
    Returns (x0, y0, x1, y1)."""
    h, w = mask_u8.shape[:2]
    ys, xs = np.where(mask_u8 > 8)
    if ys.size == 0:
        # empty mask — use full image
        return 0, 0, w, h
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    mh, mw = y1 - y0, x1 - x0
    pad = max(pad_min_px, int(pad_factor * max(mh, mw)))
    x0 = max(0, align_64(x0 - pad))
    y0 = max(0, align_64(y0 - pad))
    x1 = min(w, ceil_64(x1 + pad))
    y1 = min(h, ceil_64(y1 + pad))
    return x0, y0, x1, y1


def paste_with_feather(base: np.ndarray, patch: np.ndarray,
                       mask_u8: np.ndarray, roi: tuple[int, int, int, int],
                       feather_px: int = 4) -> np.ndarray:
    """Composite patch into base.copy() inside roi, soft-blending mask edges."""
    x0, y0, x1, y1 = roi
    out = base.copy()
    if patch.shape[:2] != (y1 - y0, x1 - x0):
        patch = cv2.resize(patch, (x1 - x0, y1 - y0))
    m = mask_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    if feather_px > 0:
        k = feather_px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    m3 = np.dstack([m, m, m])
    out[y0:y1, x0:x1] = (patch.astype(np.float32) * m3
                         + out[y0:y1, x0:x1].astype(np.float32) * (1 - m3)
                         ).clip(0, 255).astype(np.uint8)
    return out


# ─── Method dispatch ───────────────────────────────────────────────────────

class Method:
    name: str = ""
    backend: str = "diffusion"  # diffusion | onnx | api

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def run(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# Method classes live in scripts/inpaint_methods/*.py to keep this driver
# under 500 lines. Imported lazily so unused methods don't pay import cost.
def make_method(name: str) -> Method:
    n = name.lower()
    if n == "lama":
        from scripts.inpaint_methods.lama_onnx import LamaOnnx
        return LamaOnnx()
    if n == "flux_fill":
        from scripts.inpaint_methods.flux_fill_dev import FluxFillDev
        return FluxFillDev()
    if n == "omni_eraser":
        from scripts.inpaint_methods.omni_eraser import OmniEraser
        return OmniEraser()
    if n == "object_clear":
        from scripts.inpaint_methods.object_clear import ObjectClear
        return ObjectClear()
    if n == "flux_kontext":
        from scripts.inpaint_methods.flux_kontext import FluxKontext
        return FluxKontext()
    if n == "bria":
        from scripts.inpaint_methods.bria_erase import BriaErase
        return BriaErase()
    raise ValueError(f"unknown method: {name}")


# ─── Runner ────────────────────────────────────────────────────────────────

def run_method(method: Method, rows: list[dict], out_dir: Path) -> dict:
    """Loops `method` over all rows, writing per-row outputs to
    out_dir/<method.name>/{stem}.jpg. Returns timing summary."""
    method_dir = out_dir / method.name
    method_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{method.name}] loading …")
    t_load0 = time.time()
    method.load()
    t_load = time.time() - t_load0
    log(f"[{method.name}] loaded in {t_load:.1f}s; running {len(rows)} frames")
    per_frame_ms: list[float] = []
    failures = 0
    for i, row in enumerate(rows, 1):
        out_path = method_dir / f"{row['condition']}__{row['stem']}.jpg"
        if out_path.is_file():
            continue
        try:
            src = cv2.imread(str(row["src_path"]))
            mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if src is None or mask is None:
                failures += 1
                continue
            # Resize mask to source dims if necessary.
            if mask.shape[:2] != src.shape[:2]:
                mask = cv2.resize(mask, (src.shape[1], src.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            # ROI crop, inpaint, paste back.
            roi = compute_roi(mask)
            x0, y0, x1, y1 = roi
            src_crop = src[y0:y1, x0:x1]
            mask_crop = mask[y0:y1, x0:x1]
            t0 = time.time()
            patch = method.run(src_crop, mask_crop)
            per_frame_ms.append((time.time() - t0) * 1000)
            out = paste_with_feather(src, patch, mask, roi)
            cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        except Exception as e:
            log(f"  [{method.name}] {row['stem']} failed: {e}")
            failures += 1
        if i % 5 == 0:
            log(f"  {i}/{len(rows)} done")
    log(f"[{method.name}] unloading …")
    method.unload()
    return {
        "name": method.name,
        "load_s": round(t_load, 2),
        "per_frame_ms_mean": round(float(np.mean(per_frame_ms)), 1)
                              if per_frame_ms else None,
        "per_frame_ms_p95": round(float(np.percentile(per_frame_ms, 95)), 1)
                              if per_frame_ms else None,
        "failures": failures,
    }


def build_composite(rows: list[dict], methods: list[str],
                    out_dir: Path) -> None:
    """One JPG per row, with [src | mask | method1 | method2 | … | vendor].
    Saved to out_dir/composites/<condition>__<stem>.jpg."""
    comp_dir = out_dir / "composites"
    comp_dir.mkdir(parents=True, exist_ok=True)
    target_h = 540
    for row in rows:
        panels: list[tuple[str, np.ndarray]] = []
        src = cv2.imread(str(row["src_path"]))
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape[:2] != src.shape[:2]:
            mask = cv2.resize(mask, (src.shape[1], src.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if src is None:
            continue
        panels.append(("SOURCE", src))
        if mask is not None:
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_vis[mask > 8] = (40, 80, 200)  # red-orange overlay
            blend = cv2.addWeighted(src, 0.6, mask_vis, 0.4, 0)
            panels.append(("MASK", blend))
        for m in methods:
            mp = out_dir / m / f"{row['condition']}__{row['stem']}.jpg"
            if mp.is_file():
                p = cv2.imread(str(mp))
                panels.append((m.upper(), p))
        if row.get("vendor_path"):
            v = cv2.imread(str(row["vendor_path"]))
            panels.append(("VENDOR", v))
        # Resize all to common height, stack horizontally with a label bar.
        bar_h = 28
        fitted = []
        for label, im in panels:
            h, w = im.shape[:2]
            scale = target_h / h
            im_r = cv2.resize(im, (int(w * scale), target_h),
                              interpolation=cv2.INTER_AREA)
            fitted.append((label, im_r))
        gap = np.zeros((target_h, 8, 3), dtype=np.uint8)
        strip = fitted[0][1]
        for label, im in fitted[1:]:
            strip = np.concatenate([strip, gap, im], axis=1)
        bar = np.zeros((bar_h, strip.shape[1], 3), dtype=np.uint8)
        bar[:] = (20, 20, 20)
        x = 0
        for label, im in fitted:
            cv2.putText(bar, label, (x + 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)
            x += im.shape[1] + 8
        comp = np.concatenate([bar, strip], axis=0)
        out_path = comp_dir / f"{row['condition']}__{row['stem']}.jpg"
        cv2.imwrite(str(out_path), comp, [int(cv2.IMWRITE_JPEG_QUALITY), 84])


def build_html_index(rows: list[dict], methods: list[str],
                     timings: list[dict], out_dir: Path) -> None:
    # Sort rows by mask area descending (largest mask = hardest = most visible).
    def mask_area(r):
        m = cv2.imread(str(r["mask_path"]), cv2.IMREAD_GRAYSCALE)
        return int((m > 8).sum()) if m is not None else 0
    rows_sorted = sorted(rows, key=lambda r: -mask_area(r))
    methods_th = "".join(f"<th>{m}</th>" for m in methods)
    timing_html = "<table class='t'><tr><th>method</th><th>load</th>"\
                  "<th>mean ms/frame</th><th>p95 ms</th><th>failures</th></tr>"
    for t in timings:
        timing_html += (f"<tr><td>{t['name']}</td><td>{t['load_s']} s</td>"
                        f"<td>{t['per_frame_ms_mean']}</td>"
                        f"<td>{t['per_frame_ms_p95']}</td>"
                        f"<td>{t['failures']}</td></tr>")
    timing_html += "</table>"
    row_html = []
    for r in rows_sorted:
        comp = f"composites/{r['condition']}__{r['stem']}.jpg"
        row_html.append(
            f'<div class="row"><div class="meta">{r["condition"]} · '
            f'{r["stem"]}</div>'
            f'<img src="{comp}" loading="lazy" /></div>'
        )
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Inpainter bake-off</title>'
        '<style>'
        'body{background:#0c0c0c;color:#ddd;'
        'font-family:-apple-system,sans-serif;margin:0;padding:14px;}'
        '.wrap{max-width:1900px;margin:0 auto;}'
        '.row{margin-bottom:14px;border:1px solid #222;border-radius:6px;'
        'padding:8px;background:#141414;}'
        '.row img{display:block;width:100%;height:auto;border-radius:4px;}'
        '.meta{font-family:SF Mono,Menlo,monospace;font-size:12px;'
        'color:#aaa;margin-bottom:6px;}'
        'table.t{border-collapse:collapse;margin-bottom:18px;font-size:13px;}'
        'table.t th,table.t td{border:1px solid #333;padding:4px 10px;}'
        '</style></head><body><div class="wrap">'
        f'<h1>Inpainter bake-off — {len(rows)} frames × {len(methods)} methods</h1>'
        f'{timing_html}'
        '<div class="subtitle" style="color:#888;margin-bottom:18px;">'
        'Sorted by mask area descending (largest masks first = hardest cases).</div>'
        + "".join(row_html)
        + '</div></body></html>'
    )
    (out_dir / "index.html").write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods",
                    default="lama,flux_fill,omni_eraser,object_clear,flux_kontext",
                    help="Comma-separated list of methods to evaluate.")
    ap.add_argument("--conditions", default="reflection,dead-fixture",
                    help="Comma-separated condition keys to draw test frames from.")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="/tmp/inpaint_bakeoff")
    ap.add_argument("--stage-dir", default="/tmp/inpaint_bakeoff_stage")
    ap.add_argument("--no-build", action="store_true",
                    help="Skip building composites + HTML (just run methods).")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required"); return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(args.stage_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    log("=== Building test set ===")
    rows = build_test_set(conditions, args.limit, stage_dir)
    log(f"test set: {len(rows)} (image, mask, vendor?) triples")
    if not rows:
        log("ERROR: empty test set"); return 3
    (out_dir / "manifest.json").write_text(json.dumps(
        [{**r, "src_path": str(r["src_path"]),
          "mask_path": str(r["mask_path"]),
          "vendor_path": str(r["vendor_path"]) if r.get("vendor_path") else None}
         for r in rows], indent=2))

    log(f"\n=== Running {len(methods)} methods ===")
    timings: list[dict] = []
    for name in methods:
        try:
            method = make_method(name)
        except Exception as e:
            log(f"[{name}] init error: {e}")
            timings.append({"name": name, "load_s": None,
                            "per_frame_ms_mean": None,
                            "per_frame_ms_p95": None, "failures": -1})
            continue
        t = run_method(method, rows, out_dir)
        timings.append(t)
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))

    if not args.no_build:
        log("\n=== Composites + HTML ===")
        build_composite(rows, methods, out_dir)
        build_html_index(rows, methods, timings, out_dir)
        log(f"  index.html: {out_dir / 'index.html'}")

    log("\n=== Done ===")
    for t in timings:
        log(f"  {t['name']:14s} mean={t.get('per_frame_ms_mean')} ms  "
            f"p95={t.get('per_frame_ms_p95')} ms  failures={t.get('failures')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
