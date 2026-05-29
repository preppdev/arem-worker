"""Inpaint sandbox worker — R2-queue daemon (no DB).

Polls an R2 prefix for pending sandbox requests written by the dashboard,
turns the user's bounding box into a tight SAM2 mask, runs the local
inpainters (LaMa + Flux Fill + ObjectClear) on that clean mask, and
writes results back to R2 for the dashboard to display.

Queue protocol (bucket = R2_BUCKET, default arem-training-data):

  inpaint-sandbox/requests/<id>.json
      {
        "id": "<id>",
        "sourceBucket": "arem-production-edit-jobs",
        "sourceR2Path": "tours/.../0011-DSC08157.jpg",
        "bbox": {"x":0.52,"y":0.31,"w":0.18,"h":0.22},   # fractional
        "methods": ["lama","flux_fill","object_clear"],
        "createdAt": "..."
      }

  inpaint-sandbox/results/<id>/done.json     (written when complete)
      {
        "id": "<id>", "status": "done"|"error",
        "maskR2Path": "...",
        "compositeR2Path": "...",
        "results": {"lama":"...","flux_fill":"...","object_clear":"..."},
        "error": null, "runtimeMs": 12345
      }

A request is considered claimed/finished once its done.json exists.

Run on the 3090:
    WORKER_TOKEN=... HF_TOKEN=$(cat ~/.cache/huggingface/token) \\
      python -m scripts.inpaint_sandbox_worker --loop
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

sys.path.insert(0, "/home/jordan/arem-worker")
from scripts.sam_box_mask import box_to_mask  # type: ignore
from scripts.inpaint_bakeoff import compute_roi, paste_with_feather, make_method  # type: ignore

RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
REQ_PREFIX = "inpaint-sandbox/requests"
RES_PREFIX = "inpaint-sandbox/results"
DEFAULT_METHODS = ["lama", "flux_fill", "object_clear"]


def log(m: str) -> None:
    print(m, flush=True)


def rclone(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True,
                          timeout=timeout)


def list_requests() -> list[str]:
    r = rclone(["lsf", f"{RCLONE_R2}:{R2_BUCKET}/{REQ_PREFIX}/",
                "--include", "*.json"])
    if r.returncode != 0:
        return []
    return [ln.strip()[:-5] for ln in r.stdout.splitlines()
            if ln.strip().endswith(".json")]


def result_done(req_id: str) -> bool:
    r = rclone(["lsf", f"{RCLONE_R2}:{R2_BUCKET}/{RES_PREFIX}/{req_id}/done.json"])
    return r.returncode == 0 and "done.json" in r.stdout


def fetch_json(key: str, dest: Path) -> dict | None:
    r = rclone(["copyto", f"{RCLONE_R2}:{R2_BUCKET}/{key}", str(dest)])
    if r.returncode != 0:
        return None
    try:
        return json.loads(dest.read_text())
    except Exception:
        return None


def put_file(local: Path, key: str, bucket: str = R2_BUCKET) -> bool:
    r = rclone(["copyto", str(local), f"{RCLONE_R2}:{bucket}/{key}"])
    return r.returncode == 0


def fetch_file(key: str, dest: Path, bucket: str) -> bool:
    r = rclone(["copyto", f"{RCLONE_R2}:{bucket}/{key}", str(dest)])
    return r.returncode == 0


def overlay(src: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    layer = np.zeros_like(src)
    layer[mask > 8] = color
    return cv2.addWeighted(src, 0.6, layer, 0.4, 0)


def build_composite(src, mask, results: dict[str, np.ndarray]) -> np.ndarray:
    H = 540
    panels = [("SOURCE", src), ("MASK", overlay(src, mask, (60, 200, 60)))]
    for name, im in results.items():
        panels.append((name.upper(), im))
    fitted = []
    for label, im in panels:
        h, w = im.shape[:2]
        fitted.append((label, cv2.resize(im, (int(w * H / h), H),
                                         interpolation=cv2.INTER_AREA)))
    gap = np.zeros((H, 8, 3), np.uint8)
    strip = fitted[0][1]
    for _, im in fitted[1:]:
        strip = np.concatenate([strip, gap, im], axis=1)
    bar = np.full((26, strip.shape[1], 3), 20, np.uint8)
    x = 0
    for label, im in fitted:
        cv2.putText(bar, label, (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (230, 230, 230), 1, cv2.LINE_AA)
        x += im.shape[1] + 8
    return np.concatenate([bar, strip], axis=0)


# Method instances are cached across requests so we don't reload the 12B
# FLUX weights for every box. Loaded lazily on first use.
_METHOD_CACHE: dict[str, object] = {}


def get_method(name: str):
    if name not in _METHOD_CACHE:
        m = make_method(name)
        m.load()
        _METHOD_CACHE[name] = m
    return _METHOD_CACHE[name]


def process(req: dict, scratch: Path) -> dict:
    rid = req["id"]
    t0 = time.time()
    src_bucket = req.get("sourceBucket", "arem-production-edit-jobs")
    src_key = req["sourceR2Path"]
    methods = req.get("methods") or DEFAULT_METHODS
    src_local = scratch / f"{rid}__src.jpg"
    if not fetch_file(src_key, src_local, src_bucket):
        return {"id": rid, "status": "error", "error": f"fetch source failed: {src_key}"}
    src = cv2.imread(str(src_local))
    if src is None:
        return {"id": rid, "status": "error", "error": "source decode failed"}

    # 1) SAM2 clean mask from the box.
    try:
        mask = box_to_mask(src, req["bbox"])
    except Exception as e:
        return {"id": rid, "status": "error", "error": f"sam mask failed: {e}"}
    mask_local = scratch / f"{rid}__mask.png"
    cv2.imwrite(str(mask_local), mask)
    mask_key = f"{RES_PREFIX}/{rid}/mask.png"
    put_file(mask_local, mask_key)

    # 2) Run each method on the clean mask via the ROI crop.
    roi = compute_roi(mask)
    x0, y0, x1, y1 = roi
    results_img: dict[str, np.ndarray] = {}
    result_keys: dict[str, str] = {}
    for name in methods:
        try:
            method = get_method(name)
            patch = method.run(src[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            out = paste_with_feather(src, patch, mask, roi)
        except Exception as e:
            log(f"  [{rid}] {name} failed: {e}")
            continue
        results_img[name] = out
        op = scratch / f"{rid}__{name}.jpg"
        cv2.imwrite(str(op), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        k = f"{RES_PREFIX}/{rid}/{name}.jpg"
        put_file(op, k)
        result_keys[name] = k

    # 3) Composite.
    comp = build_composite(src, mask, results_img)
    cp = scratch / f"{rid}__composite.jpg"
    cv2.imwrite(str(cp), comp, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    comp_key = f"{RES_PREFIX}/{rid}/composite.jpg"
    put_file(cp, comp_key)

    return {
        "id": rid,
        "status": "done" if result_keys else "error",
        "maskR2Path": mask_key,
        "compositeR2Path": comp_key,
        "results": result_keys,
        "error": None if result_keys else "all methods failed",
        "runtimeMs": int((time.time() - t0) * 1000),
    }


def write_done(rid: str, payload: dict, scratch: Path) -> None:
    dj = scratch / f"{rid}__done.json"
    dj.write_text(json.dumps(payload, indent=2))
    put_file(dj, f"{RES_PREFIX}/{rid}/done.json")


def poll_once(scratch: Path) -> int:
    ids = list_requests()
    n = 0
    for rid in ids:
        if result_done(rid):
            continue
        req = fetch_json(f"{REQ_PREFIX}/{rid}.json", scratch / f"{rid}__req.json")
        if not req:
            continue
        log(f"[sandbox] processing {rid} bbox={req.get('bbox')}")
        try:
            res = process(req, scratch)
        except Exception as e:
            res = {"id": rid, "status": "error", "error": str(e)}
        write_done(rid, res, scratch)
        log(f"[sandbox] {rid} -> {res['status']} ({res.get('runtimeMs')}ms)")
        n += 1
    return n


def unload_all() -> None:
    """Free every cached inpaint model from the GPU. Called after the
    daemon has been idle, so an always-on service doesn't permanently
    hold ~10 GB away from the production workers sharing this 3090."""
    if not _METHOD_CACHE:
        return
    for name, m in list(_METHOD_CACHE.items()):
        try:
            m.unload()  # type: ignore[attr-defined]
        except Exception:
            pass
    _METHOD_CACHE.clear()
    try:
        import torch  # type: ignore
        torch.cuda.empty_cache()
    except Exception:
        pass
    log("[sandbox] idle — unloaded models, GPU freed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="poll forever")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--idle-unload-sec", type=int,
                    default=int(os.environ.get("SANDBOX_IDLE_UNLOAD_SEC", "300")),
                    help="unload models from the GPU after this many idle "
                         "seconds (0 disables)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="arem-sandbox-") as sd:
        scratch = Path(sd)
        if not args.loop:
            poll_once(scratch)
            return 0
        log(f"[sandbox] polling for requests (idle-unload={args.idle_unload_sec}s)…")
        last_active = time.time()
        while True:
            try:
                did = poll_once(scratch)
            except Exception as e:
                log(f"[sandbox] poll error: {e}")
                did = 0
            now = time.time()
            if did > 0:
                last_active = now
            elif (args.idle_unload_sec > 0 and _METHOD_CACHE
                  and now - last_active > args.idle_unload_sec):
                unload_all()
            time.sleep(args.interval if did == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
