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
CTRL_PREFIX = "inpaint-sandbox/control"
DEFAULT_METHODS = ["lama"]
START_TS = time.time()


def _close_peninsulas(mask: np.ndarray) -> np.ndarray:
    """Morphological close to bridge narrow wall-gaps inside an object's
    silhouette (e.g. between a lamp's shade and its base, where SAM leaves
    the lamp neck unmasked). The kernel is sized to the object's smaller
    dimension and capped, so it closes peninsulas without ballooning out
    far enough to merge the object with an adjacent keep-object (a bed).
    """
    ys, xs = np.where(mask > 8)
    if ys.size == 0:
        return mask
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    k = int(0.25 * min(bw, bh))
    k = max(31, min(81, k)) | 1  # odd, clamped 31..81
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)


def box_mask(image_bgr: np.ndarray, bbox) -> np.ndarray:
    """Solid filled rectangle from a fractional {x,y,w,h} bbox."""
    h, w = image_bgr.shape[:2]
    x = bbox["x"]; y = bbox["y"]; bw = bbox["w"]; bh = bbox["h"]
    x0 = int(round(x * w)); y0 = int(round(y * h))
    x1 = int(round((x + bw) * w)); y1 = int(round((y + bh) * h))
    m = np.zeros((h, w), np.uint8)
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255
    return m


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

    # 0) Text-based erase (Bria erase_by_text): mask-free, the user just
    #    names the object. Branches out before any mask work.
    object_text = (req.get("objectText") or "").strip()
    if object_text:
        from scripts.inpaint_methods.bria_erase import erase_by_text
        key = os.environ.get("BRIA_API_KEY")
        if not key:
            return {"id": rid, "status": "error", "error": "BRIA_API_KEY not set"}
        try:
            out = erase_by_text(src, object_text, key)
        except Exception as e:
            return {"id": rid, "status": "error", "error": f"erase_by_text: {e}"}
        op = scratch / f"{rid}__bria_text.jpg"
        cv2.imwrite(str(op), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        res_key = f"{RES_PREFIX}/{rid}/bria_text.jpg"
        put_file(op, res_key)
        comp = build_composite(src, np.zeros(src.shape[:2], np.uint8),
                               {f'bria · "{object_text[:40]}"': out})
        cp = scratch / f"{rid}__composite.jpg"
        cv2.imwrite(str(cp), comp, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        comp_key = f"{RES_PREFIX}/{rid}/composite.jpg"
        put_file(cp, comp_key)
        return {
            "id": rid, "status": "done",
            "compositeR2Path": comp_key,
            "results": {"bria_text": res_key},
            "primaryResultR2Path": res_key,
            "sourceR2Path": src_key,
            "sourceBucket": src_bucket,
            "label": f'text·"{object_text[:40]}"',
            "createdAt": req.get("createdAt") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": None, "runtimeMs": int((time.time() - t0) * 1000),
        }

    # 1) Build the removal mask.
    #   maskMode="box" (default): the drawn rectangle, filled solid. Best
    #     paired with LaMa for objects against plain walls/floors — full
    #     coverage (incl. cords/legs) and LaMa can't invent furniture.
    #   maskMode="sam": SAM2 tight object silhouette. Pair with ObjectClear
    #     on complex/textured backgrounds where you need real synthesis.
    mask_mode = req.get("maskMode", "box")
    user_mask_key = req.get("maskR2Path")
    try:
        if user_mask_key:
            # Brush-refined mask painted in the dashboard. Authoritative —
            # skip auto-generation entirely.
            um_local = scratch / f"{rid}__usermask.png"
            if not fetch_file(user_mask_key, um_local, R2_BUCKET):
                return {"id": rid, "status": "error",
                        "error": f"fetch user mask failed: {user_mask_key}"}
            um = cv2.imread(str(um_local), cv2.IMREAD_GRAYSCALE)
            if um is None:
                return {"id": rid, "status": "error", "error": "user mask decode failed"}
            if um.shape[:2] != src.shape[:2]:
                um = cv2.resize(um, (src.shape[1], src.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            mask = (um > 127).astype(np.uint8) * 255
        elif mask_mode == "sam":
            mask = box_to_mask(src, req["bbox"])
            mask = _close_peninsulas(mask)
        else:
            mask = box_mask(src, req["bbox"])
    except Exception as e:
        return {"id": rid, "status": "error", "error": f"mask ({mask_mode}) failed: {e}"}
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

    primary = next(iter(result_keys.values()), None)
    return {
        "id": rid,
        "status": "done" if result_keys else "error",
        "maskR2Path": mask_key,
        "compositeR2Path": comp_key,
        "results": result_keys,
        "primaryResultR2Path": primary,
        "sourceR2Path": src_key,
        "sourceBucket": src_bucket,
        "label": f"{mask_mode}·{'+'.join(methods)}",
        "createdAt": req.get("createdAt") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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


def _code_version() -> str:
    """Max mtime across the worker's own source files — lets the dashboard
    tell whether a restart actually picked up newer code."""
    files = [
        "/home/jordan/arem-worker/scripts/inpaint_sandbox_worker.py",
        "/home/jordan/arem-worker/scripts/sam_box_mask.py",
        "/home/jordan/arem-worker/scripts/inpaint_methods/lama_onnx.py",
        "/home/jordan/arem-worker/scripts/inpaint_methods/object_clear.py",
    ]
    mt = 0.0
    for f in files:
        try:
            mt = max(mt, os.path.getmtime(f))
        except OSError:
            pass
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mt))


def write_heartbeat(scratch: Path, *, loaded: list[str]) -> None:
    hb = scratch / "heartbeat.json"
    hb.write_text(json.dumps({
        "pid": os.getpid(),
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(START_TS)),
        "lastPollAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "codeVersion": _code_version(),
        "modelsLoaded": loaded,
    }))
    put_file(hb, f"{CTRL_PREFIX}/heartbeat.json")


def check_restart(scratch: Path) -> bool:
    """If a restart was requested after we started, consume it and return
    True. The caller then re-execs the process so new code loads."""
    rf = scratch / "restart.json"
    r = rclone(["copyto", f"{RCLONE_R2}:{R2_BUCKET}/{CTRL_PREFIX}/restart.json",
                str(rf)])
    if r.returncode != 0 or not rf.exists():
        return False
    try:
        req_ts = json.loads(rf.read_text()).get("requestedAtEpoch", 0)
    except Exception:
        req_ts = time.time()  # malformed → treat as fresh
    if req_ts <= START_TS:
        return False  # stale signal from before this process started
    # Consume so we don't loop-restart.
    rclone(["deletefile", f"{RCLONE_R2}:{R2_BUCKET}/{CTRL_PREFIX}/restart.json"])
    return True


def reexec() -> None:
    """Replace this process with a fresh one — reloads all .py from disk.
    Inherits the current (systemd-provided) environment."""
    log("[sandbox] restart requested — re-exec to reload code")
    os.chdir("/home/jordan/arem-worker")
    os.execv(sys.executable,
             [sys.executable, "-m", "scripts.inpaint_sandbox_worker",
              "--loop", "--interval", "5"])


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
        last_hb = 0.0
        while True:
            # Remote control: restart (re-exec to reload code).
            try:
                if check_restart(scratch):
                    reexec()  # never returns
            except Exception as e:
                log(f"[sandbox] restart-check error: {e}")
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
            # Heartbeat every ~30s so the dashboard can show liveness.
            if now - last_hb > 30:
                try:
                    write_heartbeat(scratch, loaded=list(_METHOD_CACHE.keys()))
                except Exception as e:
                    log(f"[sandbox] heartbeat error: {e}")
                last_hb = now
            time.sleep(args.interval if did == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
