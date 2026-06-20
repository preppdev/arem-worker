"""Standalone sky-swap worker.

Polls R2 sky-swap-jobs/requests/ for jobs queued by the production
pipeline (and any future ad-hoc enqueuers), runs the v9 sky-replace
pipeline on each image, uploads the result to BOTH:

  1. A new R2 key (so the dashboard can serve it via the image proxy)
  2. A Dropbox subfolder of the shoot's delivery path
     (`<shoot>/08-Test-Edit/sky-swap/`) for QC review.

Architecture mirrors the inpaint sandbox daemon: R2 control prefix for
restart + heartbeat, idle-unload of cached pipeline state, and the same
remote-restart protocol so the dashboard can manage it without SSH.

Request JSON shape (sky-swap-jobs/requests/<id>.json):
  {
    "id": "<uuid>",
    "sourceBucket": "arem-production-edit-jobs",
    "sourceR2Path": "tours/<jobId>/photos/<n>-<stem>.jpg",
    "plateBucket":  "arem-training-data",
    "plateR2Path":  "sky-library/<hash>.jpg",
    "dropboxJobPath": "AREM (Spiro Uploads)/2026/Q2/<photog>/<shoot>",
    "dropboxSubfolder": "08-Test-Edit/sky-swap",
    "stem":  "<midStem>",     # for the uploaded filename
    "jobId":  "<jobId>",
    "imageReviewId":  "<id>",
    "createdAt":  "..."
  }

done.json shape:
  {
    "id": "...", "status": "done"|"error",
    "primaryResultR2Path": "<new R2 key>",
    "dropboxPath": "<full dropbox destination path>",
    "skyImageId": "...",
    "runtimeMs": 12345,
    "error": null
  }
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
from scripts.sky_replace_v9_baseline import run_v9  # type: ignore

RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
REQ_PREFIX = "sky-swap-jobs/requests"
RES_PREFIX = "sky-swap-jobs/results"
CTRL_PREFIX = "sky-swap-jobs/control"
START_TS = time.time()

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")


def log(m: str) -> None:
    print(m, flush=True)


def post_stream_event(payload: dict) -> None:
    """Best-effort: tell the dashboard's master photo stream a swap was made,
    so it shows on the /photo-stream wall. Served from the Dropbox file."""
    if not WORKER_TOKEN:
        return
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{DASHBOARD_URL}/api/internal/stream-event",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
        )
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        log(f"[skyswap] stream-event post failed: {e}")


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


def upload_to_dropbox(local: Path, dropbox_full_path: str) -> bool:
    """`dropbox_full_path` = "AREM (Spiro Uploads)/.../08-Test-Edit/sky-swap/<file.jpg>"."""
    # Requests may pass dropboxJobPath WITH the rclone remote prefix already
    # attached ("dropbox:AREM (Spiro Uploads)/..."). Strip any leading
    # "dropbox:" so we do not build a "dropbox:dropbox:..." destination,
    # which rclone files under a literal top-level "dropbox:AREM ..." folder.
    p = dropbox_full_path
    while p.startswith("dropbox:"):
        p = p[len("dropbox:"):]
    r = rclone(["copyto", str(local), f"dropbox:{p}"])
    return r.returncode == 0


# ── Heartbeat / remote restart ────────────────────────────────────────

def _code_version() -> str:
    files = [
        "/home/jordan/arem-worker/scripts/sky_swap_worker.py",
        "/home/jordan/arem-worker/scripts/sky_replace_v9_baseline.py",
    ]
    mt = 0.0
    for f in files:
        try:
            mt = max(mt, os.path.getmtime(f))
        except OSError:
            pass
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mt))


def write_heartbeat(scratch: Path, *, loaded: bool) -> None:
    hb = scratch / "heartbeat.json"
    hb.write_text(json.dumps({
        "pid": os.getpid(),
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(START_TS)),
        "lastPollAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "codeVersion": _code_version(),
        "skysegLoaded": loaded,
    }))
    put_file(hb, f"{CTRL_PREFIX}/heartbeat.json")


def check_restart(scratch: Path) -> bool:
    rf = scratch / "restart.json"
    r = rclone(["copyto", f"{RCLONE_R2}:{R2_BUCKET}/{CTRL_PREFIX}/restart.json",
                str(rf)])
    if r.returncode != 0 or not rf.exists():
        return False
    try:
        req_ts = json.loads(rf.read_text()).get("requestedAtEpoch", 0)
    except Exception:
        req_ts = time.time()
    if req_ts <= START_TS:
        return False
    rclone(["deletefile", f"{RCLONE_R2}:{R2_BUCKET}/{CTRL_PREFIX}/restart.json"])
    return True


def reexec() -> None:
    log("[skyswap] restart requested — re-exec to reload code")
    os.chdir("/home/jordan/arem-worker")
    os.execv(sys.executable,
             [sys.executable, "-m", "scripts.sky_swap_worker",
              "--loop", "--interval", "5"])


def unload_pipeline() -> None:
    """Drop the cached skyseg ONNX session + free GPU memory."""
    import scripts.sky_replace_v9_baseline as v9  # type: ignore
    v9._SKYSEG = None  # type: ignore[attr-defined]
    try:
        import torch  # type: ignore
        torch.cuda.empty_cache()
    except Exception:
        pass
    log("[skyswap] idle — unloaded skyseg, GPU freed")


# ── Job processing ────────────────────────────────────────────────────

def process(req: dict, scratch: Path) -> dict:
    rid = req["id"]
    t0 = time.time()
    src_bucket = req.get("sourceBucket", "arem-production-edit-jobs")
    src_key = req["sourceR2Path"]
    plate_bucket = req.get("plateBucket", R2_BUCKET)
    plate_key = req.get("plateR2Path")
    if not plate_key:
        return {"id": rid, "status": "error", "error": "plateR2Path required"}

    src_local = scratch / f"{rid}__src.jpg"
    plate_local = scratch / f"{rid}__plate.jpg"
    if not fetch_file(src_key, src_local, src_bucket):
        return {"id": rid, "status": "error",
                "error": f"fetch source failed: {src_key}"}
    if not fetch_file(plate_key, plate_local, plate_bucket):
        return {"id": rid, "status": "error",
                "error": f"fetch plate failed: {plate_key}"}
    src = cv2.imread(str(src_local))
    plate = cv2.imread(str(plate_local))
    if src is None or plate is None:
        return {"id": rid, "status": "error", "error": "decode failed"}

    # ── Interior gate (stage 5 defense) ──────────────────────────────
    # Two sources of truth, in priority order:
    #   1. EXIF UserComment JSON (embedded by run_pipeline.py at stage 3,
    #      preserved through stage 4 auto_upright via piexif passthrough)
    #   2. req["isInterior"] (producer-supplied; auto-hook sets this from
    #      triplets, manual scripts may pass it explicitly)
    # Fail closed: if neither source identifies the frame as exterior,
    # refuse to composite. Interior frames produce false-positive sky
    # paint on bright windows/skylights/fixtures (see DSC07561, etc.).
    def _classification_from_exif(jpg: Path) -> dict | None:
        try:
            import piexif as _pe, json as _j
            ex = _pe.load(str(jpg))
            uc = ex.get("Exif", {}).get(_pe.ExifIFD.UserComment)
            if not uc: return None
            # Format: b"ASCII\x00\x00\x00" + json_bytes
            if isinstance(uc, bytes) and uc.startswith(b"ASCII\x00\x00\x00"):
                uc = uc[8:]
            return _j.loads(uc.decode("utf-8", errors="replace"))
        except Exception:
            return None

    _exif_cls = _classification_from_exif(src_local) or {}
    _is_interior = _exif_cls.get("isInterior")
    _source = "exif"
    if _is_interior is None:
        _is_interior = req.get("isInterior")
        _source = "request" if _is_interior is not None else None
    if _is_interior is True:
        return {"id": rid, "status": "skipped",
                "reason": "interior_frame",
                "isInterior": True, "source": _source}
    if _is_interior is None:
        return {"id": rid, "status": "skipped",
                "reason": "classification_unknown",
                "note": "no EXIF UserComment classification AND no isInterior in request"}

    # Deterministic per-frame seed for plate-fit randomization (extra
    # zoom + paired XY offset). Same frame always gets same composite;
    # different frames in the same shoot get different views of the
    # plate. Horizontal flip is NOT seed-driven — see flip_plate below.
    seed_basis = (req.get("imageReviewId") or req.get("stem")
                  or req.get("sourceR2Path") or rid)
    import hashlib as _h
    plate_seed = int.from_bytes(_h.sha256(str(seed_basis).encode()).digest()[:4], "big")

    # Horizontal flip is driven by exterior-subtype (front vs rear of
    # house) from the producer. Same perspective → same flip. The
    # producer (worker.py auto-hook or a manual enqueue script) reads
    # ImageReview.exteriorSubType and sets flipPlate accordingly.
    flip_plate = bool(req.get("flipPlate", False))

    # Vertical extent: fraction of source height the plate occupies,
    # anchored at the top. 0.5 (default) aligns the plate's horizon
    # gradient with the natural horizon line of a real-estate exterior.
    vertical_extent = float(req.get("verticalExtent", 0.5))

    try:
        alpha, result = run_v9(src, plate, plate_seed=plate_seed,
                               flip_plate=flip_plate,
                               vertical_extent=vertical_extent)
    except Exception as e:
        return {"id": rid, "status": "error", "error": f"v9: {e}"}

    op = scratch / f"{rid}__result.jpg"
    cv2.imwrite(str(op), result, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 1) Upload to a new R2 key beside the source (dashboard can serve it).
    ts = int(time.time())
    head, _, tail = src_key.rpartition(".")
    result_r2_key = f"{head}__skyswap_{ts}.{tail or 'jpg'}"
    if not put_file(op, result_r2_key, bucket=src_bucket):
        return {"id": rid, "status": "error", "error": "R2 upload failed"}

    # 2) Upload to Dropbox subfolder under the shoot's delivery folder.
    dropbox_full_path = None
    dropbox_ok = None
    dbx_root = req.get("dropboxJobPath")
    dbx_sub = req.get("dropboxSubfolder", "08-Test-Edit/sky-swap")
    stem = req.get("stem") or Path(src_key).stem
    if dbx_root:
        dropbox_full_path = f"{dbx_root}/{dbx_sub}/{stem}.jpg"
        dropbox_ok = upload_to_dropbox(op, dropbox_full_path)
        if dropbox_ok:
            post_stream_event({
                "kind": "skyswap",
                "jobId": req.get("jobId"),
                "label": stem,
                "dropboxPath": dropbox_full_path,
                "meta": {"alphaCoverage": float((alpha > 0.5).mean())},
            })

    return {
        "id": rid,
        "status": "done",
        "primaryResultR2Path": result_r2_key,
        "sourceR2Path": src_key,
        "sourceBucket": src_bucket,
        "plateR2Path": plate_key,
        "skyImageId": req.get("skyImageId"),
        "imageReviewId": req.get("imageReviewId"),
        "jobId": req.get("jobId"),
        "stem": stem,
        "dropboxPath": dropbox_full_path,
        "dropboxUploaded": dropbox_ok,
        "runtimeMs": int((time.time() - t0) * 1000),
        "alphaCoverage": float((alpha > 0.5).mean()),
        "createdAt": req.get("createdAt") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": None,
    }


def write_done(rid: str, payload: dict, scratch: Path) -> None:
    dj = scratch / f"{rid}__done.json"
    dj.write_text(json.dumps(payload, indent=2))
    put_file(dj, f"{RES_PREFIX}/{rid}/done.json")


def delete_request(rid: str) -> None:
    """Remove a request file from R2 once we have written done.json (or
    discovered an existing one). Without this the queue accumulates ghost
    entries forever: each poll re-lists the file, sees the done marker,
    skips processing — but never clears the request itself."""
    try:
        rclone(["deletefile", f"{RCLONE_R2}:{R2_BUCKET}/{REQ_PREFIX}/{rid}.json"])
    except Exception as e:
        log(f"[skyswap] failed to delete request {rid}: {e}")


def poll_once(scratch: Path) -> int:
    ids = list_requests()
    n = 0
    n_cleared = 0
    for rid in ids:
        if result_done(rid):
            delete_request(rid)
            n_cleared += 1
            continue
        req = fetch_json(f"{REQ_PREFIX}/{rid}.json", scratch / f"{rid}__req.json")
        if not req:
            continue
        log(f"[skyswap] processing {rid} src={req.get('sourceR2Path')}")
        try:
            res = process(req, scratch)
        except Exception as e:
            res = {"id": rid, "status": "error", "error": str(e)}
        write_done(rid, res, scratch)
        delete_request(rid)
        log(f"[skyswap] {rid} -> {res['status']} ({res.get('runtimeMs')}ms)")
        n += 1
    if n_cleared > 0:
        log(f"[skyswap] cleared {n_cleared} ghost requests (already had done.json)")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--idle-unload-sec", type=int,
                    default=int(os.environ.get("SKYSWAP_IDLE_UNLOAD_SEC", "300")))
    args = ap.parse_args()

    # Preflight self-check: refuse to run (and claim jobs) if the sky-seg
    # dependencies aren't present on THIS node. Without this, a node missing
    # onnxruntime or the model silently fails every frame it pulls from the
    # shared queue — ~60% of fleet sky-swaps were lost this way (Compute-2,
    # 2026-06-20) with no dashboard signal. Exit non-zero so systemd marks the
    # service FAILED and the breakage is visible. Set SKYSWAP_SKIP_PREFLIGHT=1
    # to bypass (e.g. a node intentionally not running swaps).
    if os.environ.get("SKYSWAP_SKIP_PREFLIGHT", "").lower() not in ("1", "true", "yes"):
        try:
            import onnxruntime  # noqa: F401  # type: ignore
        except Exception as e:
            log(f"[skyswap] PREFLIGHT FAIL: onnxruntime not importable ({e}). "
                f"`pip install onnxruntime==1.26.0` in this env. Exiting.")
            return 3
        from scripts.sky_replace_v9_baseline import SKYSEG_ONNX  # type: ignore
        if not Path(SKYSEG_ONNX).is_file():
            log(f"[skyswap] PREFLIGHT FAIL: sky model missing at {SKYSEG_ONNX}. "
                f"Copy it from a healthy node (~/sky_models/). Exiting.")
            return 4
        log("[skyswap] preflight OK (onnxruntime + sky model present)")

    with tempfile.TemporaryDirectory(prefix="arem-skyswap-") as sd:
        scratch = Path(sd)
        if not args.loop:
            poll_once(scratch)
            return 0
        log(f"[skyswap] polling for jobs (idle-unload={args.idle_unload_sec}s) …")
        last_active = time.time()
        last_hb = 0.0
        while True:
            try:
                if check_restart(scratch):
                    reexec()
            except Exception as e:
                log(f"[skyswap] restart-check error: {e}")
            try:
                did = poll_once(scratch)
            except Exception as e:
                log(f"[skyswap] poll error: {e}")
                did = 0
            now = time.time()
            if did > 0:
                last_active = now
            elif (args.idle_unload_sec > 0
                  and now - last_active > args.idle_unload_sec):
                try:
                    import scripts.sky_replace_v9_baseline as v9  # type: ignore
                    if v9._SKYSEG is not None:  # type: ignore[attr-defined]
                        unload_pipeline()
                except Exception:
                    pass
            if now - last_hb > 30:
                try:
                    import scripts.sky_replace_v9_baseline as v9  # type: ignore
                    write_heartbeat(scratch,
                                    loaded=v9._SKYSEG is not None)  # type: ignore[attr-defined]
                except Exception:
                    write_heartbeat(scratch, loaded=False)
                last_hb = now
            time.sleep(args.interval if did == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
