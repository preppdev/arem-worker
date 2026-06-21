#!/usr/bin/env python3
"""Single-image SCENE-REPROCESS worker.

A reviewer in the delivery gallery can re-run one already-delivered frame from
its Stage-1 output with the scene route FORCED (interior↔exterior) — used when
the auto-classifier sent a frame down the wrong lane (e.g. an exterior baked
with the interior look). This worker does the GPU + storage half; the dashboard
owns the DB half (it finalizes ImageReview when it sees the done marker).

Forcing the route means we skip the classifier entirely and run the exact
production Stage-2 → Stage-3 path on the chosen lane, at full resolution, so the
output matches a real deliverable. Reuses the production loaders + helpers from
`worker.py`/`pipeline.run_pipeline` so the look can't drift from production.

Queue (R2 arem-training-data):
  reprocess-jobs/requests/<id>.json
      { jobId, midStem, sortOrder, scene:"interior"|"exterior",
        stage1Key, productionKey, dropboxPath, polishStyle }
  reprocess-jobs/results/<id>.done.json
      { status:"done", cfImageId, productionKey, scene, stage3Variant, ms }
      | { status:"error", error }

The result JPEG is written straight to its production locations (production R2
key overwrite + Dropbox 08-Test-Edit/<midStem>.jpg overwrite + a fresh CF Images
upload). Dropbox 08 names are the bare <midStem>.jpg (upright strips the route
suffix), so a route flip is a clean overwrite — no rename/cleanup needed.
"""
import os
import sys
import time
import json
import tempfile
from pathlib import Path

import torch

BASE = Path(os.environ.get("AREM_WORKER_DIR", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(BASE))

# Production helpers — reuse the exact checkpoints, CF-Images upload, Dropbox
# path resolution, polish-ckpt resolver, and rclone wrapper the worker uses so
# a reprocess is byte-for-byte the same pipeline as the original delivery.
import worker  # noqa: E402
from pipeline.run_pipeline import load_stage2, load_stage3, stage2_infer, stage3_infer  # noqa: E402

R2 = os.environ.get("RCLONE_R2", "r2")
QUEUE_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
REQ = f"{R2}:{QUEUE_BUCKET}/reprocess-jobs/requests"
RES = f"{R2}:{QUEUE_BUCKET}/reprocess-jobs/results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Stage-3 polish models are loaded lazily and cached by local ckpt path so a
# burst of reprocesses on one style doesn't reload the same ~300 MB net.
_S3_CACHE: dict[str, object] = {}


def log(m):
    print(f"[reprocess] {m}", flush=True)


def rc(args, timeout=120):
    return worker.rclone(args, timeout=timeout)


def write_result(rid, payload):
    p = Path(tempfile.gettempdir()) / f"rpres_{rid}.json"
    p.write_text(json.dumps(payload))
    rc(["copyto", str(p), f"{RES}/{rid}.done.json"], timeout=30)
    try:
        p.unlink()
    except OSError:
        pass


def _stage3_for(route: str, style):
    """Resolve + load (cached) the Stage-3 polish model for a route, or None."""
    ckpts = worker.resolve_polish_ckpts(style)
    path = ckpts.get(route)
    if not path:
        return None, None
    key = str(path)
    if key not in _S3_CACHE:
        _S3_CACHE[key] = load_stage3(Path(path), DEV)
    # stage3Variant is the ckpt's parent dir name (e.g. "polish_exterior_v1"),
    # mirroring how the production run records it on ImageReview.
    return _S3_CACHE[key], Path(path).parent.name


def process(rid, s2_int, s2_ext):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        if rc(["copyto", f"{REQ}/{rid}.json", str(td / "req.json")], timeout=30).returncode != 0:
            raise RuntimeError("request fetch failed")
        req = json.loads((td / "req.json").read_text())

        job_id = req["jobId"]
        mid_stem = req["midStem"]
        sort_order = int(req.get("sortOrder") or 0)
        scene = req["scene"]
        if scene not in ("interior", "exterior"):
            raise RuntimeError(f"bad scene {scene!r}")
        route = "interior" if scene == "interior" else "exterior"
        stage1_key = req["stage1Key"]
        production_key = req["productionKey"]
        dropbox_path = req.get("dropboxPath") or ""
        style = req.get("polishStyle")

        # 1) pull the Stage-1 frame (full res) from the training bucket
        src = td / "stage1.jpg"
        if rc(["copyto", f"{R2}:{QUEUE_BUCKET}/{stage1_key}", str(src)], timeout=120).returncode != 0:
            raise RuntimeError(f"stage1 fetch failed: {stage1_key}")

        # 2) Stage-2 on the FORCED lane (no classify)
        s2_out = td / "stage2.jpg"
        stage2_infer(s2_int if route == "interior" else s2_ext, src, s2_out, DEV)

        # 3) Stage-3 polish for that lane (same resolver as production); the
        #    polished file replaces the Stage-2 output, matching production.
        s3_model, s3_variant = _stage3_for(route, style)
        applied_variant = None
        if s3_model is not None:
            s3_out = td / "stage3.jpg"
            try:
                stage3_infer(s3_model, s2_out, s3_out, DEV)
                s3_out.replace(s2_out)
                applied_variant = s3_variant
            except Exception as e:
                log(f"{rid} stage3 fail (keeping stage2): {str(e)[:160]}")

        final = s2_out

        # 4) overwrite the production R2 object (exact existing key)
        if rc(["copyto", str(final), f"{R2}:{worker.R2_OUTPUT_BUCKET}/{production_key}",
               "--progress=false", "--s3-no-check-bucket"], timeout=180).returncode != 0:
            raise RuntimeError("production R2 copy failed")

        # 5) fresh CF Images upload (new id busts the gallery cache)
        cf_id = worker._cf_images_upload(final, job_id=job_id, mid_stem=mid_stem, sort_order=sort_order)

        # 6) overwrite the Dropbox 08-Test-Edit staging JPEG (bare <stem>.jpg)
        if dropbox_path:
            dbx = worker.resolve_dropbox_source(dropbox_path)
            dst = f"{dbx}/{worker.DROPBOX_OUTPUT_FOLDER}/{mid_stem}.jpg"
            r = rc(["copyto", str(final), dst, "--progress=false"], timeout=180)
            if r.returncode != 0:
                log(f"{rid} WARN dropbox copy rc={r.returncode}: {r.stderr[-160:].strip()}")

        dt = int((time.time() - t0) * 1000)
        write_result(rid, {
            "status": "done",
            "cfImageId": cf_id,
            "productionKey": production_key,
            "scene": scene,
            "stage3Variant": applied_variant,
            "ms": dt,
        })
        log(f"{rid} -> done {mid_stem} {scene} cf={cf_id} {dt}ms")


def main():
    log(f"loading Stage-2 models on {DEV} …")
    s2_int = load_stage2(worker.CHECKPOINT_INTERIOR, DEV)
    s2_ext = load_stage2(worker.CHECKPOINT_EXTERIOR, DEV)
    log("ready, polling reprocess-jobs/requests …")
    while True:
        r = rc(["lsf", REQ, "--include", "*.json"], timeout=30)
        reqs = [x for x in r.stdout.split() if x.endswith(".json")] if r.returncode == 0 else []
        if not reqs:
            time.sleep(0.5)
            continue
        for rj in reqs:
            rid = rj[:-5]
            try:
                process(rid, s2_int, s2_ext)
            except Exception as e:
                log(f"{rid} ERROR {str(e)[:200]}")
                write_result(rid, {"status": "error", "error": str(e)[:200]})
            rc(["deletefile", f"{REQ}/{rj}"], timeout=20)


if __name__ == "__main__":
    sys.exit(main())
