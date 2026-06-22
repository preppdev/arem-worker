#!/usr/bin/env python3
"""Warm single-frame PREVIEW worker. Polls R2 for preview requests from the
uploader app, runs the real enhance path at preview resolution (fuse bracket →
route interior/exterior → Stage-2 NAFNet), and writes the result back. Models
stay resident so each preview is sub-second of GPU. Reuses the production
loaders from pipeline.run_pipeline so the look matches the real deliverable.

Queue (R2 arem-training-data):
  preview-jobs/requests/<id>.json   {frames:N, ts}
  preview-jobs/input/<id>/<i>.jpg   the N embedded JPEGs the app sent
  preview-jobs/results/<id>.jpg      enhanced preview
  preview-jobs/results/<id>.done.json {status, resultKey, scene, ms} | {status:error,error}
"""
import os, sys, time, json, subprocess, tempfile, glob
from pathlib import Path
import cv2, numpy as np, torch

BASE = Path(os.environ.get("AREM_WORKER_DIR", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(BASE))
from pipeline.run_pipeline import load_stage2, load_classifier, classify, stage2_infer  # noqa: E402

R2 = os.environ.get("RCLONE_R2", "r2")
BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
REQ = f"{R2}:{BUCKET}/preview-jobs/requests"
INP = f"{R2}:{BUCKET}/preview-jobs/input"
RES = f"{R2}:{BUCKET}/preview-jobs/results"
PREVIEW_MAX = int(os.environ.get("PREVIEW_MAX_PX", "1600"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Multi-worker sharding: run this worker on N nodes, each with a distinct
# PREVIEW_SHARD in [0, PREVIEW_SHARDS). Each node only claims requests whose id
# hashes to its shard, so the queue splits evenly with NO coordination and no
# request is ever processed twice. Throughput scales linearly with node count.
SHARD = int(os.environ.get("PREVIEW_SHARD", "0"))
SHARDS = int(os.environ.get("PREVIEW_SHARDS", "1"))

def log(m): print(f"[preview] {m}", flush=True)
def rc(args, timeout=60): return subprocess.run(["rclone"] + args, capture_output=True, text=True, timeout=timeout)

def write_result(rid, payload):
    p = Path(tempfile.gettempdir()) / f"pvres_{rid}.json"
    p.write_text(json.dumps(payload))
    rc(["copyto", str(p), f"{RES}/{rid}.done.json"], timeout=20)
    try: p.unlink()
    except OSError: pass

def process(rid, clf, s2_int, s2_ext):
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        if rc(["copyto", f"{REQ}/{rid}.json", str(td / "req.json")], timeout=20).returncode != 0:
            return
        rc(["copy", f"{INP}/{rid}", str(td)], timeout=120)
        ins = sorted(glob.glob(str(td / "*.jpg")))
        if not ins:
            raise RuntimeError("no input frames")
        # fuse the bracket (window/exposure recovery) if >1 frame
        if len(ins) >= 2:
            imgs = [cv2.imread(p) for p in ins]
            h = min(i.shape[0] for i in imgs); w = min(i.shape[1] for i in imgs)
            imgs = [cv2.resize(i, (w, h)) for i in imgs]
            base = np.clip(cv2.createMergeMertens().process(imgs) * 255, 0, 255).astype(np.uint8)
        else:
            base = cv2.imread(ins[0])
        h, w = base.shape[:2]; lo = max(h, w)
        if lo > PREVIEW_MAX:
            s = PREVIEW_MAX / lo; base = cv2.resize(base, (int(round(w * s)), int(round(h * s))))
        src = td / "src.jpg"; cv2.imwrite(str(src), base, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # route + enhance (the AREM look)
        is_int, _, label = classify(*clf, str(src), DEV)
        out = td / "out.jpg"
        stage2_infer(s2_int if is_int else s2_ext, src, out, DEV)
        rc(["copyto", str(out), f"{RES}/{rid}.jpg"], timeout=60)
        dt = int((time.time() - t0) * 1000)
        write_result(rid, {"status": "done", "resultKey": f"preview-jobs/results/{rid}.jpg", "scene": label, "ms": dt})
        log(f"{rid} -> done {label} {dt}ms")

def main():
    log(f"loading models on {DEV} …")
    clf = load_classifier(BASE / "pipeline" / "classifier_v4.pth", DEV)
    s2_int = load_stage2(BASE / "checkpoints" / "may26_interior_w32_b4_4gpu_ep35_inference.pth", DEV)
    s2_ext = load_stage2(BASE / "checkpoints" / "may26_exterior_w32_b4_4gpu_ep29_inference.pth", DEV)
    log(f"ready (shard {SHARD}/{SHARDS}), polling preview-jobs/requests …")
    while True:
        r = rc(["lsf", REQ, "--include", "*.json"], timeout=30)
        reqs = [x for x in r.stdout.split() if x.endswith(".json")] if r.returncode == 0 else []
        if SHARDS > 1:
            # Claim only this node's slice of the id-space (hex id → shard).
            reqs = [x for x in reqs if int(x[:8], 16) % SHARDS == SHARD]
        if not reqs:
            time.sleep(0.25); continue
        for rj in reqs:
            rid = rj[:-5]
            try:
                process(rid, clf, s2_int, s2_ext)
            except Exception as e:
                log(f"{rid} ERROR {str(e)[:200]}"); write_result(rid, {"status": "error", "error": str(e)[:200]})
            rc(["deletefile", f"{REQ}/{rj}"], timeout=20)

if __name__ == "__main__":
    sys.exit(main())
