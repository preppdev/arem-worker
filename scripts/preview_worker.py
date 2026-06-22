#!/usr/bin/env python3
"""Warm single-frame PREVIEW worker. Polls R2 for preview requests from the
uploader app, runs the real enhance path at preview resolution (fuse bracket →
route interior/exterior → Stage-2 NAFNet), and writes the result back. Models
stay resident so each preview is sub-second of GPU. Reuses the production
loaders from pipeline.run_pipeline so the look matches the real deliverable.

I/O is in-process **boto3** (persistent, connection-pooled S3 client) — NOT
rclone. rclone shelled out ~4 processes per request (~0.3-0.5s spawn each) and
DOMINATED the ~3s/request time; the GPU work at ≤1600px is only ~0.2-0.5s. boto3
keeps everything in memory → most of that overhead disappears.

Multi-worker: PREVIEW_SHARD / PREVIEW_SHARDS shard the queue by request-id hash
so N nodes split the load with no coordination and no double-processing.

Queue (R2 arem-training-data):
  preview-jobs/requests/<id>.json   {frames:N, ts}
  preview-jobs/input/<id>/<i>.jpg   the N embedded JPEGs the app sent
  preview-jobs/results/<id>.jpg      enhanced preview
  preview-jobs/results/<id>.done.json {status, resultKey, scene, ms} | {status:error,error}
"""
import os, sys, time, json, tempfile
from pathlib import Path
import cv2, numpy as np, torch, boto3
import torch.nn as nn
import torchvision.models as tvm

BASE = Path(os.environ.get("AREM_WORKER_DIR", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(BASE))
from pipeline.run_pipeline import load_stage2, stage2_infer  # noqa: E402

BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
P_REQ = "preview-jobs/requests"
P_INP = "preview-jobs/input"
P_RES = "preview-jobs/results"
PREVIEW_MAX = int(os.environ.get("PREVIEW_MAX_PX", "1600"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHARD = int(os.environ.get("PREVIEW_SHARD", "0"))
SHARDS = int(os.environ.get("PREVIEW_SHARDS", "1"))

# Persistent S3 (Cloudflare R2) client — reused across every request.
S3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

def log(m): print(f"[preview] {m}", flush=True)

def s3_list(prefix):
    out, tok = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if tok:
            kw["ContinuationToken"] = tok
        r = S3.list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r.get("NextContinuationToken")

def s3_get(key): return S3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
def s3_put(key, data, ct): S3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=ct)
def s3_del(key): S3.delete_object(Bucket=BUCKET, Key=key)

# Fast int/ext router (mobilenet_v3_small @384) — distilled from the 12MP teacher.
# ~7ms vs ~280ms; preview-only (delivery keeps the full-accuracy router). The
# routing just picks which Stage-2 model renders the preview.
FAST_ROUTER_KEY = os.environ.get("FAST_ROUTER_R2KEY", "models/preview-router/fast_router_v1.pt")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def load_fast_router():
    local = BASE / "checkpoints" / "fast_router_v1.pt"
    if not (local.is_file() and local.stat().st_size > 0):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(s3_get(FAST_ROUTER_KEY))
    ck = torch.load(str(local), map_location=DEV)
    m = tvm.mobilenet_v3_small()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m.load_state_dict(ck["model_state"])
    return m.to(DEV).eval(), ck["label_names"]

def _letterbox384(bgr):
    h, w = bgr.shape[:2]; s = 384 / max(h, w); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(bgr, (nw, nh)); c = np.zeros((384, 384, 3), np.uint8)
    t, l = (384 - nh) // 2, (384 - nw) // 2; c[t:t + nh, l:l + nw] = r; return c

def classify_fast(model, labels, bgr):
    t = torch.from_numpy(_letterbox384(bgr)[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    x = ((t - _MEAN) / _STD).unsqueeze(0).to(DEV)
    with torch.inference_mode():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    idx = int(p.argmax())
    return labels[idx] == "interior", labels[idx]

def write_result(rid, payload):
    s3_put(f"{P_RES}/{rid}.done.json", json.dumps(payload).encode(), "application/json")

def process(rid, router, rlabels, s2_int, s2_ext):
    t0 = time.time()
    # 1) pull the bracket's input frames (in memory — no rclone)
    in_keys = sorted(s3_list(f"{P_INP}/{rid}/"))
    imgs = []
    for k in in_keys:
        im = cv2.imdecode(np.frombuffer(s3_get(k), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    if not imgs:
        raise RuntimeError("no decodable input frames")
    # 2) fuse the bracket (window/exposure recovery) if >1 frame
    if len(imgs) >= 2:
        h = min(i.shape[0] for i in imgs); w = min(i.shape[1] for i in imgs)
        imgs = [cv2.resize(i, (w, h)) for i in imgs]
        base = np.clip(cv2.createMergeMertens().process(imgs) * 255, 0, 255).astype(np.uint8)
    else:
        base = imgs[0]
    h, w = base.shape[:2]; lo = max(h, w)
    if lo > PREVIEW_MAX:
        s = PREVIEW_MAX / lo; base = cv2.resize(base, (int(round(w * s)), int(round(h * s))))
    # 3) route (fast mobilenet on the in-memory image) + enhance
    is_int, label = classify_fast(router, rlabels, base)
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        src = td / "src.jpg"; cv2.imwrite(str(src), base, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out = td / "out.jpg"
        stage2_infer(s2_int if is_int else s2_ext, src, out, DEV)
        result = out.read_bytes()
    # 4) write result + done marker (boto3)
    s3_put(f"{P_RES}/{rid}.jpg", result, "image/jpeg")
    dt = int((time.time() - t0) * 1000)
    write_result(rid, {"status": "done", "resultKey": f"{P_RES}/{rid}.jpg", "scene": label, "ms": dt})
    log(f"{rid} -> done {label} {dt}ms")

def main():
    log(f"loading models on {DEV} …")
    router, rlabels = load_fast_router()
    s2_int = load_stage2(BASE / "checkpoints" / "may26_interior_w32_b4_4gpu_ep35_inference.pth", DEV)
    s2_ext = load_stage2(BASE / "checkpoints" / "may26_exterior_w32_b4_4gpu_ep29_inference.pth", DEV)
    log(f"ready (shard {SHARD}/{SHARDS}, boto3 I/O, fast router), polling {P_REQ} …")
    while True:
        try:
            keys = s3_list(f"{P_REQ}/")
        except Exception as e:
            log(f"list error {str(e)[:160]}"); time.sleep(1); continue
        reqs = [k.split("/")[-1] for k in keys if k.endswith(".json")]
        if SHARDS > 1:
            reqs = [x for x in reqs if int(x[:8], 16) % SHARDS == SHARD]
        if not reqs:
            time.sleep(0.25); continue
        for rj in reqs:
            rid = rj[:-5]
            try:
                process(rid, router, rlabels, s2_int, s2_ext)
            except Exception as e:
                log(f"{rid} ERROR {str(e)[:200]}"); write_result(rid, {"status": "error", "error": str(e)[:200]})
            try:
                s3_del(f"{P_REQ}/{rj}")
            except Exception as e:
                log(f"{rid} del error {str(e)[:120]}")

if __name__ == "__main__":
    sys.exit(main())
