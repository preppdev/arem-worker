#!/usr/bin/env python3
"""Enhancement-pipeline worker.

Independent of the main editing worker. Polls the dashboard's
/api/internal/enhancement-jobs/claim every IDLE_SLEEP seconds; when it
gets a job, it iterates every ImageReview in the parent Job × every
configured step, running the detector + repair as the step spec
dictates, and POSTs an EnhancementRequest per (image × step) to the
dashboard.

Designed for the open-ended step registry in lib/enhancement-steps.ts —
the worker doesn't hard-code step behavior; it reads each step's
detector and repair specs from the claim response and dispatches based
on type (`vlm-haiku` / `nano-banana-pro`). Adding a new step on the
dashboard side requires no worker change as long as the type fields
are already supported here.

Usage:
    python enhancement_worker.py            # poll forever
    python enhancement_worker.py --once     # claim+process one, exit
    python enhancement_worker.py --dry-run  # log what we'd do, don't POST

Env (required):
    WORKER_TOKEN          dashboard auth
    DASHBOARD_URL         default https://arem-editing-dashboard.vercel.app
    ANTHROPIC_API_KEY     required for vlm-haiku detector
    GOOGLE_AI_API_KEY     required for nano-banana-pro repair
    CAMTRIP_DETECTOR_PATH required for frcnn-local detector (default
                          /home/jordan/models/camtrip_detector_v3_native.pth)

Env (optional):
    WORKER_ID             default enhancement-local-<hostname>
    IDLE_SLEEP            seconds between idle polls (default 5)
    R2_PRODUCTION_BUCKET  default arem-production-edit-jobs (sources)
    R2_TRAINING_BUCKET    default arem-training-data (derivatives)
    RCLONE_R2             rclone remote (default 'r2')
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Third-party deps imported lazily so --help works even when missing.

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
WORKER_ID = os.environ.get("WORKER_ID", f"enhancement-local-{socket.gethostname()}")
IDLE_SLEEP = int(os.environ.get("IDLE_SLEEP", "5"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY", "")
R2_PRODUCTION_BUCKET = os.environ.get("R2_PRODUCTION_BUCKET", "arem-production-edit-jobs")
R2_TRAINING_BUCKET = os.environ.get("R2_TRAINING_BUCKET", "arem-training-data")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
CAMTRIP_DETECTOR_PATH = os.environ.get(
    "CAMTRIP_DETECTOR_PATH", "/home/jordan/models/camtrip_detector_v4_native.pth")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Magenta marker matches the dashboard's lib/enhancement.ts.
MARKER_RGB = (255, 0, 220)
MARKER_WIDTH_PX = 8


def log(msg: str) -> None:
    print(f"[{WORKER_ID}] {msg}", flush=True)


# ---- Dashboard HTTP ----------------------------------------------------------

def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"{method} {path} → HTTP {e.code}: {body_text}") from e


def claim_job() -> dict | None:
    r = _request("POST", "/api/internal/enhancement-jobs/claim", {"workerId": WORKER_ID})
    return r.get("job") and r or None


def report_progress(job_id: str, *, processed=0, detected=0, repaired=0, failed=0) -> None:
    _request("POST", f"/api/internal/enhancement-jobs/{job_id}/progress", {
        "dProcessed": processed, "dDetected": detected,
        "dRepaired": repaired, "dFailed": failed,
    })


def complete_job(job_id: str, ok: bool, error_message: str | None = None) -> None:
    _request("POST", f"/api/internal/enhancement-jobs/{job_id}/complete", {
        "ok": ok, "errorMessage": error_message,
    })


def post_request(payload: dict) -> None:
    _request("POST", "/api/internal/enhancement-requests", payload)


# ---- R2 ---------------------------------------------------------------------

def r2_fetch(key: str, bucket: str, dest: Path) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", f"{RCLONE_R2}:{bucket}/{key}", str(dest),
         "--retries=2", "--low-level-retries=3"],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        log(f"  R2 fetch FAIL {bucket}/{key}: {r.stderr[-160:].strip()}")
        return False
    return True


def r2_upload(local: Path, key: str, bucket: str) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", str(local), f"{RCLONE_R2}:{bucket}/{key}",
         "--retries=2", "--low-level-retries=3"],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        log(f"  R2 upload FAIL {bucket}/{key}: {r.stderr[-160:].strip()}")
        return False
    return True


# ---- Image annotation -------------------------------------------------------

def render_annotated(source_bytes: bytes, bbox: dict) -> bytes:
    """Draw a magenta rectangle border at the fractional bbox. Returns
    JPEG bytes. Mirrors lib/enhancement.ts renderAnnotatedImage."""
    from PIL import Image, ImageDraw  # type: ignore
    img = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    W, H = img.size
    px = round(bbox["x"] * W); py = round(bbox["y"] * H)
    pw = max(2, round(bbox["w"] * W)); ph = max(2, round(bbox["h"] * H))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(px, py), (px + pw, py + ph)], outline=MARKER_RGB,
                   width=MARKER_WIDTH_PX)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


# ---- Detector + Repair dispatchers ------------------------------------------

# Lazy-loaded FRCNN (frcnn-local detector). Loaded on first use within a
# job and released after (release_frcnn) — this node's GPU is shared with
# the editing worker, so we don't keep ~2GB of weights resident while idle.
_FRCNN = None
_FRCNN_LABELS = {1: "camera", 2: "tripod"}


def _load_frcnn():
    global _FRCNN
    if _FRCNN is None:
        import torch  # type: ignore
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2  # type: ignore
        ck = torch.load(CAMTRIP_DETECTOR_PATH, map_location="cpu", weights_only=False)
        m = fasterrcnn_resnet50_fpn_v2(
            weights=None, num_classes=ck["num_classes"],
            min_size=ck["min_size"], max_size=ck["max_size"])
        m.load_state_dict(ck["model_state"])
        m.eval().cuda()
        _FRCNN = (m, ck.get("version", "frcnn-local"))
        log(f"frcnn loaded: {ck.get('version')} (mAP50 {ck.get('metrics', {}).get('mAP50')})")
    return _FRCNN


def release_frcnn() -> None:
    global _FRCNN
    if _FRCNN is not None:
        import torch  # type: ignore
        _FRCNN = None
        torch.cuda.empty_cache()


def call_frcnn_detector(spec: dict, image_bytes: bytes) -> dict:
    """In-house Faster R-CNN camera/tripod-reflection detector, run locally
    on the node GPU. Adds `boxes`: fractional [{x,y,w,h,label,score}] for
    every detection >= 0.5 (the dashboard applies its own flag threshold)."""
    import torch  # type: ignore
    import torchvision.io  # type: ignore
    t0 = time.time()
    try:
        model, _version = _load_frcnn()
        data = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        img = torchvision.io.decode_image(data, mode=torchvision.io.ImageReadMode.RGB)
        _, H, W = img.shape
        with torch.inference_mode():
            out = model([img.float().div(255).cuda()])[0]
    except Exception as e:
        return {"positive": False, "confidence": None, "region": None,
                "reason": "", "runtimeMs": int((time.time() - t0) * 1000),
                "costUsd": None, "error": f"frcnn: {e}", "boxes": []}
    boxes = []
    for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
        score = float(s)
        if score < 0.5:
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        boxes.append({
            "x": x1 / W, "y": y1 / H, "w": (x2 - x1) / W, "h": (y2 - y1) / H,
            "label": _FRCNN_LABELS.get(int(l), str(int(l))), "score": round(score, 4),
        })
    max_score = max((b["score"] for b in boxes), default=0.0)
    return {
        "positive": bool(boxes),
        "confidence": max_score or None,
        "region": None,
        "reason": ", ".join(f"{b['label']} {b['score']:.2f}" for b in boxes)[:480],
        "runtimeMs": int((time.time() - t0) * 1000),
        "costUsd": None, "error": None, "boxes": boxes,
    }


def call_detector(spec: dict, image_bytes: bytes) -> dict:
    """Returns: { positive: bool, confidence: float|None, region: str|None,
                  reason: str, runtimeMs: int, costUsd: float|None,
                  error: str|None, boxes?: [...] (frcnn-local only) }"""
    if spec["type"] == "frcnn-local":
        return call_frcnn_detector(spec, image_bytes)
    if spec["type"] != "vlm-haiku":
        raise ValueError(f"unsupported detector type: {spec['type']}")
    import anthropic  # type: ignore
    # Anthropic 5MB cap — resize if needed.
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(img.size) > 1600:
        scale = 1600 / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=spec["model"],
            max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": spec["promptText"]},
            ]}],
        )
    except Exception as e:
        return {"positive": False, "confidence": None, "region": None,
                "reason": "", "runtimeMs": int((time.time() - t0) * 1000),
                "costUsd": None, "error": f"haiku call: {e}"}
    runtime_ms = int((time.time() - t0) * 1000)
    text = resp.content[0].text.strip() if resp.content else ""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except Exception as e:
        return {"positive": False, "confidence": None, "region": None,
                "reason": "", "runtimeMs": runtime_ms, "costUsd": None,
                "error": f"json parse: {e}: {text[:200]}"}
    cost = (resp.usage.input_tokens * 1.0 + resp.usage.output_tokens * 5.0) / 1_000_000
    return {
        "positive": bool(data.get("reflection") or data.get("positive")
                         or data.get(spec.get("targetField", "reflection"))),
        "confidence": data.get("confidence"),
        "region": data.get("region"),
        "reason": (data.get("reason") or "")[:480],
        "runtimeMs": runtime_ms, "costUsd": cost, "error": None,
    }


def call_repair(spec: dict, image_bytes: bytes, prompt: str) -> dict:
    """Returns: { resultBytes: bytes|None, modelUsed: str, runtimeMs: int,
                  costUsd: float|None, error: str|None }"""
    if spec["type"] != "nano-banana-pro":
        raise ValueError(f"unsupported repair type: {spec['type']}")
    fallback = "gemini-2.5-flash-image"
    last_err = None
    for model in [spec["model"], fallback]:
        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(image_bytes).decode()}},
            ]}],
            "generationConfig": {"imageConfig": {"imageSize": spec["imageSize"]}},
        }
        req = urllib.request.Request(
            f"{GEMINI_API_BASE}/{model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"x-goog-api-key": GOOGLE_AI_API_KEY, "Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            last_err = f"{model} HTTP {e.code}: {msg}"
            if e.code == 503 or "UNAVAILABLE" in msg:
                continue
            return {"resultBytes": None, "modelUsed": model,
                    "runtimeMs": int((time.time() - t0) * 1000),
                    "costUsd": None, "error": last_err}
        except Exception as e:
            last_err = f"{model}: {e}"
            continue
        runtime_ms = int((time.time() - t0) * 1000)
        for c in data.get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                if p.get("inlineData", {}).get("data"):
                    return {
                        "resultBytes": base64.b64decode(p["inlineData"]["data"]),
                        "modelUsed": model, "runtimeMs": runtime_ms,
                        "costUsd": None, "error": None,
                    }
        last_err = f"{model}: no inline image in response"
    return {"resultBytes": None, "modelUsed": spec["model"],
            "runtimeMs": 0, "costUsd": None, "error": last_err}


# ---- Region → bbox ----------------------------------------------------------

def boxes_union_bbox(boxes: list[dict]) -> dict | None:
    """Union of fractional detector boxes, 3% pad each side (clamped).
    Mirrors the dashboard's /api/corrections union for the repair marker."""
    if not boxes:
        return None
    x1 = min(b["x"] for b in boxes); y1 = min(b["y"] for b in boxes)
    x2 = max(b["x"] + b["w"] for b in boxes); y2 = max(b["y"] + b["h"] for b in boxes)
    px1, py1 = max(0.0, x1 - 0.03), max(0.0, y1 - 0.03)
    return {"x": px1, "y": py1,
            "w": min(1.0, x2 + 0.03) - px1, "h": min(1.0, y2 + 0.03) - py1}


def region_to_bbox(region: str | None) -> dict | None:
    """Map Haiku's 3×3 grid cell to a fractional bbox (~37% of the image).
    Slight overlap with neighbors so the model has context for the inpaint."""
    if not region:
        return None
    rows = {"top": 0.0, "middle": 0.31, "bottom": 0.62}
    cols = {"left": 0.0, "center": 0.31, "right": 0.62}
    parts = region.split("-")
    if len(parts) != 2 or parts[0] not in rows or parts[1] not in cols:
        return None
    return {"x": cols[parts[1]], "y": rows[parts[0]], "w": 0.38, "h": 0.38}


# ---- Process one EnhancementJob ---------------------------------------------

def process_job(envelope: dict, dry_run: bool = False) -> tuple[int, int, int]:
    """Returns (n_processed, n_detected, n_repaired)."""
    job = envelope["job"]
    parent = envelope["parent"]
    images = envelope["images"]
    step_configs = envelope["stepConfigs"]
    job_id = job["id"]

    log(f"job {job_id}: parent={parent.get('dropboxPath','?')[:50]} "
        f"images={len(images)} steps={[s['key'] for s in step_configs]}")

    n_processed = n_detected = n_repaired = n_failed = 0

    with tempfile.TemporaryDirectory(prefix="enh-worker-") as scratch_str:
        scratch = Path(scratch_str)
        for img in images:
            # Only handle images with a productionR2Path (the new shape).
            # Legacy Dropbox-only paths are skipped — they'd need a
            # different fetch path.
            src_key = img.get("productionR2Path")
            if not src_key:
                continue

            for step in step_configs:
                step_key = step["key"]
                detector_spec = step.get("detector")
                repair_spec = step.get("repair")

                tmp_src = scratch / f"{img['id']}.jpg"
                if not r2_fetch(src_key, R2_PRODUCTION_BUCKET, tmp_src):
                    n_failed += 1; n_processed += 1
                    if not dry_run:
                        report_progress(job_id, processed=1, failed=1)
                    continue
                src_bytes = tmp_src.read_bytes()

                # --- detection phase ---
                det_positive = True
                det_confidence = None
                det_region = None
                det_reason = ""
                det_source = None
                if detector_spec:
                    det_source = f"{detector_spec['type']}-{detector_spec['promptVersion']}"
                    if dry_run:
                        log(f"  DRY {step_key} detect: skipped"); det_positive = False
                    else:
                        det = call_detector(detector_spec, src_bytes)
                        det_positive = det["positive"]
                        det_confidence = det["confidence"]
                        det_region = det["region"]
                        det_reason = det["reason"]
                        floor = detector_spec.get("confidenceFloor")
                        if floor and det_confidence is not None and det_confidence < floor:
                            det_positive = False
                        if det.get("error"):
                            log(f"  detect ERR {step_key} {img['midStem']}: {det['error']}")

                if detector_spec and not det_positive:
                    n_processed += 1
                    if not dry_run:
                        report_progress(job_id, processed=1)
                    continue

                n_detected += 1 if detector_spec else 0
                det_boxes = (det.get("boxes") if detector_spec and not dry_run else None) or []

                # frcnn positives become CorrectionFlag rows on the dashboard
                # (red outline on /delivery, pooled on /corrections). Distinct
                # from the EnhancementRequest audit row below.
                if det_boxes and not dry_run:
                    try:
                        _request("POST", "/api/internal/camtrip-detections", {
                            "jobId": parent["id"],
                            "midStem": img["midStem"],
                            "modelVersion": detector_spec["model"],
                            "boxes": det_boxes,
                        })
                    except Exception as e:
                        log(f"  flag POST ERR {img['midStem']}: {e}")

                # --- repair phase ---
                bbox = boxes_union_bbox(det_boxes) or (region_to_bbox(det_region) if det_region else None)
                if not repair_spec:
                    # detection-only step: just log the result row, no repair.
                    if not dry_run:
                        post_request({
                            "enhancementJobId": job_id,
                            "imageReviewId": img["id"],
                            "sourceR2Path": src_key,
                            "condition": step_key,
                            "prompt": detector_spec.get("promptText", "") if detector_spec else "",
                            "bbox": bbox,
                            "status": "complete",
                            "modelVersion": detector_spec["model"] if detector_spec else None,
                            "detectionSource": det_source,
                            "detectionConfidence": det_confidence,
                            "detectionRegion": det_region,
                        })
                        report_progress(job_id, processed=1, detected=1)
                    n_processed += 1
                    continue

                # Render annotated input if we have a bbox
                model_input = src_bytes
                annotated_key = None
                if bbox:
                    try:
                        model_input = render_annotated(src_bytes, bbox)
                        annotated_key = f"enhancements/{job_id}/{img['id']}/{step_key}_annotated.jpg"
                        tmp_ann = scratch / f"ann_{img['id']}.jpg"
                        tmp_ann.write_bytes(model_input)
                        if not dry_run:
                            r2_upload(tmp_ann, annotated_key, R2_TRAINING_BUCKET)
                    except Exception as e:
                        log(f"  annotate ERR {img['midStem']}: {e}")

                # Wrap the prompt with bbox preamble if applicable. Mirrors
                # lib/enhancement.ts wrapPrompt.
                base_prompt = repair_spec["promptText"]
                if bbox:
                    final_prompt = (
                        "There is a magenta rectangle drawn on this image marking the "
                        "region you must focus on. Inside that rectangle, perform this edit:\n\n"
                        + base_prompt +
                        "\n\nCRITICAL:\n"
                        "- The magenta rectangle is ONLY a marker for your attention. It is NOT part of the original photo.\n"
                        "- Do NOT include the magenta rectangle in your output.\n"
                        "- Preserve every pixel OUTSIDE the rectangle exactly — same colors, lighting, framing, composition.\n"
                        "- Only modify pixels INSIDE the rectangle as instructed above."
                    )
                else:
                    final_prompt = base_prompt

                if dry_run:
                    log(f"  DRY {step_key} repair: {img['midStem']} bbox={bbox}")
                    n_processed += 1
                    continue

                rep = call_repair(repair_spec, model_input, final_prompt)
                if rep["error"] or not rep["resultBytes"]:
                    log(f"  repair FAIL {img['midStem']}: {rep['error']}")
                    post_request({
                        "enhancementJobId": job_id,
                        "imageReviewId": img["id"],
                        "sourceR2Path": src_key,
                        "condition": step_key,
                        "prompt": final_prompt,
                        "bbox": bbox,
                        "annotatedImageR2Path": annotated_key,
                        "status": "failed",
                        "errorMessage": rep["error"],
                        "modelVersion": rep["modelUsed"],
                        "runtimeMs": rep["runtimeMs"],
                        "detectionSource": det_source,
                        "detectionConfidence": det_confidence,
                        "detectionRegion": det_region,
                    })
                    n_failed += 1; n_processed += 1
                    report_progress(job_id, processed=1, detected=1, failed=1)
                    continue

                # Upload the repair result
                result_key = f"enhancements/{job_id}/{img['id']}/{step_key}_result.jpg"
                tmp_res = scratch / f"res_{img['id']}.jpg"
                tmp_res.write_bytes(rep["resultBytes"])
                r2_upload(tmp_res, result_key, R2_TRAINING_BUCKET)

                post_request({
                    "enhancementJobId": job_id,
                    "imageReviewId": img["id"],
                    "sourceR2Path": src_key,
                    "condition": step_key,
                    "prompt": final_prompt,
                    "bbox": bbox,
                    "annotatedImageR2Path": annotated_key,
                    "resultImageR2Path": result_key,
                    "status": "complete",
                    "modelVersion": rep["modelUsed"],
                    "runtimeMs": rep["runtimeMs"],
                    "detectionSource": det_source,
                    "detectionConfidence": det_confidence,
                    "detectionRegion": det_region,
                })
                n_repaired += 1; n_processed += 1
                report_progress(job_id, processed=1, detected=1, repaired=1)

    return n_processed, n_detected, n_repaired


# ---- Main loop --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="claim+process one job, then exit")
    ap.add_argument("--dry-run", action="store_true", help="log what we'd do, don't POST")
    args = ap.parse_args()

    if not WORKER_TOKEN: log("ERROR: WORKER_TOKEN required"); return 2
    if not ANTHROPIC_API_KEY: log("WARN: ANTHROPIC_API_KEY missing — detector calls will fail")
    if not GOOGLE_AI_API_KEY: log("WARN: GOOGLE_AI_API_KEY missing — repair calls will fail")

    log(f"polling {DASHBOARD_URL} every {IDLE_SLEEP}s (--once={args.once})")
    while True:
        try:
            r = _request("POST", "/api/internal/enhancement-jobs/claim", {"workerId": WORKER_ID})
            envelope = r if r.get("job") else None
        except Exception as e:
            log(f"claim error: {e}"); envelope = None

        if envelope:
            jid = envelope["job"]["id"]
            try:
                p, d, r = process_job(envelope, dry_run=args.dry_run)
                log(f"job {jid} done: processed={p} detected={d} repaired={r}")
                if not args.dry_run:
                    complete_job(jid, ok=True)
            except Exception as e:
                log(f"job {jid} FAILED: {e}")
                if not args.dry_run:
                    complete_job(jid, ok=False, error_message=str(e)[:500])
            finally:
                release_frcnn()  # GPU is shared with the editing worker
            if args.once: return 0
        else:
            if args.once:
                log("queue empty (--once)"); return 0
            time.sleep(IDLE_SLEEP)


if __name__ == "__main__":
    sys.exit(main())
