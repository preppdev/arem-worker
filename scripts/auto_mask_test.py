"""Auto-mask test harness — Florence-2 over images flagged for one
condition (e.g. reflection). Generates a binary mask per image, uploads
to R2, and POSTs the result to the dashboard so /auto-mask/<condition>
can render the side-by-side gallery.

Reads candidates from /api/internal/auto-mask-candidates and writes
results to /api/internal/auto-mask-test. Resumable — already-processed
(condition, image, modelVersion) tuples are skipped server-side.

Usage:
    python -m scripts.auto_mask_test \\
        --condition reflection \\
        --model microsoft/Florence-2-large \\
        --prompt "reflection on glass or mirror" \\
        --limit 500

Env:
    WORKER_TOKEN              dashboard auth (required)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data (mask uploads land in
                              <bucket>/auto-masks/<condition>/<safe-model>/...)
    RCLONE_R2                 default 'r2'
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs (source images are
                              fetched from here)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image, ImageDraw  # type: ignore
from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")


def log(msg: str) -> None:
    print(msg, flush=True)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {text[:300]}") from e


def get_candidates(*, condition: str, model_version: str,
                   limit: int) -> list[str]:
    sp = urllib.parse.urlencode({
        "condition": condition,
        "modelVersion": model_version,
        "limit": str(limit),
    })
    resp = _request("GET", f"/api/internal/auto-mask-candidates?{sp}")
    log(f"  candidates: {resp.get('todo')}/{resp.get('total')} todo "
        f"(already done: {resp.get('alreadyDone')})")
    return resp.get("items") or []


def post_result(payload: dict) -> dict:
    return _request("POST", "/api/internal/auto-mask-test", payload)


def rclone(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True,
                          timeout=timeout)


def safe_model_id(model: str) -> str:
    """microsoft/Florence-2-large -> florence2-large"""
    name = model.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    name = name.replace("florence-2", "florence2")
    return name


def fetch_image_from_r2(r2_key: str, dest: Path) -> bool:
    src = f"{RCLONE_R2}:{R2_OUTPUT_BUCKET}/{r2_key}"
    r = rclone(["copyto", src, str(dest), "--progress=false"], timeout=120)
    if r.returncode != 0:
        log(f"    WARN fetch failed for {r2_key}: rc={r.returncode} "
            f"{r.stderr[-160:].strip()}")
        return False
    return True


def upload_mask(local: Path, r2_key: str) -> bool:
    dst = f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}"
    r = rclone(["copyto", str(local), dst, "--progress=false"], timeout=120)
    if r.returncode != 0:
        log(f"    WARN mask upload {r2_key}: rc={r.returncode} "
            f"{r.stderr[-160:].strip()}")
        return False
    return True


# ---- Florence-2 helpers ----

def load_florence2(model_id: str, device: str):
    log(f"  loading {model_id} on {device} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    log(f"  loaded in {time.time() - t0:.1f}s")
    return model, processor


def florence2_segment(model, processor, image: Image.Image, prompt: str,
                      device: str) -> list[list[tuple[float, float]]]:
    """Returns a list of polygons (each a list of (x, y) vertices in image
    coords). Empty list if Florence-2 returned no region."""
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"
    full_prompt = f"{task}{prompt}"
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        gen = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
        )
    text = processor.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task, image_size=image.size,
    )
    out = parsed.get(task) or {}
    polygons_raw = out.get("polygons") or []
    polys: list[list[tuple[float, float]]] = []
    # Florence-2 returns: [ [ [x1,y1,x2,y2,...] ], ... ] — outer per-instance,
    # inner per-ring, flat coord list per ring.
    for inst in polygons_raw:
        for ring in inst:
            if len(ring) < 6:
                continue
            pts = [(float(ring[i]), float(ring[i + 1]))
                   for i in range(0, len(ring) - 1, 2)]
            polys.append(pts)
    return polys


def polygons_to_mask(polys: list[list[tuple[float, float]]],
                     size: tuple[int, int]) -> Image.Image:
    """Rasterize polygons into a binary L-mode PNG (255 = positive)."""
    mask = Image.new("L", size, 0)
    if not polys:
        return mask
    draw = ImageDraw.Draw(mask)
    for pts in polys:
        if len(pts) < 3:
            continue
        flat = [int(round(c)) for pt in pts for c in pt]
        draw.polygon(flat, fill=255)
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model", default="microsoft/Florence-2-large",
                    help="HF repo id for the segmenter")
    ap.add_argument("--prompt", default=None,
                    help="text prompt; defaults to the condition name")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required")
        return 2

    prompt = args.prompt or args.condition
    model_id = safe_model_id(args.model)
    log(f"auto-mask test: condition={args.condition} model_id={model_id} "
        f"prompt='{prompt}' device={args.device}")

    items = get_candidates(condition=args.condition,
                           model_version=model_id, limit=args.limit)
    if not items:
        log("nothing to do")
        return 0

    model, processor = load_florence2(args.model, args.device)

    n_done = n_hit = n_err = 0
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="arem-automask-") as scratch:
        scratch_p = Path(scratch)
        for idx, image_key in enumerate(items, start=1):
            t0 = time.time()
            local = scratch_p / f"img-{idx:05d}.jpg"
            if not fetch_image_from_r2(image_key, local):
                try:
                    post_result({
                        "condition": args.condition,
                        "imageR2Path": image_key,
                        "modelVersion": model_id,
                        "prompt": prompt,
                        "errorMessage": "fetch from R2 failed",
                    })
                except Exception as e:
                    log(f"    WARN POST: {e}")
                n_err += 1
                continue
            try:
                with Image.open(local) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    polys = florence2_segment(model, processor, im, prompt, args.device)
                mask = polygons_to_mask(polys, (w, h))
                px = int(np.asarray(mask, dtype=np.uint8).sum() // 255)
                mask_path: str | None = None
                if px > 0:
                    safe_name = re.sub(r"[^a-z0-9._-]+", "_",
                                       image_key.split("/")[-1].lower())
                    mask_key = (
                        f"auto-masks/{args.condition}/{model_id}/"
                        f"{safe_name.rsplit('.', 1)[0]}.png"
                    )
                    mask_local = scratch_p / f"mask-{idx:05d}.png"
                    mask.save(mask_local, format="PNG", optimize=True)
                    if upload_mask(mask_local, mask_key):
                        mask_path = mask_key
                runtime_ms = int((time.time() - t0) * 1000)
                post_result({
                    "condition": args.condition,
                    "imageR2Path": image_key,
                    "modelVersion": model_id,
                    "prompt": prompt,
                    "autoMaskR2Path": mask_path,
                    "pixelCount": px,
                    "imageWidth": w,
                    "imageHeight": h,
                    "runtimeMs": runtime_ms,
                })
                n_done += 1
                if px > 0:
                    n_hit += 1
                if idx % 10 == 0 or px > 0:
                    log(f"  [{idx}/{len(items)}] {image_key}: "
                        f"{px} px {'(hit)' if px > 0 else '(empty)'} "
                        f"{runtime_ms}ms")
            except Exception as e:
                log(f"  [{idx}/{len(items)}] {image_key}: ERROR {str(e)[:200]}")
                try:
                    post_result({
                        "condition": args.condition,
                        "imageR2Path": image_key,
                        "modelVersion": model_id,
                        "prompt": prompt,
                        "errorMessage": str(e)[:500],
                        "runtimeMs": int((time.time() - t0) * 1000),
                    })
                except Exception as e2:
                    log(f"    WARN POST: {e2}")
                n_err += 1
            finally:
                try:
                    local.unlink(missing_ok=True)
                except Exception:
                    pass

    dt = time.time() - t_start
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(items),1):.1f}s/image avg) ---")
    log(f"  processed: {n_done}  hits: {n_hit}  errors: {n_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
