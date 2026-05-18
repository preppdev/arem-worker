"""50-image window-mask audit: OWLv2 detection + SAM segmentation.

Picks N random paired interior images from ImageReview, runs OWLv2
open-vocabulary detection on the prompt "a window", pipes the boxes through SAM,
uploads the union mask PNG to R2 at
    arem-training-data/auto-masks/window/florence2-sam2/<safe-key>.png
and POSTs to /api/internal/auto-mask-test so /auto-mask/window renders
the side-by-side gallery.

Usage:
    python -m scripts.audit_window_masks --n 50

Env:
    DATABASE_URL       (read from .env.local)
    WORKER_TOKEN       dashboard auth
    DASHBOARD_URL      default https://arem-editing-dashboard.vercel.app
    R2_BUCKET          default arem-training-data
    RCLONE_R2          default 'r2'
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from transformers import (  # type: ignore
    Owlv2ForObjectDetection,
    Owlv2Processor,
    SamModel,
    SamProcessor,
)

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
CF_HASH = "hfTNewiEIKKL4K9rrhwauQ"

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
SAM_MODEL = "facebook/sam-vit-base"
MODEL_VERSION = "owlv2-base+sam-vit-base"
OWL_PROMPT = "a window"
OWL_THRESHOLD = 0.15  # detection confidence floor; OWLv2 is generous with windows


def log(s: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)


# ---------- candidates ----------

def pick_candidates(conn, n: int) -> list[dict]:
    """Paired interior images, randomised, never re-audited by this model."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT ir.id, ir."productionR2Path", ir."cfImageId", ir."midStem"
        FROM "ImageReview" ir
        WHERE ir."productionR2Path" IS NOT NULL
          AND ir."vendorImageR2Path" IS NOT NULL
          AND ir."cfImageId" IS NOT NULL
          AND (ir."isInteriorCorrected" = true
               OR (ir."isInteriorCorrected" IS NULL AND ir."isInteriorWorker" = true))
          AND NOT EXISTS (
            SELECT 1 FROM "AutoMaskTest" amt
            WHERE amt.condition = 'window'
              AND amt."modelVersion" = %s
              AND amt."imageR2Path" = ir."productionR2Path"
          )
        ORDER BY random()
        LIMIT %s
        """,
        (MODEL_VERSION, n),
    )
    rows = list(cur.fetchall())
    cur.close()
    return rows


# ---------- image fetch ----------

def fetch_via_cf(cf_id: str, variant: str = "w2400") -> Image.Image:
    url = f"https://imagedelivery.net/{CF_HASH}/{cf_id}/{variant}"
    req = urllib.request.Request(url, headers={"User-Agent": "arem-window-audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


# ---------- OWLv2 ----------

class OwlGrounder:
    def __init__(self, device: str = "cuda"):
        log(f"  loading {OWL_MODEL} on {device} ...")
        t0 = time.time()
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).eval()
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.device = device
        log(f"  loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def boxes(self, image: Image.Image) -> list[list[float]]:
        inputs = self.processor(
            text=[[OWL_PROMPT]], images=image, return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs)
        # target_sizes is (height, width) per HF convention
        target = torch.tensor([[image.size[1], image.size[0]]])
        results = self.processor.post_process_grounded_object_detection(
            outputs=out,
            target_sizes=target,
            threshold=OWL_THRESHOLD,
            text_labels=[[OWL_PROMPT]],
        )[0]
        boxes_px = results["boxes"].cpu().numpy()  # (N, 4) xyxy pixel
        w, h = image.size
        return [[float(b[0]/w), float(b[1]/h), float(b[2]/w), float(b[3]/h)]
                for b in boxes_px]


# ---------- SAM-2 ----------

class SamSegmenter:
    def __init__(self, device: str = "cuda"):
        log(f"  loading {SAM_MODEL} on {device} ...")
        t0 = time.time()
        self.model = SamModel.from_pretrained(SAM_MODEL).to(device).eval()
        self.processor = SamProcessor.from_pretrained(SAM_MODEL)
        self.device = device
        log(f"  loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def mask(self, image: Image.Image,
             boxes_norm: list[list[float]]) -> np.ndarray:
        w, h = image.size
        if not boxes_norm:
            return np.zeros((h, w), dtype=bool)
        boxes_px = [[
            max(0.0, b[0]*w), max(0.0, b[1]*h),
            min(float(w), b[2]*w), min(float(h), b[3]*h),
        ] for b in boxes_norm]
        inputs = self.processor(
            image, input_boxes=[boxes_px], return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (n_boxes, 1, H, W)
        return masks.squeeze(1).any(dim=0).numpy().astype(bool)


# ---------- R2 upload ----------

def safe_key(production_r2_path: str) -> str:
    # Reuse the production key as-is, swap final .jpg/.jpeg/.png for .png
    base = re.sub(r"\.(jpg|jpeg|png)$", "", production_r2_path, flags=re.I)
    return base


def upload_mask(mask: np.ndarray, key: str) -> str:
    """Write the mask PNG to a temp file and rclone-copy it to R2."""
    r2_key = f"auto-masks/window/florence2-sam2/{key}.png"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(tf.name)
        tf.flush()
        dst = f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}"
        res = subprocess.run(
            ["rclone", "copyto", tf.name, dst,
             "--s3-no-check-bucket", "--no-traverse"],
            capture_output=True, text=True,
        )
        try:
            os.unlink(tf.name)
        except OSError:
            pass
    if res.returncode != 0:
        raise RuntimeError(
            f"rclone copyto failed ({res.returncode}): {res.stderr[:300]}"
        )
    return r2_key


# ---------- dashboard POST ----------

def post_result(payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{DASHBOARD_URL}/api/internal/auto-mask-test",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-worker-token": WORKER_TOKEN,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for var in ("DATABASE_URL", "WORKER_TOKEN"):
        if not os.environ.get(var):
            sys.exit(f"missing env: {var}")

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    candidates = pick_candidates(conn, args.n)
    conn.close()
    log(f"picked {len(candidates)} candidates")
    if not candidates:
        log("nothing to do — every paired interior already audited by this model")
        return

    grounder = OwlGrounder(device=args.device)
    sam = SamSegmenter(device=args.device)

    t0 = time.time()
    done = err = no_window = 0

    for i, row in enumerate(candidates, 1):
        prod_path = row["productionR2Path"]
        mid = row["midStem"]
        t_start = time.time()
        try:
            img = fetch_via_cf(row["cfImageId"])
            boxes = grounder.boxes(img)
            mask = sam.mask(img, boxes)
            n_pos = int(mask.sum())
            payload: dict[str, Any] = {
                "condition": "window",
                "imageR2Path": prod_path,
                "modelVersion": MODEL_VERSION,
                "prompt": OWL_PROMPT,
                "imageWidth": img.width,
                "imageHeight": img.height,
                "pixelCount": n_pos,
                "runtimeMs": int((time.time() - t_start) * 1000),
            }
            if n_pos > 0:
                r2_key = upload_mask(mask, safe_key(prod_path))
                payload["autoMaskR2Path"] = r2_key
            else:
                no_window += 1
            post_result(payload)
            done += 1
            log(f"  [{i}/{len(candidates)}] {mid} boxes={len(boxes)} px={n_pos} ({payload['runtimeMs']}ms)")
        except Exception as e:
            err += 1
            log(f"  [{i}/{len(candidates)}] ERR {mid}: {str(e)[:180]}")
            try:
                post_result({
                    "condition": "window",
                    "imageR2Path": prod_path,
                    "modelVersion": MODEL_VERSION,
                    "errorMessage": str(e)[:500],
                    "runtimeMs": int((time.time() - t_start) * 1000),
                })
            except Exception:
                pass

    log(f"finished: done={done} err={err} no-window={no_window} elapsed={time.time()-t0:.0f}s")
    log(f"view: {DASHBOARD_URL}/auto-mask/window")


if __name__ == "__main__":
    main()
