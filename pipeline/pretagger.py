"""
Thin client for the local arem-pretagger inference service.

Called once per output image after Stage 2 + upright + production
upload completes. The pre-tag result lands on ImageReview.preTagConfidence
via /api/internal/pretag-result on the dashboard.

Fail-soft posture:
  - if PRETAG_URL / PRETAG_TOKEN aren't set, no-op (pretagger disabled)
  - if the service is down, log and continue (no pre-tags this run)
  - if /pretag returns an error, log and continue
  - if the dashboard POST fails, log and continue
The editing pipeline never blocks on the pretagger.
"""

from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Defaults match the on-box install:
#   PRETAG_URL=http://127.0.0.1:8090
PRETAG_URL   = os.environ.get("PRETAG_URL", "http://127.0.0.1:8090").rstrip("/")
PRETAG_TOKEN = os.environ.get("PRETAG_TOKEN", "")
# How long we'll wait for one pre-tag inference. Real MobileSAM +
# YOLOv8n on a 26 MP image is ~200-500 ms; 5 s is a generous ceiling.
PRETAG_TIMEOUT_S = float(os.environ.get("PRETAG_TIMEOUT_S", "5.0"))


def pretagger_enabled() -> bool:
    return bool(PRETAG_TOKEN)


def pretag_image(image_path: Path) -> Optional[dict]:
    """POST the file to the local pretagger /pretag endpoint. Returns
    the parsed JSON response, or None on any failure."""
    if not pretagger_enabled():
        return None
    if not image_path.is_file():
        return None

    boundary = f"----arem-pretag-{int(time.time() * 1000)}"
    body = _build_multipart(image_path, boundary)
    req = urllib.request.Request(
        f"{PRETAG_URL}/pretag",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Pretag-Token": PRETAG_TOKEN,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=PRETAG_TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")[:300]
        print(f"[pretagger] HTTP {e.code}: {text}", flush=True)
        return None
    except Exception as e:
        print(f"[pretagger] {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None


def _build_multipart(image_path: Path, boundary: str) -> bytes:
    """Minimal multipart/form-data encoder — one file field named 'image'."""
    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        'Content-Type: image/jpeg\r\n\r\n'.encode()
    )
    parts.append(image_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts)


def to_dashboard_payload(
    job_id: str,
    mid_stem: str,
    pretag_response: dict,
) -> dict:
    """Convert the pretagger /pretag response into the shape the
    dashboard's /api/internal/pretag-result expects."""
    conditions = {}
    for cond, pred in (pretag_response.get("conditions") or {}).items():
        # Pretagger may evolve; tolerate either {"confidence": ...} or
        # just a float.
        if isinstance(pred, dict):
            conditions[cond] = {
                "confidence": float(pred.get("confidence", 0.0)),
                "mask_r2_path": pred.get("mask_r2_path"),
                "bboxes": pred.get("bboxes"),
            }
        else:
            conditions[cond] = {"confidence": float(pred)}
    return {
        "jobId": job_id,
        "midStem": mid_stem,
        "conditions": conditions,
        "modelVersions": pretag_response.get("model_versions") or {},
    }
