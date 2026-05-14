"""AREM Command Center media ingest.

Posts a batch of finished assets to CC's external ingest endpoint after
the editing pipeline has landed them in R2 + Cloudflare Images.

Contract (locked 2026-05-14 with the AREM CC team):
  POST <AREM_CC_BASE_URL>/api/external/shoots/<jobId>/media
    Authorization: Bearer <AREM_CC_INGEST_TOKEN>
    Body: {
      "jobId": "<Job.id>",
      "deliveryAt": "<iso8601-utc>",
      "assets": [
        {
          "kind": "photo",                    // singular; CC normalises plural
          "sortOrder": 1,                     // 1-based ordinal within shoot
          "r2Bucket": "arem-production-edit-jobs",
          "r2Key":   "tours/<jobId>/photos/0001-<midStem>.jpg",
          "cfImageId": "<id>",                // required for photo/floorplan/panorama/drone
          "mimeType": "image/jpeg",
          "width":  6048,
          "height": 4024,
          "sizeBytes": 3128492,
          "room": "bedroom",                  // optional; from classifier
          "isHero": false,
          "altText": null,
          "caption": null,
          "checksum": null                    // "<algo>:<hex>" or null
        },
        ...
      ]
    }

Response:
  200 — at least one asset accepted or already-registered:
    { "shootId", "accepted":[{r2Key, ccMediaId}], "skipped":[{r2Key, reason}],
      "errors":[{index, r2Key, reason}] }
  400 — every asset failed validation, or malformed body
  401 — bad/missing bearer
  404 — no Job with that ID

This module is best-effort: a 4xx/5xx is logged + returned to the
caller; the caller must NOT raise on failure since R2 + CF Images
have already succeeded by the time we POST here. The CC team can
re-fetch any missed assets via a later reconciliation pass.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import requests  # type: ignore


CC_BASE_URL = os.environ.get(
    "AREM_CC_BASE_URL", "https://aremcommandcenter.vercel.app").rstrip("/")
CC_INGEST_TOKEN = os.environ.get("AREM_CC_INGEST_TOKEN", "")


def _log(msg: str) -> None:
    print(msg, flush=True)


def jpeg_dims(local_path: str | Path) -> tuple[int | None, int | None]:
    """Read JPEG width/height without loading the full image. Returns
    (None, None) if PIL isn't installed or the file isn't readable."""
    try:
        from PIL import Image  # type: ignore
        with Image.open(local_path) as im:
            return im.width, im.height
    except Exception:
        return None, None


def post_media(*, job_id: str, assets: list[dict[str, Any]],
               delivery_at: str | None = None,
               timeout: int = 30) -> dict[str, Any]:
    """POST a batch of asset records to CC. Returns CC's JSON response
    on 2xx OR a synthetic { "error": "..." } dict on transport / 4xx /
    5xx failure. Callers should log and proceed; do not raise."""
    if not CC_INGEST_TOKEN:
        return {"error": "AREM_CC_INGEST_TOKEN not set; skipping CC POST"}
    if not assets:
        return {"error": "no assets to post"}
    url = f"{CC_BASE_URL}/api/external/shoots/{job_id}/media"
    body = {
        "jobId": job_id,
        "deliveryAt": delivery_at or _dt.datetime.now(_dt.timezone.utc)
                                          .isoformat().replace("+00:00", "Z"),
        "assets": assets,
    }
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {CC_INGEST_TOKEN}",
                     "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except Exception as e:
        _log(f"  WARN CC ingest transport: {str(e)[:200]}")
        return {"error": f"transport: {str(e)[:200]}"}
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text[:200]}
    if resp.status_code >= 400:
        _log(f"  WARN CC ingest HTTP {resp.status_code}: "
             f"{str(data)[:300]}")
        return {"error": f"HTTP {resp.status_code}", **data}
    return data
