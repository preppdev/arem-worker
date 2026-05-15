"""AREM Command Center media ingest.

Posts a batch of finished assets to CC's external ingest endpoint after
the editing pipeline has landed them in R2 + Cloudflare Images.

Contract (locked 2026-05-14 with the AREM CC team, find-or-create
amendment 2026-05-14 evening):

  POST <AREM_CC_BASE_URL>/api/external/shoots/{key}/media
    Authorization: Bearer <AREM_CC_INGEST_TOKEN>

  {key} is one of:
    - a previously-returned CC shootId (cuid)         → fast path, no dedup
    - the literal string "new"                         → dedup via ensureJob

  Body: {
    "ensureJob": {                            // required when {key}=="new"
      "dropboxPath": "AREM (Spiro Uploads)/...",
      "address": "104 E Severn Rd",
      "city": null, "state": null, "zip": null,
      "photographerName": "Blair Hartzell",
      "completedAt": "2026-05-14T22:00:00Z"
    },
    "deliveryAt": "<iso8601-utc>",
    "assets": [
      {
        "kind": "photo",                      // singular; CC normalises plural
        "sortOrder": 1,                       // 1-based ordinal within shoot
        "r2Bucket": "arem-production-edit-jobs",
        "r2Key":   "tours/<jobId>/photos/0001-<midStem>.jpg",
        "cfImageId": "<id>",                  // required for photo/floorplan/panorama/drone
        "mimeType": "image/jpeg",
        "width":  6048,
        "height": 4024,
        "sizeBytes": 3128492,
        "room": "bedroom",                    // optional; from classifier
        "isHero": false,
        "altText": null,
        "caption": null,
        "checksum": null                      // "<algo>:<hex>" or null
      },
      ...
    ]
  }

  CC's Shoot ↔ our Job.id are SEPARATE id spaces — there's no overlap.
  Dedup precedence: ensureJob.dropboxPath > address+state+completedAt >
  create-new-stub.

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
import re
from pathlib import Path
from typing import Any

import requests  # type: ignore


CC_BASE_URL = os.environ.get(
    "AREM_CC_BASE_URL", "https://aremcommandcenter.vercel.app").rstrip("/")
CC_INGEST_TOKEN = os.environ.get("AREM_CC_INGEST_TOKEN", "")


def _log(msg: str) -> None:
    print(msg, flush=True)


_SHOOT_FOLDER_RE = re.compile(
    r"^(?P<date>\d{2}\.\d{2}\.\d{4})\s+"
    r"(?P<num>\d+)\s+"
    r"(?P<addr_slug>[^()]+?)"
    r"(?:\s+\((?P<agent>[^)]+)\))?\s*$"
)


def parse_shoot_folder(dropbox_path: str | None) -> dict[str, str | None]:
    """Best-effort parse of a Spiro shoot folder name. Returns a dict
    with photographer / shootDate / address / agent / shootNum keys
    when the path matches our standard layout:
      AREM (Spiro Uploads)/<year>/<quarter>/<Photographer>/<MM.DD.YYYY NN address-slug  (agent))
    Any field that can't be parsed is None — the caller decides what
    to do with the gaps."""
    out: dict[str, str | None] = {
        "photographer": None, "shootDate": None, "shootNum": None,
        "address": None, "agent": None,
    }
    if not dropbox_path:
        return out
    parts = dropbox_path.strip("/").split("/")
    # photographer is the segment just before the shoot folder
    if len(parts) >= 2:
        out["photographer"] = parts[-2]
    shoot_dir = parts[-1] if parts else ""
    m = _SHOOT_FOLDER_RE.match(shoot_dir)
    if not m:
        return out
    out["shootDate"] = m.group("date")
    out["shootNum"] = m.group("num")
    addr_slug = (m.group("addr_slug") or "").strip()
    # "104-e-severn-rd" → "104 E Severn Rd"
    out["address"] = " ".join(w.capitalize() for w in addr_slug.split("-")) or None
    out["agent"] = m.group("agent")
    return out


def build_ensure_job(*, editor_job_id: str | None,
                     dropbox_path: str | None,
                     photographer: str | None = None,
                     completed_at: str | None = None) -> dict[str, Any]:
    """Build the ensureJob block from what we have.

    editorJobId is OUR Job.id (cuid). CC dedups in this order:
      editorJobId → orderId → dropboxPath → address+state+completedAt → create
    editorJobId is exact-match and the most reliable signal, so we
    always send it. Address / photographer / completedAt are softer
    fallbacks for the very first POST before CC has seen this job.

    CC's Zod schema rejects null values on optional fields — only
    include keys whose value is non-empty. completedAt falls back to
    "now" (UTC) when not provided since the live worker POSTs at job-
    completion time and Job.completedAt isn't set yet at that moment.
    """
    parsed = parse_shoot_folder(dropbox_path)
    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    out: dict[str, Any] = {
        "editorJobId": editor_job_id,
        "dropboxPath": dropbox_path,
        "photographerName": photographer or parsed["photographer"],
        "completedAt": completed_at or now_iso,
        "address": parsed["address"],
    }
    return {k: v for k, v in out.items() if v is not None and v != ""}


def jpeg_dims(local_path: str | Path) -> tuple[int | None, int | None]:
    """Read JPEG width/height without loading the full image. Returns
    (None, None) if PIL isn't installed or the file isn't readable."""
    try:
        from PIL import Image  # type: ignore
        with Image.open(local_path) as im:
            return im.width, im.height
    except Exception:
        return None, None


def post_media(*, assets: list[dict[str, Any]],
               shoot_key: str = "new",
               ensure_job: dict[str, Any] | None = None,
               delivery_at: str | None = None,
               timeout: int = 30) -> dict[str, Any]:
    """POST a batch of asset records to CC.

    shoot_key:
      - "new" (default): CC will dedup-or-create via ensureJob
      - a CC shootId (cuid): bypasses dedup, requires the shoot to exist

    Returns CC's JSON response on 2xx OR a synthetic
    { "error": "..." } dict on transport / 4xx / 5xx failure. Callers
    should log and proceed; do not raise."""
    if not CC_INGEST_TOKEN:
        return {"error": "AREM_CC_INGEST_TOKEN not set; skipping CC POST"}
    if not assets:
        return {"error": "no assets to post"}
    if shoot_key == "new" and not ensure_job:
        return {"error": "ensure_job required when shoot_key=='new'"}

    url = f"{CC_BASE_URL}/api/external/shoots/{shoot_key}/media"
    body: dict[str, Any] = {
        "deliveryAt": delivery_at or _dt.datetime.now(_dt.timezone.utc)
                                          .isoformat().replace("+00:00", "Z"),
        "assets": assets,
    }
    if ensure_job:
        body["ensureJob"] = ensure_job
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
