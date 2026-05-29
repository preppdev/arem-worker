"""Bria Eraser API — mask-based object removal (premium, full-resolution).

POST https://engine.prod.bria-api.com/v2/image/edit/erase
  image + mask (white=erase, black=keep) → object erased, background
  continued, returned at original resolution. Commercially-safe model
  (licensed training data, standard indemnification on the self-serve tier).

Auth: BRIA_API_KEY env (header `api_token`). Self-serve key from
platform.bria.ai; 100 free generations, then $0.02/image.

Unlike the local models this is an API call — no GPU. We send the ROI
crop as base64 (smaller payload than the full 4000x6000 image; Bria
preserves non-masked pixels regardless) with sync=true and download the
returned image_url.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

import cv2  # type: ignore
import numpy as np  # type: ignore

ENDPOINT = "https://engine.prod.bria-api.com/v2/image/edit/erase"
USER_AGENT = "BriaPlatform/Sandbox/LLMsAgent"


class BriaErase:
    name = "bria"
    backend = "api"

    def __init__(self) -> None:
        self.key: str | None = None

    def load(self) -> None:
        self.key = os.environ.get("BRIA_API_KEY")
        if not self.key:
            raise RuntimeError("BRIA_API_KEY not set — add it to the worker env")

    def unload(self) -> None:
        self.key = None

    def run(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        ok_i, ib = cv2.imencode(".jpg", image_bgr,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ok_m, mb = cv2.imencode(".png", (mask_u8 > 8).astype(np.uint8) * 255)
        if not (ok_i and ok_m):
            raise RuntimeError("encode failed")
        payload = json.dumps({
            "image": base64.b64encode(ib.tobytes()).decode(),
            "mask": base64.b64encode(mb.tobytes()).decode(),
            "mask_type": "manual",
            "sync": True,
            "preserve_alpha": False,
            "visual_input_content_moderation": False,
            "visual_output_content_moderation": False,
        }).encode()
        req = urllib.request.Request(
            ENDPOINT, data=payload, method="POST",
            headers={
                "api_token": self.key or "",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"bria {e.code}: {body}") from None
        url = (data.get("result") or {}).get("image_url")
        if not url:
            raise RuntimeError(f"bria: no image_url in response: {str(data)[:200]}")
        with urllib.request.urlopen(url, timeout=120) as r:
            buf = r.read()
        arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError("bria result decode failed")
        return arr
