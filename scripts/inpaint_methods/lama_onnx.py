"""LaMa (FP16 ONNX) inpainter — RapidRAW's model, cached locally.

CPU-by-default ONNXRuntime; flips to CUDAExecutionProvider if available.
Resolution capped at 1536 px on long side for speed (rapidraw uses 768
but on a 3090 we can push higher); 64-px aligned.
"""
from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore


MODEL_PATH = Path("/home/jordan/inpaint_models/lama_rapidraw/lama_fp16.onnx")
MAX_DIM = 1536


def align_64(x: int) -> int:
    return ((x + 63) // 64) * 64


class LamaOnnx:
    name = "lama"
    backend = "onnx"

    def __init__(self) -> None:
        self.session = None

    def load(self) -> None:
        import onnxruntime as ort  # type: ignore
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        try:
            self.session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(MODEL_PATH),
                                                providers=["CPUExecutionProvider"])

    def unload(self) -> None:
        self.session = None

    def run(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        h0, w0 = image_bgr.shape[:2]
        # Cap long side, 64-aligned.
        long_side = max(h0, w0)
        scale = min(1.0, MAX_DIM / long_side)
        if scale < 1.0:
            new_w, new_h = int(w0 * scale), int(h0 * scale)
        else:
            new_w, new_h = w0, h0
        new_w = align_64(new_w)
        new_h = align_64(new_h)
        img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # LaMa expects RGB float32 in [0,1] NCHW, mask as 1-channel binary.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        msk_in = (msk > 8).astype(np.float32)
        x = rgb.transpose(2, 0, 1)[None]  # (1,3,H,W)
        m = msk_in[None, None]            # (1,1,H,W)
        out = self.session.run(None, {"image": x, "mask": m})
        y = out[0][0]  # (3,H,W) float32 in [0,1] (the rapidraw model is normalized)
        if y.max() > 1.5:  # some variants return 0-255
            y = y / 255.0
        rgb_out = np.clip(y.transpose(1, 2, 0), 0, 1)
        bgr_out = cv2.cvtColor((rgb_out * 255.0).astype(np.uint8),
                                cv2.COLOR_RGB2BGR)
        # Resize back to original crop dims.
        if (bgr_out.shape[1], bgr_out.shape[0]) != (w0, h0):
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out
