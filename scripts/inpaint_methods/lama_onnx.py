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
        # Mirror rapidraw's run_lama_inpainting exactly: cap long side at
        # 768, build a SQUARE tensor of size ceil_64(max(h,w)) with the
        # image placed top-left and the remainder edge-clamped, run, read
        # back the [0..h,0..w] region. The exported FFC ONNX is traced for
        # a square input and outputs values in the [0,255] range — feeding
        # a non-square tensor (the previous bug) produced a black blob.
        h0, w0 = image_bgr.shape[:2]
        long_side = max(h0, w0)
        scale = min(1.0, 768.0 / long_side)
        fw = max(1, int(round(w0 * scale)))
        fh = max(1, int(round(h0 * scale)))
        img = cv2.resize(image_bgr, (fw, fh), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask_u8, (fw, fh), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        msk_in = (msk > 0).astype(np.float32)

        td = align_64(max(fw, fh))
        # Edge-clamp pad to a square td×td (replicate the border, like
        # rapidraw's sx.min(fw-1) sampling).
        img_sq = cv2.copyMakeBorder(rgb, 0, td - fh, 0, td - fw,
                                    cv2.BORDER_REPLICATE)
        msk_sq = cv2.copyMakeBorder(msk_in, 0, td - fh, 0, td - fw,
                                    cv2.BORDER_REPLICATE)
        x = img_sq.transpose(2, 0, 1)[None]          # (1,3,td,td)
        m = msk_sq[None, None]                        # (1,1,td,td)
        out = self.session.run(None, {"image": x, "mask": m})
        y = out[0][0]                                 # (3,td,td), [0,255] range
        y = y[:, :fh, :fw]                            # drop the padded region
        rgb_out = np.clip(y.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        if (bgr_out.shape[1], bgr_out.shape[0]) != (w0, h0):
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out
