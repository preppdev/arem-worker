"""FLUX.1-Fill-dev — official mask-based fill model from BFL.

Differs from FLUX.1-Kontext (instruction-based): takes (image, mask)
and a prompt; the prompt steers the fill content (we use "empty
background" to push toward removal rather than creative replacement).
"""
from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

MODEL_PATH = Path("/home/jordan/inpaint_models/flux_fill_dev")
PROMPT_REMOVE = (
    "empty background, photorealistic, continue the surrounding pattern, "
    "no object, no people, professional real-estate interior photo"
)
MAX_DIM = 1280  # FLUX Fill works best ≤1440; we leave headroom.
STEPS = 28
GUIDANCE = 30.0  # FLUX Fill uses higher guidance than Kontext.


def round_to_16(x: int) -> int:
    return ((x + 15) // 16) * 16


class FluxFillDev:
    name = "flux_fill"
    backend = "diffusion"

    def __init__(self) -> None:
        self.pipe = None

    def load(self) -> None:
        from diffusers import FluxFillPipeline  # type: ignore
        from scripts.inpaint_methods._flux_common import load_flux_nf4
        self.pipe = load_flux_nf4(FluxFillPipeline, str(MODEL_PATH))

    def unload(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()

    def run(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        from PIL import Image  # type: ignore
        h0, w0 = image_bgr.shape[:2]
        # Resize to FLUX-friendly dims.
        long_side = max(h0, w0)
        scale = min(1.0, MAX_DIM / long_side)
        new_w = round_to_16(int(w0 * scale))
        new_h = round_to_16(int(h0 * scale))
        img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        msk_pil = Image.fromarray((msk > 8).astype(np.uint8) * 255).convert("L")
        out = self.pipe(
            prompt=PROMPT_REMOVE,
            image=img_pil,
            mask_image=msk_pil,
            height=new_h,
            width=new_w,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator("cuda").manual_seed(12345),
        ).images[0]
        bgr_out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        if (bgr_out.shape[1], bgr_out.shape[0]) != (w0, h0):
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out
