"""FLUX.1-Kontext-dev — instruction-based editor. Included in the bake-off
as the current production baseline (what scripts/inpaint_flux_local.py
uses). Mask is ignored at inference time; the model is steered by a
textual instruction. We synthesize a removal instruction per condition.
"""
from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
MAX_DIM = 1280
STEPS = 28
GUIDANCE = 2.5

REMOVE_PROMPT = (
    "Remove the highlighted object from this real estate photo. "
    "Replace it with the surrounding background — wall, floor, sky, or "
    "whatever surface lies behind it would naturally continue. Leave "
    "all other parts of the photo completely unchanged."
)


def round_to_16(x: int) -> int:
    return ((x + 15) // 16) * 16


class FluxKontext:
    name = "flux_kontext"
    backend = "diffusion"

    def __init__(self) -> None:
        self.pipe = None

    def load(self) -> None:
        from diffusers import FluxKontextPipeline  # type: ignore
        from scripts.inpaint_methods._flux_common import load_flux_nf4
        self.pipe = load_flux_nf4(FluxKontextPipeline, MODEL_ID)

    def unload(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()

    def run(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        from PIL import Image  # type: ignore
        h0, w0 = image_bgr.shape[:2]
        long_side = max(h0, w0)
        scale = min(1.0, MAX_DIM / long_side)
        new_w = round_to_16(int(w0 * scale))
        new_h = round_to_16(int(h0 * scale))
        img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Highlight the mask region in red so the instruction has something
        # to anchor to. (Kontext doesn't take a mask channel — only an
        # input image + prompt.)
        msk = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        overlay = img.copy()
        red = np.zeros_like(img)
        red[..., 2] = 220  # red channel in BGR
        overlay[msk > 8] = cv2.addWeighted(img[msk > 8], 0.4,
                                            red[msk > 8], 0.6, 0)
        img_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        out = self.pipe(
            prompt=REMOVE_PROMPT,
            image=img_pil,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator("cuda").manual_seed(12345),
        ).images[0]
        bgr_out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        if (bgr_out.shape[1], bgr_out.shape[0]) != (w0, h0):
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out
