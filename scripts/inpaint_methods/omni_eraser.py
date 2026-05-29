"""OmniEraser (arXiv 2501.07397) — FLUX.1 backbone + Alimama ControlNet,
finetuned on Video4Removal (paired video frames where target is the
empty background). The intent of the model is "leave empty space" rather
than "creatively fill" — which is exactly what we want for object
removal.

Note: as of writing, the upstream HF model id is not yet permanent.
We rely on the FLUX.1-Fill-dev base weights + a downloaded OmniEraser
ControlNet checkpoint. Fallback: until the checkpoint is staged, this
method runs FLUX-Fill with a stronger "empty" prompt as a placeholder.
"""
from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

CONTROLNET_PATH = Path("/home/jordan/inpaint_models/omnieraser_controlnet")
FLUX_BASE = Path("/home/jordan/inpaint_models/flux_fill_dev")
MAX_DIM = 1280
STEPS = 28
GUIDANCE = 30.0


def round_to_16(x: int) -> int:
    return ((x + 15) // 16) * 16


class OmniEraser:
    name = "omni_eraser"
    backend = "diffusion"

    def __init__(self) -> None:
        self.pipe = None
        self._fallback_to_flux = False

    def load(self) -> None:
        # If the OmniEraser ControlNet checkpoint isn't downloaded yet,
        # fall back to Flux-Fill with the OmniEraser "empty" prompt — this
        # still gives us a removal-biased baseline pending checkpoint stage.
        from diffusers import FluxFillPipeline  # type: ignore
        if not CONTROLNET_PATH.is_dir():
            self._fallback_to_flux = True
            from scripts.inpaint_methods._flux_common import load_flux_nf4
            self.pipe = load_flux_nf4(FluxFillPipeline, str(FLUX_BASE))
            return
        # When the OmniEraser CN is available, load via ControlNet branch.
        # (Wire-up here is left as a TODO until we have the checkpoint.)
        from diffusers import FluxControlNetInpaintPipeline  # type: ignore
        from diffusers.models import FluxControlNetModel  # type: ignore
        cn = FluxControlNetModel.from_pretrained(
            str(CONTROLNET_PATH), torch_dtype=torch.bfloat16)
        self.pipe = FluxControlNetInpaintPipeline.from_pretrained(
            str(FLUX_BASE), controlnet=cn, torch_dtype=torch.bfloat16)
        self.pipe.enable_sequential_cpu_offload()

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
        msk = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        msk_pil = Image.fromarray((msk > 8).astype(np.uint8) * 255).convert("L")
        # OmniEraser's published inference: empty prompt, high guidance,
        # base ControlNet weight 1.0 (the "leave empty" bias is in the
        # checkpoint itself).
        empty_prompt = ""
        kwargs = dict(
            prompt=empty_prompt,
            image=img_pil,
            mask_image=msk_pil,
            height=new_h, width=new_w,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator("cuda").manual_seed(12345),
        )
        if self._fallback_to_flux:
            # Without the CN, push the prompt to be more removal-biased.
            kwargs["prompt"] = (
                "empty background, continue the existing wall pattern, "
                "no objects, no people, no furniture, photoreal"
            )
        out = self.pipe(**kwargs).images[0]
        bgr_out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        if (bgr_out.shape[1], bgr_out.shape[0]) != (w0, h0):
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out
