"""ObjectClear (arXiv 2505.22636) — SDXL-based inpainter with object-effect
attention + attention-guided fusion. Uses the upstream custom pipeline
(github.com/zjx0101/ObjectClear), trained on OBER which pairs objects with
their shadows/reflections, so the model removes both in one shot.

The model was trained at 512×512; the upstream inference resizes the
short side to 512 for best results.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

REPO_PATH = "/home/jordan/inpaint_models/ObjectClear_repo"
MODEL_PATH = "/home/jordan/inpaint_models/objectclear"
STEPS = 20
GUIDANCE = 2.5
# ObjectClear trained at 512; the ROI crop is resized so its short side
# hits this before inference, then upscaled back. Higher = sharper fill
# but further off the training distribution. Override via env for testing.
SHORT_SIDE = int(os.environ.get("OBJECTCLEAR_SHORT_SIDE", "512"))


def _resize_short_side(img: np.ndarray, target: int,
                       interp: int) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)
    scale = target / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # Dims must be divisible by 16: the VAE downsamples by 8, and the AGF
    # step (resize_attn_map_divide2) further halves the latent (H//2, W//2),
    # so an odd latent dimension throws a shape error mid-denoise — which
    # leaves the UNet attention-wrapped and cascades into recursion on the
    # next frame. 16-alignment guarantees even latents.
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


class ObjectClear:
    name = "object_clear"
    backend = "diffusion"

    def __init__(self) -> None:
        self.pipe = None
        self.generator = None

    def load(self) -> None:
        if REPO_PATH not in sys.path:
            sys.path.insert(0, REPO_PATH)
        from objectclear.pipelines import ObjectClearPipeline  # type: ignore
        self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            MODEL_PATH,
            torch_dtype=torch.float16,
            apply_attention_guided_fusion=True,
            variant="fp16",
        )
        self.pipe.to("cuda")
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self.generator = torch.Generator(device="cuda").manual_seed(12345)

    def unload(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()

    def _restore_unet(self) -> None:
        """Idempotently undo the AGF attention wrapping. Safe to call after
        either a successful run (already restored — no-op) or a failed one."""
        p = self.pipe
        if p is None:
            return
        orig = getattr(p, "original_state", None)
        if not orig:
            return
        try:
            p.unet = p.unet_restore_attention_processor(p.unet, orig)
            p.clear_cross_attention_scores(p.cross_attention_scores)
        except Exception:
            pass
        p.original_state = None

    def run(self, image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        from PIL import Image  # type: ignore
        h0, w0 = image_bgr.shape[:2]
        img_r = _resize_short_side(image_bgr, SHORT_SIDE, cv2.INTER_CUBIC)
        msk_r = _resize_short_side(
            (mask_u8 > 8).astype(np.uint8) * 255, SHORT_SIDE, cv2.INTER_NEAREST)
        h, w = img_r.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(msk_r).convert("L")
        try:
            result = self.pipe(
                prompt="remove the instance of object",
                image=image_pil,
                mask_image=mask_pil,
                generator=self.generator,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                height=h,
                width=w,
                return_attn_map=False,
            )
        finally:
            # Safety net: if the pipe threw mid-denoise after wrapping the
            # UNet attention but before its last-timestep restore, undo the
            # wrap here so the next frame starts clean (otherwise wrappers
            # stack and recurse).
            self._restore_unet()
        out_pil = result.images[0]
        bgr_out = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        if (bgr_out.shape[1], bgr_out.shape[0]) != (w0, h0):
            bgr_out = cv2.resize(bgr_out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return bgr_out
