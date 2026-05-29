"""SAM2 box-prompt → tight binary object mask.

Takes an image + a bounding box (pixel or fractional coords) and returns
a clean uint8 mask (0/255) of the dominant object inside the box. Used by
the inpaint sandbox so a rough user-drawn box becomes a tight removal
mask, instead of the fragmented diff-derived masks.

Loads the HuggingFace Sam2 weights downloaded to
/home/jordan/sky_models/sam2_hiera_base_plus.
"""
from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

SAM2_PATH = "/home/jordan/sky_models/sam2_hiera_base_plus"

_MODEL = None
_PROCESSOR = None


def _device() -> str:
    import torch  # type: ignore
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sam2():
    """Lazy-load and cache the SAM2 model + processor."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    import torch  # type: ignore
    from transformers import Sam2Model, Sam2Processor  # type: ignore
    _MODEL = Sam2Model.from_pretrained(SAM2_PATH, torch_dtype=torch.float32).to(_device()).eval()
    _PROCESSOR = Sam2Processor.from_pretrained(SAM2_PATH)
    return _MODEL, _PROCESSOR


def _to_pixel_box(bbox, w: int, h: int) -> list[float]:
    """Accept either {x,y,w,h} fractional, [x0,y0,x1,y1] px, or
    [x,y,w,h] px and return [x0,y0,x1,y1] in pixels."""
    if isinstance(bbox, dict):
        x = bbox["x"]; y = bbox["y"]; bw = bbox["w"]; bh = bbox["h"]
        # Fractional if all <= 1.0 (with small tolerance).
        if max(x + bw, y + bh) <= 1.5:
            return [x * w, y * h, (x + bw) * w, (y + bh) * h]
        return [x, y, x + bw, y + bh]
    seq = list(bbox)
    if len(seq) == 4:
        # Heuristic: if values look fractional, scale up.
        if max(seq) <= 1.5:
            return [seq[0] * w, seq[1] * h, seq[2] * w, seq[3] * h]
        return [float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])]
    raise ValueError(f"unrecognized bbox: {bbox!r}")


def box_to_mask(image_bgr: np.ndarray, bbox, dilate_px: int = 10) -> np.ndarray:
    """Return a uint8 0/255 mask of the object SAM2 finds inside `bbox`.

    `dilate_px` slightly grows the mask so the inpainter also covers the
    object's thin edges / contact shadow rim. Falls back to a filled box
    if SAM returns an empty/degenerate mask.
    """
    import torch  # type: ignore
    model, processor = load_sam2()
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = _to_pixel_box(bbox, w, h)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image  # type: ignore
    pil = Image.fromarray(rgb)
    inputs = processor(
        images=pil,
        input_boxes=[[[x0, y0, x1, y1]]],
        return_tensors="pt",
    ).to(_device())
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]  # tensor (num_boxes, 1, H, W) or (1, H, W)
    m = masks
    if hasattr(m, "numpy"):
        m = m.numpy()
    m = np.asarray(m)
    # Squeeze to (H, W).
    while m.ndim > 2:
        m = m[0]
    mask = (m > 0.5).astype(np.uint8) * 255
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if int((mask > 0).sum()) < 0.0005 * w * h:
        # SAM found nothing meaningful — fall back to a filled box.
        mask = np.zeros((h, w), np.uint8)
        mask[int(y0):int(y1), int(x0):int(x1)] = 255
    # Fill internal holes. SAM often masks an object's outer surface but
    # leaves textured/specular sub-parts as unmasked islands (e.g. a lamp's
    # body inside its masked shade). Those islands survive the removal as
    # mangled ghosts, so close them. Flood-fill from the border: anything
    # the fill can't reach is an interior hole.
    ff = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 255)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(ff))
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--box", required=True,
                    help="x0,y0,x1,y1 (pixels) or fractional x,y,w,h with --frac")
    ap.add_argument("--frac", action="store_true")
    ap.add_argument("--out", default="/tmp/sam_mask.png")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    parts = [float(v) for v in a.box.split(",")]
    box = {"x": parts[0], "y": parts[1], "w": parts[2], "h": parts[3]} if a.frac else parts
    mask = box_to_mask(img, box)
    cv2.imwrite(a.out, mask)
    print(f"mask: {(mask>0).sum()} px -> {a.out}")
