"""VLM-grounded detection + SAM segmentation.

Pipeline per image:
  1. Send image to Claude (or compatible VLM) with a structured prompt
     describing the conditions we care about. Receive JSON list of
     {condition, bbox, confidence, description}.
  2. For each detected box matching the requested --condition, run SAM
     (transformers SamModel + bbox prompt) to get a pixel mask.
  3. Union the per-detection masks, upload, POST to /api/internal/auto-mask-test
     as modelVersion "vlm__<model-short>_<sam-short>__<tag>".

This is the *match-mode* play: detection by a model that already knows
what tripods, cameras, fingers, dead fixtures look like; SAM2 turns
bboxes into clean pixel masks. No per-condition fine-tuning required.

Usage:
    python -m scripts.vlm_detect_mask \\
        --condition reflection \\
        --vlm claude-sonnet-4-6 \\
        --sam facebook/sam-vit-base \\
        --target-set hand-labeled \\
        --tag v1

Env:
    WORKER_TOKEN              dashboard auth (required)
    ANTHROPIC_API_KEY         Anthropic API key (required for claude-*)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data (mask uploads)
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs (source images)
    RCLONE_R2                 default 'r2'
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from transformers import SamModel, SamProcessor  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # checked at runtime

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def log(msg: str) -> None:
    print(msg, flush=True)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {text[:300]}") from e


def rclone_fetch(r2_key: str, dest: Path, *, bucket: str) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", f"{RCLONE_R2}:{bucket}/{r2_key}", str(dest),
         "--progress=false"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        log(f"  WARN fetch {r2_key}: rc={r.returncode} {r.stderr[-160:].strip()}")
        return False
    return True


def rclone_upload(local: Path, r2_key: str) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", str(local), f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}",
         "--progress=false"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        log(f"  WARN upload {r2_key}: rc={r.returncode} {r.stderr[-160:].strip()}")
        return False
    return True


# ---- VLM prompt + parsing ----

# Per-condition descriptions tuned for the model to ground correctly.
# Add more here as we extend to other conditions.
CONDITION_PROMPTS: dict[str, str] = {
    "reflection": (
        "REFLECTIONS of the PHOTOGRAPHER or their equipment visible in any "
        "reflective surface in the image (mirrors, windows, glass shower doors, "
        "polished countertops, TV screens, picture frames, faucets, oven doors). "
        "ONLY count the photographer themselves, their CAMERA, their TRIPOD, or "
        "their human SILHOUETTE — NOT other items reflected in the surface (such "
        "as the room's lamps, walls, ceiling, furniture). NOT the reflective "
        "surface itself; ONLY the silhouette/equipment reflected on it. If you "
        "see a tripod sitting on the floor (not in a reflection), DO NOT count it."
    ),
    "dead_fixture": (
        "Light fixtures (lamps, sconces, chandeliers, ceiling lights, pendants) "
        "that are visibly DARK / NOT EMITTING LIGHT when nearby lights are on. "
        "Bulbs missing, fixture turned off, or fixture obviously dim relative "
        "to peers in the same room."
    ),
    "finger-in-photo": (
        "A human FINGER, hand, or part of a hand intruding into the frame "
        "(usually from a corner/edge), typically caused by the photographer's "
        "thumb covering part of the lens."
    ),
    "sun-flare": (
        "Lens flare, sun glare, bright sunburst, or hexagonal flare artifacts "
        "from the sun directly hitting the lens. NOT just bright sunlight."
    ),
    "lens-hood-in-photo": (
        "The dark crescent or ring of a lens hood / shade visible at the corners "
        "or edges of the frame."
    ),
}


def make_prompt(condition: str) -> str:
    desc = CONDITION_PROMPTS.get(
        condition,
        f"items matching the condition '{condition}'",
    )
    return (
        f"You are inspecting a real-estate listing photo for defects that need "
        f"retouching. Find all instances of: {desc}\n\n"
        f"For each one, return a tight bounding box around the defect itself "
        f"(not the surface containing it).\n\n"
        f"Return STRICT JSON only, no prose. Schema:\n"
        f'{{"detections": [{{"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0, '
        f'"description": "<one short phrase>"}}, ...]}}\n\n'
        f"Coordinates are normalized to [0, 1] with origin at TOP-LEFT, "
        f"x along width, y along height. If no defects of this type are "
        f"present, return {{\"detections\": []}}."
    )


def call_claude_vlm(client, model: str, image: Image.Image,
                    prompt: str, max_tokens: int = 1024) -> dict:
    # Resize to Claude's optimal long-edge ~1568px for cheaper requests.
    w, h = image.size
    max_side = max(w, h)
    target = 1568
    if max_side > target:
        scale = target / max_side
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return parse_json_blob(text)


def parse_json_blob(text: str) -> dict:
    """Tolerant JSON extraction. The model is asked for strict JSON but may
    wrap with ```json fences or trailing text. Pull the first {...} blob."""
    s = text.strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Find first balanced {...}
    start = s.find("{")
    if start < 0:
        return {"detections": []}
    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {"detections": []}
    try:
        return json.loads(s[start:end])
    except json.JSONDecodeError:
        return {"detections": []}


# ---- SAM ----

class SamBoxMasker:
    """SAM-from-bbox with selectable post-processing.

    Modes:
      single   — multimask_output=False; SAM returns one mask (typically
                 the dominant coherent object in the bbox). For
                 photographer-in-mirror this often becomes the whole
                 mirror, not the silhouette.
      smallest — multimask_output=True; SAM returns 3 masks at
                 whole/part/sub-part granularity. Pick the smallest by
                 pixel count. Closer to fine-grained items inside a
                 larger surface.
      bbox     — skip SAM entirely; just rasterize the bbox as a
                 rectangle. Rough but never over-expands.
    """

    def __init__(self, model_id: str, device: str, mode: str = "single"):
        self.mode = mode
        if mode == "bbox":
            log(f"  SAM mode=bbox (no model load)")
            self.model = None
            self.processor = None
            self.device = device
            return
        log(f"  loading {model_id} on {device} (sam mode={mode}) ...")
        t0 = time.time()
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.device = device
        log(f"  loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def mask(self, image: Image.Image,
             boxes_norm: list[list[float]]) -> np.ndarray:
        w, h = image.size
        if not boxes_norm:
            return np.zeros((h, w), dtype=bool)
        boxes_px = [[
            max(0.0, b[0] * w), max(0.0, b[1] * h),
            min(float(w), b[2] * w), min(float(h), b[3] * h),
        ] for b in boxes_norm]

        if self.mode == "bbox":
            m = np.zeros((h, w), dtype=bool)
            for x1, y1, x2, y2 in boxes_px:
                m[int(y1):int(y2), int(x1):int(x2)] = True
            return m

        inputs = self.processor(
            image.convert("RGB"),
            input_boxes=[boxes_px],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(
            **inputs,
            multimask_output=(self.mode == "smallest"),
        )
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (n_boxes, K, H, W) tensor of bool (K=1 single, K=3 smallest)

        if self.mode == "smallest":
            # For each box, pick the smallest of the K candidate masks.
            chosen = []
            for i in range(masks.shape[0]):
                cands = masks[i]  # (K, H, W)
                sizes = cands.reshape(cands.shape[0], -1).sum(dim=-1)
                k = int(sizes.argmin().item())
                chosen.append(cands[k])
            m = torch.stack(chosen, dim=0).any(dim=0).numpy().astype(bool)
        else:  # single
            m = masks.squeeze(1).any(dim=0).numpy().astype(bool)
        return m


# ---- candidate fetching ----

def get_hand_labeled(condition: str) -> list[dict]:
    sp = urllib.parse.urlencode({"condition": condition})
    items = _request(
        "GET", f"/api/internal/ground-truth-masks?{sp}"
    ).get("items", [])
    # We only need the source image path here
    return [{"imageR2Path": it["imageR2Path"]} for it in items]


def get_candidates(condition: str, model_version: str,
                   limit: int) -> list[dict]:
    """Pull both pending + done LabelingTask images, dedupe."""
    seen: dict[str, None] = {}
    for status in ("pending", "done"):
        sp = urllib.parse.urlencode({
            "condition": condition, "modelVersion": model_version,
            "status": status, "limit": str(limit),
        })
        resp = _request("GET", f"/api/internal/auto-mask-candidates?{sp}")
        for k in resp.get("items") or []:
            seen.setdefault(k, None)
        log(f"  candidates [{status}]: {resp.get('todo')}/{resp.get('total')} "
            f"todo (already done: {resp.get('alreadyDone')})")
    return [{"imageR2Path": k} for k in seen.keys()]


# ---- main ----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--vlm", default="claude-sonnet-4-6",
                    help="Anthropic model id. claude-opus-4-7 for highest "
                         "quality, claude-sonnet-4-6 for the price/perf "
                         "default, claude-haiku-4-5-20251001 for cheap.")
    ap.add_argument("--sam", default="facebook/sam-vit-base",
                    help="HF model id for SAM. base is fast (~375MB), "
                         "large/huge are slower but better. Ignored when "
                         "--sam-mode=bbox.")
    ap.add_argument("--sam-mode", choices=("single", "smallest", "bbox"),
                    default="single",
                    help="single: SAM picks the dominant object in bbox "
                         "(default; often segments the whole reflective surface). "
                         "smallest: SAM returns 3 masks at whole/part/subpart "
                         "granularity, pick smallest. bbox: skip SAM, rasterize "
                         "bbox as rectangle.")
    ap.add_argument("--target-set",
                    choices=("hand-labeled", "all-candidates"),
                    default="hand-labeled",
                    help="hand-labeled: only the 35 reflection rows we have "
                         "hand-masks for (fast smoke + IoU). all-candidates: "
                         "every LabelingTask image for the condition.")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--confidence-min", type=float, default=0.30,
                    help="drop detections below this confidence")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required")
        return 2
    if not ANTHROPIC_API_KEY:
        log("ERROR: ANTHROPIC_API_KEY env var is required for VLM calls")
        return 2
    if anthropic is None:
        log("ERROR: `anthropic` package not installed. pip install anthropic")
        return 2

    vlm_short = args.vlm.replace("claude-", "").replace("-", "")
    sam_short = (
        "bbox" if args.sam_mode == "bbox"
        else re.sub(r"[^a-z0-9]+", "-",
                    args.sam.split("/")[-1].lower()).strip("-")
        + (f"-{args.sam_mode}" if args.sam_mode != "single" else "")
    )
    model_version = f"vlm__{vlm_short}_{sam_short}__{args.tag}"
    log(f"VLM detect: condition={args.condition} vlm={args.vlm} "
        f"sam={args.sam} model_version={model_version}")

    if args.target_set == "hand-labeled":
        items = get_hand_labeled(args.condition)
        log(f"  hand-labeled rows: {len(items)}")
        # Skip ones we've already done
        sp = urllib.parse.urlencode({
            "condition": args.condition, "modelVersion": model_version,
            "listModels": "false", "limit": "2000",
        })
        done = _request("GET", f"/api/internal/auto-mask-rows?{sp}")
        done_set = {it["imageR2Path"] for it in done.get("items", [])}
        items = [it for it in items if it["imageR2Path"] not in done_set]
        log(f"  todo (after dedup): {len(items)}")
    else:
        items = get_candidates(args.condition, model_version, args.limit)
        log(f"  candidates: {len(items)}")

    if not items:
        log("  nothing to do")
        return 0

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sam = SamBoxMasker(args.sam, args.device, mode=args.sam_mode)
    prompt = make_prompt(args.condition)

    n_done = n_hit = n_err = 0
    n_detections_total = 0
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="arem-vlm-") as scratch:
        sp = Path(scratch)
        for idx, it in enumerate(items, start=1):
            t0 = time.time()
            image_key = it["imageR2Path"]
            img_local = sp / f"img-{idx:05d}.jpg"
            if not rclone_fetch(image_key, img_local, bucket=R2_OUTPUT_BUCKET):
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/auto-mask-test", {
                            "condition": args.condition,
                            "imageR2Path": image_key,
                            "modelVersion": model_version,
                            "prompt": prompt[:200] + "…",
                            "errorMessage": "fetch from R2 failed",
                        })
                    except Exception as e:
                        log(f"    WARN POST: {e}")
                n_err += 1
                continue
            try:
                with Image.open(img_local) as im:
                    im = im.convert("RGB")
                    orig_w, orig_h = im.size

                # VLM call
                t_vlm = time.time()
                parsed = call_claude_vlm(client, args.vlm, im, prompt)
                vlm_ms = int((time.time() - t_vlm) * 1000)
                detections = parsed.get("detections", []) or []
                # Filter by confidence + sanity-check bbox
                kept_boxes: list[list[float]] = []
                kept_descs: list[str] = []
                for d in detections:
                    try:
                        conf = float(d.get("confidence", 0.0))
                        bb = d.get("bbox", [])
                        if len(bb) != 4:
                            continue
                        bb = [float(x) for x in bb]
                        # clip and validate
                        bb = [
                            max(0.0, min(1.0, bb[0])),
                            max(0.0, min(1.0, bb[1])),
                            max(0.0, min(1.0, bb[2])),
                            max(0.0, min(1.0, bb[3])),
                        ]
                        if bb[2] <= bb[0] or bb[3] <= bb[1]:
                            continue
                        if conf < args.confidence_min:
                            continue
                        kept_boxes.append(bb)
                        kept_descs.append(str(d.get("description", "")))
                    except (TypeError, ValueError):
                        continue
                n_detections_total += len(kept_boxes)

                # SAM mask
                t_sam = time.time()
                mask_arr = sam.mask(im, kept_boxes)
                sam_ms = int((time.time() - t_sam) * 1000)
                mask_img = Image.fromarray(
                    (mask_arr.astype(np.uint8) * 255), mode="L")
                px = int(np.asarray(mask_img, dtype=np.uint8).sum() // 255)

                mask_path: str | None = None
                if px > 0:
                    safe_name = re.sub(r"[^a-z0-9._-]+", "_",
                                       image_key.split("/")[-1].lower())
                    mask_key = (
                        f"auto-masks/{args.condition}/{model_version}/"
                        f"{safe_name.rsplit('.', 1)[0]}.png"
                    )
                    mask_local = sp / f"mask-{idx:05d}.png"
                    mask_img.save(mask_local, format="PNG", optimize=True)
                    if not args.dry_run and rclone_upload(mask_local, mask_key):
                        mask_path = mask_key

                runtime_ms = int((time.time() - t0) * 1000)
                if not args.dry_run:
                    # Store bboxes inline (JSON) so future SAM-mode
                    # experiments can re-use them without re-calling the VLM.
                    bbox_payload = json.dumps(
                        [{"bbox": b, "desc": d}
                         for b, d in zip(kept_boxes, kept_descs)],
                        separators=(",", ":"),
                    )
                    _request("POST", "/api/internal/auto-mask-test", {
                        "condition": args.condition,
                        "imageR2Path": image_key,
                        "modelVersion": model_version,
                        "prompt": f"vlm:{args.vlm} | det={bbox_payload[:1800]}",
                        "autoMaskR2Path": mask_path,
                        "pixelCount": px,
                        "imageWidth": orig_w,
                        "imageHeight": orig_h,
                        "runtimeMs": runtime_ms,
                    })
                n_done += 1
                if px > 0:
                    n_hit += 1
                rate = idx / max(time.time() - t_start, 0.01)
                log(f"  [{idx}/{len(items)}] {image_key.rsplit('/', 1)[-1]}: "
                    f"{len(kept_boxes)} det → {px} px "
                    f"(vlm {vlm_ms}ms, sam {sam_ms}ms, total {runtime_ms}ms; "
                    f"{rate:.1f} img/s)")
            except Exception as e:
                log(f"  [{idx}/{len(items)}] {image_key}: ERROR {str(e)[:200]}")
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/auto-mask-test", {
                            "condition": args.condition,
                            "imageR2Path": image_key,
                            "modelVersion": model_version,
                            "prompt": f"vlm:{args.vlm}",
                            "errorMessage": str(e)[:500],
                            "runtimeMs": int((time.time() - t0) * 1000),
                        })
                    except Exception as e2:
                        log(f"    WARN POST: {e2}")
                n_err += 1
            finally:
                try:
                    img_local.unlink(missing_ok=True)
                except Exception:
                    pass

    dt = time.time() - t_start
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(items),1):.2f}s/img avg) ---")
    log(f"  processed: {n_done}  hits: {n_hit}  errors: {n_err}  "
        f"total_detections: {n_detections_total}")
    log(f"  model_version: {model_version}")
    log(f"  next: python -m scripts.score_against_ground_truth "
        f"--condition {args.condition} --heldout 0 "
        f"--model-version {model_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
