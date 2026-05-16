"""Self-hosted Qwen-Image-Edit (Apache 2.0) inpaint test.

Mirror of scripts/inpaint_text_edit.py but runs the model locally via
diffusers' QwenImageEditPipeline instead of calling Replicate. Designed
to A/B against the FLUX Kontext Replicate run we already have in R2.

Outputs land at the same R2 path scheme:
    inpaint-tests/<condition>/qwen-image-edit__<tag>/<stem>-edited.jpg
    inpaint-tests/<condition>/qwen-image-edit__<tag>/<stem>-3up.jpg

If --combine-with-flux <prefix> is set, builds a 4-up composite that
pulls the matching FLUX-Kontext result from R2 and stacks
source | FLUX | Qwen | vendor for direct comparison.

Memory: a 20B model in bf16 is ~40 GB — won't fit on a 24 GB 3090.
Defaults to --offload model (diffusers enable_model_cpu_offload) which
swaps modules in/out of VRAM. Slower than pure GPU but fits 24 GB and
keeps quality at full precision. --offload sequential is more aggressive
(layer-by-layer) for tighter VRAM; --offload none requires 40 GB+.

Usage:
    python -m scripts.inpaint_qwen_local \\
        --condition reflection \\
        --limit 10 --tag v1 \\
        --combine-with-flux inpaint-tests/reflection/flux-kontext-pro__v1

Env:
    WORKER_TOKEN              dashboard auth (required for candidate fetch)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs
    RCLONE_R2                 default 'r2'
    HF_TOKEN                  optional (anonymous OK for public Qwen weights)
"""
from __future__ import annotations

import argparse
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

import torch  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")


def log(msg: str) -> None:
    print(msg, flush=True)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


EDIT_PROMPTS: dict[str, str] = {
    "reflection": (
        "Remove any reflections of the photographer, their camera, or their "
        "tripod that appear in mirrors, windows, glass doors, or other "
        "reflective surfaces in this image. Replace each removed reflection "
        "with what the surface would naturally reflect (the actual room "
        "behind the photographer's position). Leave every other part of "
        "the image completely unchanged including all lighting, colors, "
        "furniture, decor, and architectural details."
    ),
    "dead_fixture": (
        "Identify any light fixtures that are currently off or not glowing "
        "and make them appear lit naturally — warm bulb glow that matches "
        "the existing warm/cool color temperature of the other lit fixtures "
        "in the same room. Leave every other part of the image unchanged."
    ),
    "finger-in-photo": (
        "Remove any human finger, thumb, or hand intruding into the corner "
        "or edge of the frame. Reconstruct what the lens would have captured "
        "behind the obstruction. Leave the rest of the image unchanged."
    ),
}


def get_candidates(condition: str, limit: int) -> list[dict]:
    sp = urllib.parse.urlencode({"condition": condition, "limit": str(limit)})
    resp = _request("GET", f"/api/internal/diff-pair-candidates?{sp}")
    return (resp.get("items") or [])[:limit]


def make_n_up(images: list[Image.Image], labels: list[str],
              target_h: int = 640) -> Image.Image:
    def fit(im: Image.Image) -> Image.Image:
        w, h = im.size
        return im.convert("RGB").resize(
            (round(w * (target_h / h)), target_h), Image.BILINEAR)
    fitted = [fit(im) for im in images]
    pad = 8
    label_h = 28
    total_w = sum(im.width for im in fitted) + pad * (len(fitted) + 1)
    total_h = target_h + label_h + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    x = pad
    for im, label in zip(fitted, labels):
        canvas.paste(im, (x, label_h + pad))
        draw.text((x + 6, pad), label, fill=(220, 220, 220), font=font)
        x += im.width + pad
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model", default="Qwen/Qwen-Image-Edit-2509",
                    help="HF repo for Qwen-Image-Edit. Defaults to the "
                         "latest 2509 release; falls back automatically.")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--offload", choices=("none", "model", "sequential"),
                    default="model",
                    help="VRAM strategy: none requires ~40 GB; model swaps "
                         "components in/out (24 GB OK, default); sequential "
                         "is layer-by-layer (tighter VRAM, slower).")
    ap.add_argument("--steps", type=int, default=30,
                    help="diffusion steps; 30 is balanced. 20 fast, 50 finest.")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--combine-with-flux", default=None,
                    help="R2 prefix containing flux outputs to include in "
                         "the comparison (e.g., inpaint-tests/reflection/"
                         "flux-kontext-pro__v1). If unset, builds a 3-up "
                         "source | qwen | vendor composite instead.")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required"); return 2
    prompt = args.prompt or EDIT_PROMPTS.get(args.condition)
    if not prompt:
        log(f"ERROR: no prompt for condition '{args.condition}'"); return 2

    # Lazy-import diffusers to keep the early sanity check cheap.
    from diffusers import QwenImageEditPipeline  # type: ignore

    log(f"qwen inpaint test: condition={args.condition} model={args.model} "
        f"offload={args.offload} steps={args.steps}")
    log(f"  loading pipeline (this can take a minute on first run for "
        f"weight download)…")
    t_load = time.time()
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    log(f"  loaded in {time.time() - t_load:.1f}s")

    items = get_candidates(args.condition, args.limit)
    log(f"  candidates: {len(items)}")
    if not items:
        log("  nothing to do"); return 0

    model_short = re.sub(r"[^a-z0-9]+", "-",
                         args.model.split("/")[-1].lower()).strip("-")
    out_prefix = f"inpaint-tests/{args.condition}/{model_short}__{args.tag}"

    n_done = n_err = 0
    t_start = time.time()
    public_urls: list[str] = []
    with tempfile.TemporaryDirectory(prefix="arem-qwen-") as scratch:
        sp = Path(scratch)
        for idx, it in enumerate(items, start=1):
            t0 = time.time()
            src_key = it["imageR2Path"]
            vendor_key = it["vendorImageR2Path"]
            stem = src_key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            src_local = sp / f"src-{idx:02d}.jpg"
            vendor_local = sp / f"vendor-{idx:02d}.jpg"
            edited_local = sp / f"edited-{idx:02d}.jpg"
            comp_local = sp / f"comp-{idx:02d}.jpg"
            flux_local = sp / f"flux-{idx:02d}.jpg" if args.combine_with_flux else None

            if not rclone_fetch(src_key, src_local, bucket=R2_OUTPUT_BUCKET):
                n_err += 1; continue
            if not rclone_fetch(vendor_key, vendor_local, bucket=R2_BUCKET):
                n_err += 1; continue
            if flux_local is not None:
                if not rclone_fetch(
                    f"{args.combine_with_flux}/{stem}-edited.jpg",
                    flux_local, bucket=R2_BUCKET,
                ):
                    log(f"  WARN no FLUX result for {stem}, skipping flux pane")
                    flux_local = None

            try:
                t_gen = time.time()
                with Image.open(src_local) as src_im:
                    src_im = src_im.convert("RGB")
                    out = pipe(
                        image=src_im,
                        prompt=prompt,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator("cuda").manual_seed(args.seed),
                    )
                edited_im = out.images[0]
                edited_im.save(edited_local, format="JPEG", quality=92)
                gen_ms = int((time.time() - t_gen) * 1000)

                # Build composite
                with Image.open(src_local) as a, \
                     Image.open(edited_local) as q, \
                     Image.open(vendor_local) as v:
                    panes = [a]
                    labels = ["SOURCE (pre-edit)"]
                    if flux_local and flux_local.exists():
                        with Image.open(flux_local) as f:
                            panes.append(f.copy())
                        labels.append("FLUX KONTEXT")
                    panes.append(q)
                    labels.append("QWEN IMAGE EDIT")
                    panes.append(v)
                    labels.append("VENDOR (AutoHDR)")
                    composite = make_n_up(panes, labels)
                    composite.save(comp_local, format="JPEG", quality=85)

                comp_key = f"{out_prefix}/{stem}-{len(panes)}up.jpg"
                edit_key = f"{out_prefix}/{stem}-edited.jpg"
                rclone_upload(comp_local, comp_key)
                rclone_upload(edited_local, edit_key)
                comp_url = (
                    f"{DASHBOARD_URL}/api/image?"
                    f"path={urllib.parse.quote(comp_key)}"
                )
                public_urls.append(comp_url)

                runtime_ms = int((time.time() - t0) * 1000)
                log(f"  [{idx}/{len(items)}] {stem}: gen={gen_ms}ms "
                    f"total={runtime_ms}ms")
                log(f"      composite: {comp_url}")
                n_done += 1
            except Exception as e:
                log(f"  [{idx}/{len(items)}] {src_key}: ERROR {str(e)[:200]}")
                n_err += 1
            finally:
                for f in (src_local, vendor_local, edited_local, comp_local):
                    try: f.unlink(missing_ok=True)
                    except Exception: pass
                if flux_local is not None:
                    try: flux_local.unlink(missing_ok=True)
                    except Exception: pass

    dt = time.time() - t_start
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(items),1):.1f}s/img avg) ---")
    log(f"  processed: {n_done}  errors: {n_err}")
    log("\nView composites:")
    for u in public_urls:
        log(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
