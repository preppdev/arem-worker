"""Self-hosted FLUX.1 Kontext [dev] inpaint test (research/evaluation only).

Uses diffusers' FluxKontextPipeline with the FLUX.1-Kontext-dev open
weights. The dev weights are **non-commercial** — this script is for
quality validation only, not production use.

Same I/O pattern as inpaint_qwen_local.py: pulls reflection-tagged
images with vendor pairs, runs the editor with our condition prompt,
builds an N-up composite that optionally includes prior runs
(--combine-with-flux-api, --combine-with-qwen) for direct A/B,
uploads to R2.

Memory: FLUX Kontext dev is ~12B (~24 GB at bf16). The 3090 has 24 GB
total — at bf16 with model offload you're tight; with --offload
sequential it fits comfortably. fp8 quantization (--quantize fp8)
fits cleanly with headroom.

Usage:
    python -m scripts.inpaint_flux_local \\
        --condition reflection \\
        --limit 10 --tag v1-local \\
        --combine-with-flux-api inpaint-tests/reflection/flux-kontext-pro__v1 \\
        --combine-with-qwen inpaint-tests/reflection/qwen-image-edit-2509__v1

Env:
    WORKER_TOKEN              dashboard auth (required)
    HF_TOKEN                  HuggingFace token with FLUX.1-Kontext-dev access
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs
    RCLONE_R2                 default 'r2'
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-Kontext-dev")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tag", default="v1-local")
    ap.add_argument("--offload", choices=("none", "model", "sequential"),
                    default="sequential",
                    help="VRAM strategy. 3090/24GB needs sequential at bf16; "
                         "model also works if no other big things on the GPU. "
                         "none requires ~24GB free with bf16.")
    ap.add_argument("--steps", type=int, default=28,
                    help="FLUX Kontext sweet spot is 20-30 steps.")
    ap.add_argument("--guidance", type=float, default=2.5,
                    help="FLUX Kontext default guidance ~2.5-4.5.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--combine-with-flux-api", default=None,
                    help="R2 prefix of prior FLUX-API run to include in "
                         "composite (e.g., inpaint-tests/reflection/"
                         "flux-kontext-pro__v1).")
    ap.add_argument("--combine-with-qwen", default=None,
                    help="R2 prefix of prior Qwen run to include.")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required"); return 2
    if not os.environ.get("HF_TOKEN"):
        log("WARN: HF_TOKEN not set in env; download may fail for gated model")

    prompt = args.prompt or EDIT_PROMPTS.get(args.condition)
    if not prompt:
        log(f"ERROR: no prompt for condition '{args.condition}'"); return 2

    from diffusers import FluxKontextPipeline  # type: ignore

    log(f"flux-local inpaint test: condition={args.condition} "
        f"model={args.model} offload={args.offload} steps={args.steps}")
    log(f"  loading pipeline (first run will download ~24 GB)…")
    t_load = time.time()
    pipe = FluxKontextPipeline.from_pretrained(
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
    with tempfile.TemporaryDirectory(prefix="arem-fluxlocal-") as scratch:
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
            flux_api_local = sp / f"fluxapi-{idx:02d}.jpg" if args.combine_with_flux_api else None
            qwen_local = sp / f"qwen-{idx:02d}.jpg" if args.combine_with_qwen else None

            if not rclone_fetch(src_key, src_local, bucket=R2_OUTPUT_BUCKET):
                n_err += 1; continue
            if not rclone_fetch(vendor_key, vendor_local, bucket=R2_BUCKET):
                n_err += 1; continue
            if flux_api_local is not None:
                if not rclone_fetch(
                    f"{args.combine_with_flux_api}/{stem}-edited.jpg",
                    flux_api_local, bucket=R2_BUCKET,
                ):
                    log(f"  WARN no flux-api result for {stem}")
                    flux_api_local = None
            if qwen_local is not None:
                if not rclone_fetch(
                    f"{args.combine_with_qwen}/{stem}-edited.jpg",
                    qwen_local, bucket=R2_BUCKET,
                ):
                    log(f"  WARN no qwen result for {stem}")
                    qwen_local = None

            try:
                t_gen = time.time()
                torch.cuda.empty_cache()
                with Image.open(src_local) as src_im:
                    src_im = src_im.convert("RGB")
                    gen_device = "cpu" if args.offload == "sequential" else "cuda"
                    out = pipe(
                        image=src_im,
                        prompt=prompt,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator(gen_device).manual_seed(args.seed),
                    )
                edited_im = out.images[0]
                edited_im.save(edited_local, format="JPEG", quality=92)
                torch.cuda.empty_cache()
                gen_ms = int((time.time() - t_gen) * 1000)

                # Build composite
                panes: list[Image.Image] = []
                labels: list[str] = []
                with Image.open(src_local) as a:
                    panes.append(a.copy()); labels.append("SOURCE (pre-edit)")
                if flux_api_local and flux_api_local.exists():
                    with Image.open(flux_api_local) as f:
                        panes.append(f.copy()); labels.append("FLUX KONTEXT (API)")
                with Image.open(edited_local) as fl:
                    panes.append(fl.copy()); labels.append("FLUX KONTEXT (local-dev)")
                if qwen_local and qwen_local.exists():
                    with Image.open(qwen_local) as q:
                        panes.append(q.copy()); labels.append("QWEN IMAGE EDIT")
                with Image.open(vendor_local) as v:
                    panes.append(v.copy()); labels.append("VENDOR (AutoHDR)")
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
                if flux_api_local is not None:
                    try: flux_api_local.unlink(missing_ok=True)
                    except Exception: pass
                if qwen_local is not None:
                    try: qwen_local.unlink(missing_ok=True)
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
