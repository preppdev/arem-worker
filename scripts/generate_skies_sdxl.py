"""Generate sky-library candidates using Stable Diffusion XL.

Produces N images per category from a curated prompt template set,
uploads each to R2 at `sky-library/<uuid>.jpg` + a 512px thumb at
`sky-library/thumbs/<uuid>.jpg`, then POSTs to /api/sky-library to
register the candidate. Reviewers approve/reject via /sky-library on
the dashboard.

The compositor (cathedral model) will eventually select only from
`status='approved'` rows.

Categories + prompt seeds are curated for real-estate-photography sky
replacement: clean horizons, no foreground objects, photo-realistic
output that composites believably behind a house.

Run from the arem-worker repo root on a GPU box (~10 GB VRAM at 1024×1024):

    set -a && . /etc/arem-worker.env && set +a
    python -m scripts.generate_skies_sdxl --per-category 20
    python -m scripts.generate_skies_sdxl --categories clear_blue,partly_cloudy --per-category 50
    python -m scripts.generate_skies_sdxl --per-category 5 --dry-run

The first run downloads the SDXL base checkpoint (~10 GB) into the
HuggingFace cache. Subsequent runs reuse it.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch  # type: ignore
from PIL import Image  # type: ignore


DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")

# Prompt templates per category. Each (positive, negative) pair gets
# rotated across multiple seeds to produce variation. Negative prompts
# scrubthe most common SDXL failure modes for sky generation:
#   - inadvertent foreground (trees, fences, buildings)
#   - artistic / painted style (we want photoreal)
#   - watermarks / signatures
#   - aerial-by-mistake (we want sky FROM the ground, not aerial views of clouds)
# Each list has 3-4 prompt variations to introduce diversity even at the
# same seed; the compositor benefits from a heterogeneous approved set.

NEG_BASE = (
    "watermark, signature, text, logo, frame, border, "
    "people, buildings, trees, mountains, ground, horizon line, "
    "painting, illustration, drawing, anime, cartoon, "
    "low quality, blurry, jpeg artifacts, oversaturated, "
    "aerial view, top-down view, cityscape, ocean, water"
)

PROMPT_TEMPLATES: dict[str, list[str]] = {
    "clear_blue": [
        "clear blue sky, no clouds, vibrant azure, looking straight up, "
        "photo-realistic, real-estate photography reference, sharp, 8k",
        "vivid bright cyan blue sky, gradient from light blue at horizon to "
        "deeper blue overhead, photoreal, no clouds, no objects, 8k",
        "pure blue sky on a sunny midday in summer, completely clear, "
        "no clouds, photoreal background plate, no foreground, 8k",
        "rich saturated blue sky, gradient, light and airy, completely "
        "cloudless, photoreal, suitable for sky replacement, 8k",
    ],
    "partly_cloudy": [
        "blue sky with scattered white cumulus clouds, puffy bright clouds, "
        "soft sunlight, photoreal, real-estate background plate, 8k",
        "blue sky with white puffy clouds, fair-weather cumulus, late "
        "morning light, photoreal, no buildings or trees, 8k",
        "summery blue sky with fluffy white cumulus, soft shadows on clouds, "
        "vivid, real-estate sky replacement plate, 8k photoreal",
        "vibrant blue sky with scattered cotton-like clouds, gentle "
        "highlights, midday photo plate, no foreground, 8k",
    ],
    "dramatic": [
        "dramatic blue sky with billowing cumulus clouds, late afternoon "
        "light, sun rays peeking, photoreal, sharp, 8k",
        "blue sky with tall cumulonimbus clouds in the distance, dramatic "
        "lighting but no rain, vivid contrast, photoreal, 8k",
        "deep blue sky with majestic white clouds piling up, sunbeams "
        "between, photoreal real-estate background, 8k",
        "moody but bright blue sky with billowing cumulus and visible light "
        "rays, photoreal, no foreground, 8k",
    ],
    "dawn_dusk": [
        "sky at sunset, soft orange and pink hues, gentle clouds, "
        "photoreal, real-estate sky replacement plate, 8k",
        "early evening sky, magenta and amber tones, wispy clouds, "
        "photoreal, no buildings, 8k",
        "sky at dusk, soft pastel pinks and oranges, gentle cloud "
        "textures, photoreal background plate, 8k",
        "sunset sky, low warm sun, broken altostratus clouds glowing peach, "
        "photoreal, real-estate plate, 8k",
    ],
    "golden_hour": [
        "warm golden hour sky, soft yellow and pale orange light, gentle "
        "clouds, photoreal, real-estate plate, 8k",
        "late afternoon golden sky with soft cumulus catching warm light, "
        "honey tones, photoreal, no foreground, 8k",
        "mellow golden-hour sky, warm low-contrast lighting, soft white-gold "
        "clouds, photoreal real-estate plate, 8k",
        "golden hour blue-gold sky, scattered clouds rimmed in warm light, "
        "photoreal, no buildings or trees, 8k",
    ],
    "overcast": [
        "soft overcast sky, diffuse white-grey clouds, gentle even lighting, "
        "photoreal real-estate plate, no harsh sun, 8k",
        "uniform overcast sky, light grey, slight blue tint, photoreal "
        "real-estate background plate, no foreground, 8k",
        "lightly overcast sky with soft cloud variations, white to pale grey, "
        "photoreal, gentle real-estate sky plate, 8k",
        "broken overcast, mostly cloud cover with hint of brightness, soft "
        "diffused light, photoreal real-estate plate, 8k",
    ],
}


# ---- HTTP / rclone helpers ----


def log(msg: str) -> None:
    print(msg, flush=True)


def _post(payload: dict) -> dict:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        f"{DASHBOARD_URL}/api/sky-library",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-worker-token": WORKER_TOKEN,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST /api/sky-library -> HTTP {e.code}: {body[:300]}") from e


def rclone(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["rclone", *args], capture_output=True, text=True, timeout=timeout,
    )


def rclone_upload(local: Path, r2_key: str) -> bool:
    """Upload one file to R2 under our bucket. Returns True on success."""
    dst = f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}"
    r = rclone(["copyto", str(local), dst, "--retries", "2"], timeout=120)
    if r.returncode != 0:
        log(f"  WARN rclone copyto {r2_key} rc={r.returncode}: "
            f"{r.stderr[-200:].strip()}")
        return False
    return True


# ---- Image helpers ----


def save_jpg(img: Image.Image, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=quality, optimize=True)


def make_thumb(src: Path, dst: Path, max_long_edge: int = 512, quality: int = 88):
    im = Image.open(src).convert("RGB")
    im.thumbnail((max_long_edge, max_long_edge))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=quality, optimize=True)


# ---- SDXL pipeline ----


def load_pipeline(dtype: str = "fp16"):
    """Load SDXL base. ~10 GB VRAM at 1024×1024 with fp16."""
    from diffusers import StableDiffusionXLPipeline  # type: ignore
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    log("loading SDXL base (stabilityai/stable-diffusion-xl-base-1.0)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if dtype == "fp16" else None,
    )
    pipe = pipe.to("cuda")
    # Memory-efficient attention. Saves a few GB; small speed cost.
    pipe.enable_xformers_memory_efficient_attention()  # type: ignore[attr-defined]
    log("  ready")
    return pipe


def generate_one(
    pipe, prompt: str, negative_prompt: str, seed: int,
    width: int, height: int, steps: int, guidance: float,
):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width, height=height,
        generator=g,
    )
    return out.images[0]


# ---- Per-candidate flow ----


def produce_candidate(
    pipe, *, category: str, prompt: str, negative: str,
    seed: int, width: int, height: int, steps: int, guidance: float,
    scratch: Path, dry_run: bool,
) -> bool:
    """Generate one image; upload + register if not dry_run.
    Returns True on success."""
    sky_id = uuid.uuid4().hex[:16]
    r2_key = f"sky-library/{sky_id}.jpg"
    thumb_key = f"sky-library/thumbs/{sky_id}.jpg"

    t0 = time.time()
    img = generate_one(
        pipe, prompt, negative, seed,
        width, height, steps, guidance,
    )
    dt = time.time() - t0
    log(f"  [{category}] seed={seed} {width}x{height}  gen={dt:.1f}s")

    if dry_run:
        # In dry-run, save samples locally so you can eyeball quality
        sample_dir = scratch / "samples"
        save_jpg(img, sample_dir / f"{category}_{sky_id}.jpg")
        log(f"    [dry] saved {sample_dir / f'{category}_{sky_id}.jpg'}")
        return True

    local = scratch / f"{sky_id}.jpg"
    save_jpg(img, local)
    thumb_local = scratch / f"{sky_id}_thumb.jpg"
    make_thumb(local, thumb_local)

    if not rclone_upload(local, r2_key):
        return False
    if not rclone_upload(thumb_local, thumb_key):
        return False

    _post({
        "r2Path": r2_key,
        "thumbR2Path": thumb_key,
        "source": "sdxl",
        "prompt": prompt,
        "category": category,
        "width": width,
        "height": height,
    })
    # Cleanup
    try:
        local.unlink()
        thumb_local.unlink()
    except FileNotFoundError:
        pass
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-category", type=int, default=20,
                    help="how many images to produce per category")
    ap.add_argument("--categories", type=str, default=None,
                    help="comma-separated subset of categories (default: all)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed-start", type=int, default=None,
                    help="base seed; defaults to time-based")
    ap.add_argument("--dry-run", action="store_true",
                    help="save samples locally, skip upload + DB")
    ap.add_argument("--scratch", type=str, default="/tmp/arem-sky-gen")
    args = ap.parse_args()

    if not args.dry_run and not WORKER_TOKEN:
        print("ERROR: WORKER_TOKEN env var is required (or pass --dry-run)",
              file=sys.stderr)
        return 2

    categories = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else list(PROMPT_TEMPLATES.keys())
    )
    bad = [c for c in categories if c not in PROMPT_TEMPLATES]
    if bad:
        print(f"ERROR: unknown categories: {bad}", file=sys.stderr)
        return 2

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    log(f"sky-gen: dashboard={DASHBOARD_URL}  categories={categories}  "
        f"per_category={args.per_category}  size={args.width}x{args.height}")
    log(f"  output: r2:{R2_BUCKET}/sky-library/  + dashboard /api/sky-library")

    pipe = load_pipeline(dtype="fp16")
    seed_base = args.seed_start if args.seed_start is not None else int(time.time())

    n_ok = 0
    n_fail = 0
    t_overall = time.time()
    for cat in categories:
        prompts = PROMPT_TEMPLATES[cat]
        for i in range(args.per_category):
            # Rotate through the prompt variations + bump seed per-candidate
            prompt = prompts[i % len(prompts)]
            seed = seed_base + hash((cat, i)) % (2 ** 31)
            try:
                ok = produce_candidate(
                    pipe, category=cat, prompt=prompt, negative=NEG_BASE,
                    seed=seed, width=args.width, height=args.height,
                    steps=args.steps, guidance=args.guidance,
                    scratch=scratch, dry_run=args.dry_run,
                )
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                log(f"  FAIL {cat} #{i}: {e}")
                n_fail += 1

    log(f"\n--- done in {time.time() - t_overall:.1f}s — "
        f"produced {n_ok} candidates ({n_fail} failed)")
    if not args.dry_run:
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
