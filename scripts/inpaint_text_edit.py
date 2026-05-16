"""Instruction-tuned image editing test.

Sends source images to an image-edit API (default Replicate /
FLUX.1 Kontext) with a text instruction describing what to remove.
For each result, builds a 3-up comparison composite
(source | edited | vendor) and uploads it to R2 so we can eyeball
visually. Optional: scores LPIPS perceptual similarity against the
vendor (AutoHDR) output as a quantitative signal.

Usage:
    python -m scripts.inpaint_text_edit \\
        --condition reflection \\
        --provider replicate \\
        --model black-forest-labs/flux-kontext-pro \\
        --limit 10 --tag v1

Env:
    WORKER_TOKEN              dashboard auth (required for candidate fetch)
    REPLICATE_API_TOKEN       Replicate token (required for --provider replicate)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data (composite uploads)
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs (source images)
    RCLONE_R2                 default 'r2'
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

from PIL import Image, ImageDraw, ImageFont  # type: ignore

try:
    import replicate  # type: ignore
except ImportError:
    replicate = None

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "arem-production-edit-jobs")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")


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


def http_get(url: str, dest: Path, timeout: int = 120) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        log(f"  WARN http_get {url}: {e}")
        return False


# ---- Condition prompts ----

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
    """Pull condition-tagged ImageReview rows with vendor pair for A/B."""
    sp = urllib.parse.urlencode({
        "condition": condition, "limit": str(limit),
    })
    resp = _request("GET", f"/api/internal/diff-pair-candidates?{sp}")
    items = resp.get("items") or []
    log(f"  candidates: {len(items)}")
    return items[:limit]


# ---- Image-edit providers ----

def edit_replicate(model: str, image_path: Path, prompt: str) -> str:
    """Call Replicate model, return URL of edited image."""
    if replicate is None:
        raise RuntimeError("`replicate` package not installed")
    with open(image_path, "rb") as f:
        out = replicate.run(
            model,
            input={
                "prompt": prompt,
                "input_image": f,
                "output_format": "jpg",
                "safety_tolerance": 2,
            },
        )
    # Replicate SDK returns a FileOutput or str depending on version.
    if isinstance(out, list):
        out = out[0]
    if hasattr(out, "url"):
        return out.url
    return str(out)


# ---- Comparison composite ----

def make_three_up(src: Image.Image, edited: Image.Image, vendor: Image.Image,
                  labels: tuple[str, str, str] = (
                      "SOURCE (pre-edit)", "FLUX KONTEXT (ours)",
                      "VENDOR (AutoHDR)",
                  )) -> Image.Image:
    """Stack horizontally at uniform height, with labels."""
    target_h = 640
    def fit(im: Image.Image) -> Image.Image:
        w, h = im.size
        new_w = round(w * (target_h / h))
        return im.convert("RGB").resize((new_w, target_h), Image.BILINEAR)
    a, b, c = fit(src), fit(edited), fit(vendor)
    pad = 8
    label_h = 28
    total_w = a.width + b.width + c.width + pad * 4
    total_h = target_h + label_h + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    x = pad
    for im, label in [(a, labels[0]), (b, labels[1]), (c, labels[2])]:
        canvas.paste(im, (x, label_h + pad))
        draw.text((x + 6, pad), label, fill=(220, 220, 220), font=font)
        x += im.width + pad
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--provider", choices=("replicate",), default="replicate")
    ap.add_argument("--model", default="black-forest-labs/flux-kontext-pro")
    ap.add_argument("--prompt", default=None,
                    help="override per-condition default")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required"); return 2
    if args.provider == "replicate" and not REPLICATE_API_TOKEN:
        log("ERROR: REPLICATE_API_TOKEN env var is required"); return 2

    prompt = args.prompt or EDIT_PROMPTS.get(args.condition)
    if not prompt:
        log(f"ERROR: no prompt defined for condition '{args.condition}'"); return 2

    model_short = re.sub(r"[^a-z0-9]+", "-",
                         args.model.split("/")[-1].lower()).strip("-")
    run_id = f"{model_short}__{args.tag}"
    out_prefix = f"inpaint-tests/{args.condition}/{run_id}"
    log(f"inpaint test: condition={args.condition} model={args.model} "
        f"prompt[:80]={prompt[:80]!r} out_prefix={out_prefix}")

    items = get_candidates(args.condition, args.limit)
    if not items:
        log("  no candidates"); return 0

    n_done = n_err = 0
    t_start = time.time()
    public_urls: list[str] = []
    with tempfile.TemporaryDirectory(prefix="arem-inpaint-") as scratch:
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

            if not rclone_fetch(src_key, src_local, bucket=R2_OUTPUT_BUCKET):
                n_err += 1; continue
            if not rclone_fetch(vendor_key, vendor_local, bucket=R2_BUCKET):
                n_err += 1; continue
            try:
                if args.dry_run:
                    log(f"  [{idx}/{len(items)}] (dry-run) would edit {src_key}")
                    continue
                t_api = time.time()
                url = edit_replicate(args.model, src_local, prompt)
                api_ms = int((time.time() - t_api) * 1000)
                if not http_get(url, edited_local):
                    n_err += 1; continue
                with Image.open(src_local) as a, \
                     Image.open(edited_local) as b, \
                     Image.open(vendor_local) as c:
                    composite = make_three_up(a, b, c)
                    composite.save(comp_local, format="JPEG", quality=85)
                # Upload composite AND raw edited
                comp_key = f"{out_prefix}/{stem}-3up.jpg"
                edit_key = f"{out_prefix}/{stem}-edited.jpg"
                rclone_upload(comp_local, comp_key)
                rclone_upload(edited_local, edit_key)
                comp_url = f"{DASHBOARD_URL}/api/image?path={urllib.parse.quote(comp_key)}"
                public_urls.append(comp_url)
                runtime_ms = int((time.time() - t0) * 1000)
                log(f"  [{idx}/{len(items)}] {stem}: api={api_ms}ms "
                    f"total={runtime_ms}ms")
                log(f"      composite: {comp_url}")
                n_done += 1
            except Exception as e:
                log(f"  [{idx}/{len(items)}] {src_key}: ERROR {str(e)[:200]}")
                n_err += 1
            finally:
                for f in (src_local, vendor_local, edited_local, comp_local):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass

    dt = time.time() - t_start
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(items),1):.1f}s/img avg) ---")
    log(f"  processed: {n_done}  errors: {n_err}")
    log(f"\nView composites (open in browser, requires dashboard login):")
    for u in public_urls:
        log(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
