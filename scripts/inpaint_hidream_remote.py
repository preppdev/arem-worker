"""HiDream-I1-Edit inpaint test, designed to run on a one-shot rented
GPU pod (RunPod / Vast / Lambda).

Mirror of scripts/inpaint_qwen_local.py — same I/O, same dashboard
integration — but loads HiDream-I1-Edit (MIT, 17B) via diffusers'
HiDreamImageEditPipeline. Drops back to the HiDream-ai official
inference scripts if the diffusers pipeline class isn't available
in the installed version.

This script is the pod's whole job: load model, drain candidate list,
post results to the dashboard, exit. Pod tears down when the process
exits and stops billing.

Usage on pod:
    python -m scripts.inpaint_hidream_remote \\
        --condition reflection --limit 10 --tag phase-a

Env (required):
    WORKER_TOKEN
    DASHBOARD_URL              default https://arem-editing-dashboard.vercel.app
    R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY + R2_ACCOUNT_ID  (for rclone)
    R2_BUCKET                  default arem-training-data (mask + result upload)
    R2_OUTPUT_BUCKET           default arem-production-edit-jobs (source images)
    HF_TOKEN                   optional (HiDream weights are open)
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


# Mirror of the prompt taxonomy from inpaint_text_edit.py so the pod
# can be self-contained without cloning the whole worker tree. Keep in
# sync — when prompts change there, update here too.
EDIT_PROMPTS: dict[str, str] = {
    "reflection": (
        "Remove any reflections of the photographer, their camera, or "
        "their tripod that appear in mirrors, windows, glass doors, or "
        "other reflective surfaces. Replace each removed reflection with "
        "what the surface would naturally reflect (the actual room behind "
        "the photographer's position).\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "- DO NOT modify, add, remove, or relocate any furniture, decor "
        "items, plants, light fixtures, art, rugs, or appliances.\n"
        "- DO NOT change wall colors, paint, flooring, ceiling, or trim.\n"
        "- DO NOT add or remove any objects from countertops, shelves, "
        "tables, or floors.\n"
        "- DO NOT change lighting brightness or color temperature.\n"
        "- DO NOT alter the framing, perspective, or angle.\n"
        "- ONLY the reflection regions inside reflective surfaces may "
        "be modified.\n\n"
        "Every pixel outside the actual photographer/camera/tripod "
        "reflection must remain pixel-identical to the source."
    ),
    "dead-fixture": (
        "Find any light fixtures (lamps, sconces, chandeliers, ceiling "
        "lights, pendants) where the bulb or lampshade is visibly dark and "
        "not glowing — particularly lampshades that should be emitting "
        "light but are not. Make those fixtures appear lit naturally: warm "
        "bulb glow showing through the shade, matching the color "
        "temperature of any other lit fixtures in the same room. If none "
        "are lit for reference, default to a warm 2700-3000K bulb glow. "
        "Leave the fixture's physical shape, color, and position completely "
        "unchanged — only change whether it appears lit. Leave every other "
        "part of the image unchanged."
    ),
    "finger-in-photo": (
        "Find and completely remove any human body part visible anywhere "
        "in the image — fingers, hands, thumbs, arms, or any skin-toned "
        "body part. These are most often the photographer's fingers or "
        "hand intruding from a corner or edge of the frame, but they can "
        "appear anywhere in the image. Generatively reconstruct the "
        "natural scene that would be visible behind the removed body part "
        "(sky, wall, ceiling, foliage, building siding, etc.), matching "
        "the surrounding color, lighting, perspective, and texture. Leave "
        "every other part of the image unchanged."
    ),
    "sun-flare": (
        "This real-estate photo was shot with a sun-blasted lens, leaving "
        "artifact overlays on the scene — bright halos, hexagonal shapes, "
        "rainbow ghost reflections, or starbursts on top of windows and "
        "sky. REPAIR these artifacts by reconstructing the actual "
        "underlying surface (clean window glass, natural sky, plain wall) "
        "beneath each one. The repaired image should look like the same "
        "scene shot with a perfectly clean lens.\n\n"
        "CRITICAL — DO NOT ADD anything:\n"
        "- Do not add or strengthen any lens flare, sunburst, halo, ghost, "
        "or starburst anywhere in the image.\n"
        "- Do not increase the brightness of the sun, sky, or windows.\n"
        "- Do not introduce new highlights or glints anywhere.\n"
        "- The output must have fewer visual artifacts than the input, "
        "never more.\n\n"
        "Leave every other part of the image — building, interior, "
        "furniture, landscaping — completely unchanged."
    ),
    "lens-hood-in-photo": (
        "Remove the dark crescent or partial ring of the lens hood "
        "intruding from the corners or edges of the frame. Reconstruct "
        "what the lens would have captured behind the hood — usually the "
        "natural continuation of the wall, ceiling, or scene visible in "
        "the rest of the image. Leave every other part of the image "
        "unchanged."
    ),
    "shine": (
        "Find large bright patches on hard floors, countertops, or other "
        "surfaces where direct natural light (typically sunlight through a "
        "window) has washed out the natural color, leaving the surface "
        "looking over-bright and desaturated. Restore the natural color "
        "and saturation in those patches by sampling and extending the "
        "natural color, grain, and texture from the surrounding unaffected "
        "areas of the same surface. The bright patch can remain partially "
        "visible (the natural light is real) but the natural surface color "
        "should show through it. Do NOT touch legitimate specular "
        "highlights on faucets, polished hardware, granite, or small metal "
        "objects — those are desirable. Leave every other part of the "
        "image (walls, furniture, decor, ceiling, fixtures) completely "
        "unchanged."
    ),
    "color-cast": (
        "Find any LOCAL or PARTIAL areas of the image where the color "
        "balance is noticeably different from the rest of the photo — "
        "typically an adjacent room or area visible through a doorway "
        "that has a cooler blue/gray tint or a warmer orange tint "
        "because it has different lighting (for example, an unlit "
        "bedroom seen from a warm-lit kitchen, or a sunlit room seen "
        "from an interior hallway). Also covers partial color tints on "
        "reflective surfaces showing color-shifted reflections. Warm "
        "or cool those local areas to better match the color temperature "
        "and tone of the main lit areas of the photo so the transition "
        "between regions looks natural and balanced. Do NOT apply a "
        "global color shift to the whole image. Leave the dominant area, "
        "furniture, decor, accent walls, and outdoor scenery completely "
        "unchanged."
    ),
    "photographer-shadow": (
        "Remove any human-shaped shadow cast by the photographer, which "
        "will always appear in the FOREGROUND near the BOTTOM of the "
        "frame (on the floor, driveway, lawn, or similar foreground "
        "surface). Reconstruct the natural surface that the shadow is "
        "covering by extending the color, grain, and lighting from the "
        "surrounding unaffected area of the same surface. Leave every "
        "other part of the image — including other natural shadows from "
        "furniture, fixtures, trees, or the building itself — completely "
        "unchanged."
    ),
    "sky": (
        "If the sky visible through windows or outdoors is blown out, "
        "white, or featureless, replace it with a natural, well-exposed "
        "blue daytime sky with light scattered clouds appropriate to the "
        "scene. Match the lighting direction and time-of-day implied by "
        "the rest of the image. Preserve everything inside the room and "
        "everything that isn't sky — window frames, trees, buildings, "
        "and any visible landscape stay completely unchanged."
    ),
    "twilight": (
        "Transform this daytime exterior real-estate photo into a "
        "stylized twilight / dusk scene. Replace the daytime sky with a "
        "deep blue-to-violet twilight sky featuring a warm orange and "
        "peach horizon glow where the sun has just set. Turn on every "
        "visible interior light through windows so they emit a warm "
        "golden glow. Add subtle warm exterior accent lighting — porch "
        "lights, eaves, landscape uplights, lamp posts — wherever such "
        "fixtures are visible. Lower the overall scene exposure to match "
        "dusk lighting while keeping the building facade clearly visible "
        "and well-exposed. Preserve the building's exact architecture, "
        "color, materials, roofline, windows, and proportions. Keep "
        "landscaping, lawn, plants, trees, driveways, and any vehicles "
        "in place but slightly darker. Do not add new structures, "
        "people, animals, or vehicles. The result should look like a "
        "professional twilight real-estate photo."
    ),
}


def get_candidates(condition: str, limit: int,
                   endpoint: str = "/api/internal/diff-pair-candidates"
                   ) -> list[dict]:
    sp = urllib.parse.urlencode({
        "condition": condition, "limit": str(limit),
    })
    resp = _request("GET", f"{endpoint}?{sp}")
    items = resp.get("items") or []
    log(f"  candidates [{endpoint}]: {len(items)}")
    return items[:limit]


def get_done(condition: str, model_run: str) -> set[str]:
    sp = urllib.parse.urlencode({"condition": condition})
    try:
        resp = _request("GET", f"/api/inpaint-tests?{sp}")
        for run in resp.get("runs") or []:
            if run["modelRun"] == model_run:
                return {it["imageR2Path"] for it in run["items"]}
    except Exception as e:
        log(f"  WARN get_done: {e}")
    return set()


# ---- Pipeline loading (defensive) ----

def load_pipeline(model_id: str, device: str, dtype: str,
                  offload: str = "none"):
    """Load HiDream-I1-Edit via diffusers. Returns callable
    (PIL.Image, prompt, **kwargs) -> PIL.Image.

    offload: none | model | sequential — same conventions as
    inpaint_qwen_local.py. Use "sequential" on <40 GB GPUs.
    """
    import torch  # noqa: F401
    td = {"bf16": torch.bfloat16, "fp16": torch.float16,
          "fp32": torch.float32}[dtype]

    # Try several plausible pipeline class names. HiDream's diffusers
    # integration name has varied across releases.
    pipe = None
    last_err: Exception | None = None
    for cls_name in ("HiDreamImageEditPipeline",
                     "HiDreamImageEditingPipeline",
                     "HiDreamI1EditPipeline"):
        try:
            mod = __import__("diffusers", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            log(f"  loading {cls_name} from {model_id} ...")
            t0 = time.time()
            pipe = cls.from_pretrained(model_id, torch_dtype=td)
            log(f"  loaded in {time.time() - t0:.1f}s")
            break
        except (ImportError, AttributeError) as e:
            last_err = e
            continue
    if pipe is None:
        raise RuntimeError(
            f"No HiDream pipeline class found in diffusers; tried "
            f"HiDreamImageEditPipeline / HiDreamImageEditingPipeline / "
            f"HiDreamI1EditPipeline. Last error: {last_err}. "
            f"You may need to install HiDream-ai/HiDream-I1 from source.")

    if offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    def edit(image, prompt, steps=28, guidance=4.5, seed=12345):
        # When sequential CPU-offload is active, generator must live on CPU.
        gen_device = "cpu" if offload == "sequential" else device
        gen = torch.Generator(gen_device).manual_seed(seed)
        out = pipe(
            image=image, prompt=prompt,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=gen,
        )
        return out.images[0]
    return edit


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
    ap.add_argument("--model", default="HiDream-ai/HiDream-I1-Edit")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--candidates-endpoint",
                    default="/api/internal/diff-pair-candidates")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--offload", choices=("none", "model", "sequential"),
                    default="sequential",
                    help="VRAM strategy. Default sequential fits 24 GB; "
                         "model fits ~40 GB comfortably; none requires "
                         "~30 GB free at bf16.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required"); return 2
    prompt = args.prompt or EDIT_PROMPTS.get(args.condition)
    if not prompt:
        log(f"ERROR: no prompt for condition '{args.condition}'"); return 2

    log(f"hidream inpaint test: condition={args.condition} model={args.model} "
        f"dtype={args.dtype} offload={args.offload} device={args.device}")
    edit_fn = load_pipeline(args.model, args.device, args.dtype, args.offload)

    items = get_candidates(args.condition, args.limit,
                           endpoint=args.candidates_endpoint)
    if not items:
        log("  no candidates"); return 0

    model_short = re.sub(r"[^a-z0-9]+", "-",
                         args.model.split("/")[-1].lower()).strip("-")
    run_id = f"{model_short}__{args.tag}"
    out_prefix = f"inpaint-tests/{args.condition}/{run_id}"

    done = get_done(args.condition, run_id)
    if done:
        before = len(items)
        items = [it for it in items if it["imageR2Path"] not in done]
        log(f"  skipping {before - len(items)} already-done; "
            f"{len(items)} to process")

    if not items:
        log("  nothing to do after dedup"); return 0

    n_done = n_err = 0
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="arem-hidream-") as scratch:
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
            if vendor_key and not rclone_fetch(vendor_key, vendor_local,
                                               bucket=R2_BUCKET):
                vendor_key = None

            try:
                t_gen = time.time()
                torch.cuda.empty_cache()
                with Image.open(src_local) as src_im:
                    src_im = src_im.convert("RGB")
                    orig_w, orig_h = src_im.size
                    edited_im = edit_fn(src_im, prompt,
                                        steps=args.steps,
                                        guidance=args.guidance,
                                        seed=args.seed)
                edited_im.save(edited_local, format="JPEG", quality=92)
                torch.cuda.empty_cache()
                gen_ms = int((time.time() - t_gen) * 1000)

                panes: list[Image.Image] = []
                labels: list[str] = []
                with Image.open(src_local) as a:
                    panes.append(a.copy()); labels.append("SOURCE")
                with Image.open(edited_local) as h:
                    panes.append(h.copy()); labels.append("HIDREAM I1 EDIT")
                if vendor_key and vendor_local.exists():
                    with Image.open(vendor_local) as v:
                        panes.append(v.copy()); labels.append("VENDOR")
                composite = make_n_up(panes, labels)
                composite.save(comp_local, format="JPEG", quality=85)

                comp_key = f"{out_prefix}/{stem}-{len(panes)}up.jpg"
                edit_key = f"{out_prefix}/{stem}-edited.jpg"
                rclone_upload(comp_local, comp_key)
                rclone_upload(edited_local, edit_key)

                runtime_ms = int((time.time() - t0) * 1000)
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/inpaint-test", {
                            "condition": args.condition,
                            "modelRun": run_id,
                            "imageR2Path": src_key,
                            "vendorR2Path": vendor_key,
                            "editedR2Path": edit_key,
                            "compositeR2Path": comp_key,
                            "prompt": prompt,
                            "modelInfo": f"{args.model} (dtype={args.dtype}, "
                                         f"offload={args.offload}) self-host",
                            "multiConditions": [c for c in (it.get("allConditions") or [])
                                                if c != args.condition],
                            "runtimeMs": runtime_ms,
                        })
                    except Exception as e:
                        log(f"    WARN dashboard POST: {e}")
                log(f"  [{idx}/{len(items)}] {stem}: gen={gen_ms}ms "
                    f"total={runtime_ms}ms")
                n_done += 1
            except Exception as e:
                log(f"  [{idx}/{len(items)}] {src_key}: ERROR {str(e)[:200]}")
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/inpaint-test", {
                            "condition": args.condition,
                            "modelRun": run_id,
                            "imageR2Path": src_key,
                            "prompt": prompt,
                            "errorMessage": str(e)[:500],
                            "runtimeMs": int((time.time() - t0) * 1000),
                        })
                    except Exception as e2:
                        log(f"    WARN POST: {e2}")
                n_err += 1
            finally:
                for f in (src_local, vendor_local, edited_local, comp_local):
                    try: f.unlink(missing_ok=True)
                    except Exception: pass

    dt = time.time() - t_start
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(items),1):.1f}s/img avg) ---")
    log(f"  processed: {n_done}  errors: {n_err}")
    log(f"  model_run: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
