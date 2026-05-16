"""Few-shot mask prediction from N hand-labeled reference (image, mask)
pairs, using DINOv2 patch-level nearest-neighbor classification.

Pipeline:
  1. Pull N hand-labeled reference pairs from /api/internal/ground-truth-masks.
  2. Hold out the first --heldout (sorted by createdAt, oldest first) for eval.
  3. From the remaining train pairs, build two feature banks:
       - positives: DINOv2 patch features whose corresponding mask cell is set
       - negatives: patch features outside the mask
  4. Pull every candidate image for the condition via auto-mask-candidates.
  5. For each candidate (train+eval+unlabeled), score each query patch:
       score(p) = max_pos_cosine_sim(p, P) - max_neg_cosine_sim(p, N)
       binary(p) = score > --score-threshold
  6. Upsample binary patch grid to image resolution, save as PNG, upload to R2,
     POST to /api/internal/auto-mask-test.

We deliberately predict on train+eval+unlabeled — the IoU scorer expects rows
to exist for the eval slice, and visualizing train predictions is useful as a
sanity check (model should reproduce training masks reasonably).

Why DINOv2 + patch k-NN, not SegGPT/PerSAM/SAM 2:
  - No fine-tune, no extra weights beyond DINOv2.
  - Already on box (transformers).
  - Designed exactly for cross-image few-shot semantic correspondence.
  - Stays in match-first mode — we're replicating "use what we have to predict
    what we don't" rather than building a per-condition fine-tune.

Usage:
    python -m scripts.few_shot_dinov2_mask \\
        --condition reflection \\
        --heldout 5 \\
        --score-threshold 0.05 \\
        --dinov2 facebook/dinov2-base

Env:
    WORKER_TOKEN              dashboard auth (required)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data  (masks: read + write)
    R2_OUTPUT_BUCKET          default arem-production-edit-jobs  (source images)
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
from typing import Any

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image  # type: ignore
from transformers import AutoImageProcessor, AutoModel  # type: ignore

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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {text[:300]}") from e


def rclone(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True,
                          timeout=timeout)


def fetch_image_from_r2(r2_key: str, dest: Path, *, bucket: str) -> bool:
    src = f"{RCLONE_R2}:{bucket}/{r2_key}"
    r = rclone(["copyto", src, str(dest), "--progress=false"], timeout=120)
    if r.returncode != 0:
        log(f"    WARN fetch {r2_key}: rc={r.returncode} {r.stderr[-160:].strip()}")
        return False
    return True


def upload_mask(local: Path, r2_key: str) -> bool:
    dst = f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}"
    r = rclone(["copyto", str(local), dst, "--progress=false"], timeout=120)
    if r.returncode != 0:
        log(f"    WARN upload {r2_key}: rc={r.returncode} {r.stderr[-160:].strip()}")
        return False
    return True


def get_candidates(*, condition: str, model_version: str,
                   limit: int) -> list[str]:
    sp = urllib.parse.urlencode({
        "condition": condition,
        "modelVersion": model_version,
        "limit": str(limit),
    })
    resp = _request("GET", f"/api/internal/auto-mask-candidates?{sp}")
    log(f"  candidates: {resp.get('todo')}/{resp.get('total')} todo "
        f"(already done: {resp.get('alreadyDone')})")
    return resp.get("items") or []


def get_truth_pairs(condition: str, decision: str = "edited") -> list[dict]:
    sp = urllib.parse.urlencode({"condition": condition, "decision": decision})
    return _request("GET",
                    f"/api/internal/ground-truth-masks?{sp}")["items"]


# ---- DINOv2 patch features ----

class PatchFeaturizer:
    """Wraps DINOv2 to produce L2-normalized patch features at a fixed
    grid resolution. We resize images to (grid * patch_size) so the
    last hidden state directly maps to (grid, grid) patches with no
    interpolation. patch_size is 14 for all DINOv2 variants."""

    def __init__(self, model_id: str, device: str, grid: int = 32):
        log(f"  loading {model_id} on {device} ...")
        t0 = time.time()
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.device = device
        self.grid = grid
        self.patch = 14  # DINOv2 patch size
        self.resize = grid * self.patch  # e.g. 32 * 14 = 448
        self.mean = torch.tensor(self.processor.image_mean).view(3, 1, 1).to(device)
        self.std = torch.tensor(self.processor.image_std).view(3, 1, 1).to(device)
        log(f"  loaded in {time.time() - t0:.1f}s  grid={grid} resize={self.resize}")

    @torch.no_grad()
    def features(self, image: Image.Image) -> torch.Tensor:
        """Return (grid, grid, dim) L2-normalized patch features."""
        im = image.convert("RGB").resize(
            (self.resize, self.resize), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        arr = arr.permute(2, 0, 1).to(self.device)  # CHW
        arr = (arr - self.mean) / self.std
        out = self.model(pixel_values=arr.unsqueeze(0))
        # last_hidden_state: (1, 1 + grid*grid, dim) — first token is CLS
        h = out.last_hidden_state[:, 1:, :]  # drop CLS
        h = F.normalize(h, dim=-1)
        return h.view(self.grid, self.grid, -1).contiguous()


def grid_mask(mask_image: Image.Image, grid: int,
              threshold: float = 0.5) -> np.ndarray:
    """Downsample a binary mask to grid×grid; return bool array where True
    means the patch contains majority positive pixels."""
    m = mask_image.convert("L").resize((grid, grid), Image.BILINEAR)
    return np.asarray(m, dtype=np.float32) / 255.0 > threshold


# ---- Feature bank scoring ----

def cosine_scores(query: torch.Tensor, bank: torch.Tensor,
                  *, topk: int = 5) -> torch.Tensor:
    """query: (P, D) L2-normalized query patches.
       bank:  (B, D) L2-normalized bank patches.
       Returns (P,) — mean of top-k cosine similarities to bank.
    """
    if bank.shape[0] == 0:
        return torch.zeros(query.shape[0], device=query.device)
    sims = query @ bank.T  # (P, B)
    k = min(topk, bank.shape[0])
    vals, _ = sims.topk(k, dim=-1)
    return vals.mean(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--dinov2", default="facebook/dinov2-base",
                    help="HF model id for DINOv2 (base or large)")
    ap.add_argument("--grid", type=int, default=32,
                    help="patch grid (image resized to grid*14)")
    ap.add_argument("--heldout", type=int, default=5,
                    help="N oldest hand-labels to hold out as eval (not used "
                         "for the feature bank)")
    ap.add_argument("--score-threshold", type=float, default=0.05,
                    help="positive_max_sim - negative_max_sim threshold")
    ap.add_argument("--topk-pos", type=int, default=3)
    ap.add_argument("--topk-neg", type=int, default=3)
    ap.add_argument("--limit", type=int, default=500,
                    help="candidate cap (we predict on every candidate, "
                         "including train+eval, so set this high enough)")
    ap.add_argument("--include-rejected-negatives", action="store_true",
                    default=True,
                    help="also use Annotation rows with decision='rejected' "
                         "(condition not present anywhere in the image) as "
                         "full-image negatives in the bank. Default on.")
    ap.add_argument("--no-rejected-negatives", dest="include_rejected_negatives",
                    action="store_false")
    ap.add_argument("--tag", default=None,
                    help="suffix appended to modelVersion (default = "
                         "'h{heldout}_t{threshold}'); makes each run its own "
                         "dashboard chip")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required")
        return 2

    tag = args.tag or f"h{args.heldout}_t{int(args.score_threshold*100):02d}"
    safe_model = re.sub(r"[^a-z0-9]+", "-",
                        args.dinov2.split("/")[-1].lower()).strip("-")
    model_version = f"{safe_model}-fewshot__{tag}"
    log(f"few-shot DINOv2 mask: condition={args.condition} "
        f"model_version={model_version} device={args.device}")

    truth = get_truth_pairs(args.condition)
    log(f"  hand-labeled pairs: {len(truth)}")
    if len(truth) <= args.heldout:
        log(f"ERROR: only {len(truth)} truth rows, can't hold out "
            f"{args.heldout}")
        return 2

    eval_rows = truth[: args.heldout]
    train_rows = truth[args.heldout:]
    log(f"  train: {len(train_rows)}  eval: {len(eval_rows)}")

    feat = PatchFeaturizer(args.dinov2, args.device, grid=args.grid)

    # ---- Build positive/negative banks from train pairs ----
    log("\n  building feature bank from train pairs ...")
    pos_chunks: list[torch.Tensor] = []
    neg_chunks: list[torch.Tensor] = []
    with tempfile.TemporaryDirectory(prefix="arem-fewshot-bank-") as scratch:
        sp = Path(scratch)
        for i, r in enumerate(train_rows, start=1):
            img_local = sp / f"img-{i:03d}.jpg"
            mask_local = sp / f"mask-{i:03d}.png"
            if not fetch_image_from_r2(r["imageR2Path"], img_local,
                                       bucket=R2_OUTPUT_BUCKET):
                continue
            if not fetch_image_from_r2(r["maskR2Path"], mask_local,
                                       bucket=R2_BUCKET):
                continue
            try:
                with Image.open(img_local) as im, Image.open(mask_local) as mk:
                    feats = feat.features(im)  # (G, G, D)
                    grid = grid_mask(mk, args.grid)  # (G, G) bool
                feats_flat = feats.view(-1, feats.shape[-1])  # (G*G, D)
                grid_flat = torch.from_numpy(grid.reshape(-1)).to(args.device)
                pos = feats_flat[grid_flat]
                neg = feats_flat[~grid_flat]
                pos_chunks.append(pos)
                neg_chunks.append(neg)
                log(f"    [{i}/{len(train_rows)}] {r['imageR2Path'].rsplit('/', 1)[-1]}: "
                    f"pos={pos.shape[0]} neg={neg.shape[0]}")
            except Exception as e:
                log(f"    WARN bank build {r['imageR2Path']}: {e}")

    if not pos_chunks:
        log("ERROR: no positive features collected — bank empty")
        return 2

    # Augment the negative bank with full-image features from images the
    # labeler said do NOT contain the condition. These are stronger negatives
    # than "patches outside the mask in a positive image" — a positive image
    # of a reflection still contains plenty of reflective surfaces that
    # aren't the photographer, and those patches end up in the within-image
    # negatives. Rejected images have none of the condition by definition.
    if args.include_rejected_negatives:
        log("\n  augmenting negative bank with rejected (no-condition) images ...")
        rejected = get_truth_pairs(args.condition, decision="rejected")
        log(f"    rejected rows: {len(rejected)}")
        with tempfile.TemporaryDirectory(prefix="arem-fewshot-neg-") as scratch:
            sp = Path(scratch)
            for i, r in enumerate(rejected, start=1):
                img_local = sp / f"img-{i:03d}.jpg"
                if not fetch_image_from_r2(r["imageR2Path"], img_local,
                                           bucket=R2_OUTPUT_BUCKET):
                    continue
                try:
                    with Image.open(img_local) as im:
                        feats = feat.features(im)
                    flat = feats.view(-1, feats.shape[-1])
                    neg_chunks.append(flat)
                    log(f"    [{i}/{len(rejected)}] "
                        f"{r['imageR2Path'].rsplit('/', 1)[-1]}: "
                        f"+{flat.shape[0]} negs")
                except Exception as e:
                    log(f"    WARN reject {r['imageR2Path']}: {e}")

    pos_bank = torch.cat(pos_chunks, dim=0)
    neg_bank = torch.cat(neg_chunks, dim=0)
    log(f"\n  bank: pos={pos_bank.shape[0]}  neg={neg_bank.shape[0]}  "
        f"dim={pos_bank.shape[1]}")

    # ---- Predict on all candidates ----
    candidates = get_candidates(condition=args.condition,
                                model_version=model_version,
                                limit=args.limit)
    if not candidates:
        log("\n  no candidates to predict on (already done?)")
        return 0
    log(f"  predicting on {len(candidates)} candidates ...")

    n_done = n_hit = n_err = 0
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="arem-fewshot-pred-") as scratch:
        sp = Path(scratch)
        for idx, image_key in enumerate(candidates, start=1):
            t0 = time.time()
            img_local = sp / f"img-{idx:05d}.jpg"
            if not fetch_image_from_r2(image_key, img_local,
                                       bucket=R2_OUTPUT_BUCKET):
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/auto-mask-test", {
                            "condition": args.condition,
                            "imageR2Path": image_key,
                            "modelVersion": model_version,
                            "prompt": f"fewshot:n={len(train_rows)}",
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
                    feats = feat.features(im)  # (G, G, D)
                q = feats.view(-1, feats.shape[-1])  # (G*G, D)
                pos_s = cosine_scores(q, pos_bank, topk=args.topk_pos)
                neg_s = cosine_scores(q, neg_bank, topk=args.topk_neg)
                score = (pos_s - neg_s).view(args.grid, args.grid)
                binary = (score > args.score_threshold).cpu().numpy()
                # Upsample to original resolution via PIL NEAREST
                mask = Image.fromarray((binary.astype(np.uint8) * 255), mode="L")
                mask = mask.resize((orig_w, orig_h), Image.NEAREST)
                px = int(np.asarray(mask, dtype=np.uint8).sum() // 255)
                mask_path: str | None = None
                if px > 0:
                    safe_name = re.sub(r"[^a-z0-9._-]+", "_",
                                       image_key.split("/")[-1].lower())
                    mask_key = (
                        f"auto-masks/{args.condition}/{model_version}/"
                        f"{safe_name.rsplit('.', 1)[0]}.png"
                    )
                    mask_local = sp / f"mask-{idx:05d}.png"
                    mask.save(mask_local, format="PNG", optimize=True)
                    if not args.dry_run and upload_mask(mask_local, mask_key):
                        mask_path = mask_key
                runtime_ms = int((time.time() - t0) * 1000)
                if not args.dry_run:
                    _request("POST", "/api/internal/auto-mask-test", {
                        "condition": args.condition,
                        "imageR2Path": image_key,
                        "modelVersion": model_version,
                        "prompt": f"fewshot:n={len(train_rows)}",
                        "autoMaskR2Path": mask_path,
                        "pixelCount": px,
                        "imageWidth": orig_w,
                        "imageHeight": orig_h,
                        "runtimeMs": runtime_ms,
                    })
                n_done += 1
                if px > 0:
                    n_hit += 1
                if idx % 10 == 0 or idx == len(candidates):
                    elapsed = time.time() - t_start
                    rate = idx / max(elapsed, 0.01)
                    log(f"  [{idx}/{len(candidates)}] {image_key}: "
                        f"{px} px {'(hit)' if px > 0 else '(empty)'} "
                        f"{runtime_ms}ms ({rate:.1f} img/s)")
            except Exception as e:
                log(f"  [{idx}/{len(candidates)}] {image_key}: ERROR {str(e)[:200]}")
                if not args.dry_run:
                    try:
                        _request("POST", "/api/internal/auto-mask-test", {
                            "condition": args.condition,
                            "imageR2Path": image_key,
                            "modelVersion": model_version,
                            "prompt": f"fewshot:n={len(train_rows)}",
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
    log(f"\n--- done in {dt:.1f}s ({dt/max(len(candidates),1):.2f}s/img avg) ---")
    log(f"  processed: {n_done}  hits: {n_hit}  errors: {n_err}")
    log(f"  model_version: {model_version}")
    log(f"  next: python -m scripts.score_against_ground_truth "
        f"--condition {args.condition} --heldout {args.heldout} "
        f"--model-version {model_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
