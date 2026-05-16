"""Score existing AutoMaskTest rows against hand-labeled ground truth.

For each (condition, modelVersion) tuple in AutoMaskTest, look up the
intersection with the hand-labeled set (Annotation table, decision=edited),
compute IoU per image, and PATCH the iou value back via
/api/internal/auto-mask-iou. The /auto-mask/<condition> gallery already
renders the iou column.

Usage:
    python -m scripts.score_against_ground_truth \\
        --condition reflection \\
        --heldout 5

Env:
    WORKER_TOKEN              dashboard auth (required)
    DASHBOARD_URL             default https://arem-editing-dashboard.vercel.app
    R2_BUCKET                 default arem-training-data (masks live here)
    RCLONE_R2                 default 'r2'

What --heldout does:
    The hand-labeled set is split into a deterministic eval slice (the
    first N by createdAt, oldest first) and a train slice (everything
    else). We only PATCH iou for rows in the eval slice — the train
    slice is reserved as reference exemplars for downstream few-shot
    models that will be trained / conditioned on it. Pass --heldout 0
    to score against every hand-labeled mask (use this when you have
    enough labels that holdout doesn't matter, or for retrospective
    scoring of CLIPSeg runs before any model uses the labels).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np  # type: ignore
from PIL import Image  # type: ignore

DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
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


def fetch_truth_masks(condition: str) -> list[dict]:
    sp = urllib.parse.urlencode({"condition": condition})
    return _request("GET", f"/api/internal/ground-truth-masks?{sp}")["items"]


def fetch_mask_from_r2(r2_key: str, dest: Path) -> bool:
    src = f"{RCLONE_R2}:{R2_BUCKET}/{r2_key}"
    r = subprocess.run(
        ["rclone", "copyto", src, str(dest), "--progress=false"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        log(f"  WARN fetch failed {r2_key}: rc={r.returncode} "
            f"{r.stderr[-160:].strip()}")
        return False
    return True


def load_binary_mask(path: Path) -> np.ndarray:
    """Open mask, convert to bool array (True = positive). PNGs from
    /label/decide are L-mode; some upstream masks may be RGBA. Treat
    any pixel above 0 as positive."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return arr > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        # Both empty: trivially perfect agreement on "nothing here".
        return 1.0
    return float(inter) / float(union)


def get_model_versions(condition: str) -> list[str]:
    """Pull distinct model versions for the condition from the public
    auto-mask summary endpoint (cookie-auth in browser; we hit the
    internal route that the gallery uses). We just need the list of
    model versions to iterate; the summary itself isn't strictly
    needed."""
    sp = urllib.parse.urlencode({"limit": "1"})
    data = _request("GET", f"/api/auto-mask/{condition}?{sp}")
    return [s["modelVersion"] for s in data.get("summary", [])]


def get_automask_rows(condition: str, model_version: str,
                      image_paths: list[str]) -> dict[str, str | None]:
    """Return {imageR2Path: autoMaskR2Path|None} for the given truth
    images. Uses the /api/auto-mask/<condition> endpoint with the
    modelVersion filter and onlyHits=false. Caller only consumes rows
    whose imageR2Path is in image_paths."""
    out: dict[str, str | None] = {}
    sp = urllib.parse.urlencode({"modelVersion": model_version, "limit": "500"})
    data = _request("GET", f"/api/auto-mask/{condition}?{sp}")
    truth_set = set(image_paths)
    for r in data.get("items", []):
        if r["imageR2Path"] in truth_set:
            out[r["imageR2Path"]] = r.get("autoMaskR2Path")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--heldout", type=int, default=0,
                    help="N oldest-by-createdAt to use as eval; rest are "
                         "reserved as train. 0 = score against all (use for "
                         "retrospective scoring of CLIPSeg runs).")
    ap.add_argument("--model-version", default=None,
                    help="Only score this modelVersion; default scores all.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        log("ERROR: WORKER_TOKEN env var is required")
        return 2

    log(f"score-against-truth: condition={args.condition} heldout={args.heldout}")
    truth = fetch_truth_masks(args.condition)
    log(f"  hand-labeled rows: {len(truth)}")
    if not truth:
        log("  nothing to score")
        return 0

    if args.heldout > 0:
        eval_rows = truth[: args.heldout]
        train_rows = truth[args.heldout:]
        log(f"  eval: {len(eval_rows)}  train: {len(train_rows)}")
        scoring = eval_rows
    else:
        scoring = truth
        log(f"  scoring against ALL {len(truth)} hand-labeled rows (no holdout)")

    truth_by_image = {r["imageR2Path"]: r for r in scoring}
    image_paths = list(truth_by_image.keys())

    model_versions = (
        [args.model_version] if args.model_version
        else get_model_versions(args.condition)
    )
    log(f"  model versions to score: {model_versions}")

    with tempfile.TemporaryDirectory(prefix="arem-score-") as scratch:
        scratch_p = Path(scratch)

        # Pre-fetch truth masks once (re-used across model versions).
        log("  fetching truth masks ...")
        truth_arrays: dict[str, np.ndarray] = {}
        for i, r in enumerate(scoring, start=1):
            dst = scratch_p / f"truth-{i:04d}.png"
            if not fetch_mask_from_r2(r["maskR2Path"], dst):
                continue
            try:
                truth_arrays[r["imageR2Path"]] = load_binary_mask(dst)
            except Exception as e:
                log(f"    WARN load truth {r['maskR2Path']}: {e}")
        log(f"  truth masks loaded: {len(truth_arrays)}")

        for mv in model_versions:
            log(f"\n  -- model_version={mv}")
            rows = get_automask_rows(args.condition, mv, image_paths)
            log(f"     auto-mask rows matching truth set: {len(rows)}")
            patches: list[dict] = []
            ious: list[float] = []
            t0 = time.time()
            for i, (img_path, auto_mask_key) in enumerate(rows.items(), start=1):
                truth_arr = truth_arrays.get(img_path)
                if truth_arr is None:
                    continue
                if auto_mask_key is None:
                    # Empty prediction → IoU = 0 unless truth is also empty
                    pred_arr = np.zeros_like(truth_arr)
                else:
                    dst = scratch_p / f"pred-{i:04d}.png"
                    if not fetch_mask_from_r2(auto_mask_key, dst):
                        continue
                    try:
                        pred_arr = load_binary_mask(dst)
                    except Exception as e:
                        log(f"     WARN load pred {auto_mask_key}: {e}")
                        continue
                # Resize pred to truth shape if needed (rare; masks should
                # already be at source resolution, but be defensive).
                if pred_arr.shape != truth_arr.shape:
                    with Image.fromarray(
                        (pred_arr * 255).astype(np.uint8), mode="L",
                    ) as im:
                        im = im.resize(
                            (truth_arr.shape[1], truth_arr.shape[0]),
                            Image.NEAREST,
                        )
                        pred_arr = np.asarray(im, dtype=np.uint8) > 0
                v = iou(pred_arr, truth_arr)
                ious.append(v)
                patches.append({
                    "condition": args.condition,
                    "imageR2Path": img_path,
                    "modelVersion": mv,
                    "iou": v,
                })
            dt = time.time() - t0
            if ious:
                arr = np.asarray(ious)
                log(f"     scored {len(ious)} images in {dt:.1f}s  "
                    f"mean IoU={arr.mean():.3f}  median={np.median(arr):.3f}  "
                    f"max={arr.max():.3f}")
            else:
                log("     no matching images to score")
            if patches and not args.dry_run:
                resp = _request("PATCH", "/api/internal/auto-mask-iou",
                                {"items": patches})
                log(f"     patched: updated={resp.get('updated')} "
                    f"missing={resp.get('missing')}")
            elif args.dry_run:
                log(f"     (dry-run) would PATCH {len(patches)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
