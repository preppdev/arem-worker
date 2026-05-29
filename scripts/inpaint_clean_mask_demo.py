"""Clean-mask inpaint demo: prove that SAM2 box-derived masks beat the
fragmented diff masks the bake-off used.

For each staged (src, diff_mask) pair from the bake-off:
  1. Compute the bounding box of the (Swiss-cheese) diff mask.
  2. Feed that box to SAM2 -> clean, tight object mask.
  3. Run LaMa + Flux Fill + ObjectClear on the CLEAN mask.
  4. Composite: [SOURCE | DIFF-MASK | SAM-MASK | LaMa | FluxFill | ObjectClear]

This isolates the variable that was hurting the earlier bake-off — mask
quality — while reusing the exact same inpaint method modules.

Usage:
    python -m scripts.inpaint_clean_mask_demo --limit 8 --out /tmp/clean_mask_demo
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

sys.path.insert(0, "/home/jordan/arem-worker")
from scripts.sam_box_mask import box_to_mask  # type: ignore
from scripts.inpaint_bakeoff import compute_roi, paste_with_feather, make_method  # type: ignore

STAGE_DIR = Path("/tmp/inpaint_bakeoff_stage")
METHODS = ["lama", "flux_fill", "object_clear"]


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 8)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay(src: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    vis = src.copy()
    layer = np.zeros_like(src)
    layer[mask > 8] = color
    return cv2.addWeighted(vis, 0.6, layer, 0.4, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", default="/tmp/clean_mask_demo")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    srcs = sorted(glob.glob(str(STAGE_DIR / "*__src.jpg")))[: args.limit]
    rows = []
    for sp in srcs:
        stem = Path(sp).name.replace("__src.jpg", "")
        mp = STAGE_DIR / f"{stem}__mask.png"
        if not mp.is_file():
            continue
        rows.append({"stem": stem, "src": sp, "diff_mask": str(mp)})
    print(f"demo set: {len(rows)} images", flush=True)

    # 1) Build SAM2 clean masks (model stays loaded across all rows).
    print("=== SAM2 clean masks ===", flush=True)
    for r in rows:
        src = cv2.imread(r["src"])
        diff = cv2.imread(r["diff_mask"], cv2.IMREAD_GRAYSCALE)
        if diff.shape[:2] != src.shape[:2]:
            diff = cv2.resize(diff, (src.shape[1], src.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        bb = mask_bbox(diff)
        if bb is None:
            r["sam_mask"] = None
            continue
        clean = box_to_mask(src, bb)
        cm_path = out / "img" / f"{r['stem']}__sammask.png"
        cv2.imwrite(str(cm_path), clean)
        r["sam_mask"] = str(cm_path)
        r["_diff_resized"] = diff
        print(f"  {r['stem']}: diff={int((diff>8).sum())}px -> "
              f"sam={int((clean>8).sum())}px", flush=True)

    # 2) Run each method on the SAM mask (load/unload one at a time).
    for name in METHODS:
        print(f"=== {name} ===", flush=True)
        method = make_method(name)
        t0 = time.time()
        method.load()
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        for r in rows:
            if not r.get("sam_mask"):
                continue
            src = cv2.imread(r["src"])
            mask = cv2.imread(r["sam_mask"], cv2.IMREAD_GRAYSCALE)
            roi = compute_roi(mask)
            x0, y0, x1, y1 = roi
            try:
                patch = method.run(src[y0:y1, x0:x1], mask[y0:y1, x0:x1])
                res = paste_with_feather(src, patch, mask, roi)
            except Exception as e:
                print(f"  {r['stem']} {name} failed: {e}", flush=True)
                continue
            cv2.imwrite(str(out / "img" / f"{r['stem']}__{name}.jpg"), res,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        method.unload()

    # 3) Composites.
    print("=== composites ===", flush=True)
    comp_dir = out / "composites"
    comp_dir.mkdir(exist_ok=True)
    H = 520
    for r in rows:
        src = cv2.imread(r["src"])
        diff = r.get("_diff_resized")
        if diff is None:
            diff = cv2.imread(r["diff_mask"], cv2.IMREAD_GRAYSCALE)
            if diff.shape[:2] != src.shape[:2]:
                diff = cv2.resize(diff, (src.shape[1], src.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        panels = [("SOURCE", src),
                  ("DIFF MASK (old)", overlay(src, diff, (40, 80, 200)))]
        if r.get("sam_mask"):
            sam = cv2.imread(r["sam_mask"], cv2.IMREAD_GRAYSCALE)
            panels.append(("SAM MASK (new)", overlay(src, sam, (60, 200, 60))))
        for name in METHODS:
            p = out / "img" / f"{r['stem']}__{name}.jpg"
            if p.is_file():
                panels.append((name.upper(), cv2.imread(str(p))))
        fitted = []
        for label, im in panels:
            h, w = im.shape[:2]
            fitted.append((label, cv2.resize(im, (int(w * H / h), H),
                                             interpolation=cv2.INTER_AREA)))
        gap = np.zeros((H, 8, 3), np.uint8)
        strip = fitted[0][1]
        for _, im in fitted[1:]:
            strip = np.concatenate([strip, gap, im], axis=1)
        bar = np.full((26, strip.shape[1], 3), 20, np.uint8)
        x = 0
        for label, im in fitted:
            cv2.putText(bar, label, (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1, cv2.LINE_AA)
            x += im.shape[1] + 8
        cv2.imwrite(str(comp_dir / f"{r['stem']}.jpg"),
                    np.concatenate([bar, strip], axis=0),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print(f"composites in {comp_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
