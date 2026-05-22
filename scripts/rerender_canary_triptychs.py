"""Re-render the window-pilot canary triptychs as base | vendor | fine.

Uses the cached inference outputs from eval_window_pilot.py's pass-1/pass-2
(at /tmp/window_eval_base_outs/ and /tmp/window_eval_fine_outs/). No GPU
inference here — pure compositing.

Layout per row:
    BASE (may26 NAFNet)  |  VENDOR TARGET  |  FINE-TUNED (pilot)
with a red mask outline drawn on each panel so the eye locks to the
window region.

Usage on the 3090:
    cd ~/arem-photo-ai-V2
    python rerender_canary_triptychs.py \\
        --dataset-root /tmp/window_pilot_v1 \\
        --base-outs /tmp/window_eval_base_outs \\
        --fine-outs /tmp/window_eval_fine_outs \\
        --canary-out /tmp/window_canary_v1 \\
        --canary-n 10 \\
        --canary-r2-prefix window-pilot/canary-v1
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image  # type: ignore


def shoot_based_split(df, train_frac=0.9):
    df = df.copy()
    if "shoot_folder" in df.columns:
        df["_shoot"] = df["shoot_folder"]
    else:
        df["_shoot"] = df["pair_id"].str.rsplit("_", n=1).str[0]
    shoots = sorted(df["_shoot"].unique())
    split = int(len(shoots) * train_frac)
    train_shoots = set(shoots[:split])
    return (
        df[df["_shoot"].isin(train_shoots)].drop(columns=["_shoot"]).reset_index(drop=True),
        df[~df["_shoot"].isin(train_shoots)].drop(columns=["_shoot"]).reset_index(drop=True),
    )


def load_rgb(p: Path) -> np.ndarray:
    arr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(str(p))
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_mask(p: Path) -> np.ndarray:
    arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(str(p))
    return (arr > 127).astype(np.uint8)


def render_triptych(base_rgb, vendor_rgb, fine_rgb, mask, out_path: Path) -> None:
    h, w = fine_rgb.shape[:2]
    # Make all three panels share fine_rgb's dims (the model output is the
    # canonical eval-time resolution).
    base_c = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_AREA) if base_rgb.shape[:2] != (h, w) else base_rgb
    vendor_c = cv2.resize(vendor_rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask_c = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Dilated boundary as red outline
    edge = cv2.dilate(mask_c, np.ones((3, 3), np.uint8), iterations=2) - mask_c

    def overlay(img):
        out = img.copy()
        out[edge > 0] = [255, 64, 64]
        return out

    triptych = np.hstack([overlay(base_c), overlay(vendor_c), overlay(fine_rgb)])
    label_band = np.zeros((36, triptych.shape[1], 3), dtype=np.uint8)
    for i, txt in enumerate(("base (may26 NAFNet)", "vendor target", "ours (fine-tuned)")):
        cv2.putText(label_band, txt, (i * w + 16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    full = np.vstack([label_band, triptych])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dataset", default="pairs.csv")
    ap.add_argument("--base-outs", required=True)
    ap.add_argument("--fine-outs", required=True)
    ap.add_argument("--canary-out", required=True)
    ap.add_argument("--canary-n", type=int, default=10)
    ap.add_argument("--canary-r2-prefix", default="window-pilot/canary-v1")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    base_dir = Path(args.base_outs)
    fine_dir = Path(args.fine_outs)
    canary_dir = Path(args.canary_out)

    pairs_df = pd.read_csv(dataset_root / args.dataset)
    _, val_df = shoot_based_split(pairs_df, train_frac=0.9)

    # Rank val rows by mask area; reuse the same selection logic as the
    # original eval so the same pair_ids land in the canary set.
    sizes = []
    for _, row in val_df.iterrows():
        pid = row["pair_id"]
        try:
            m = load_mask(dataset_root / row["mask"])
            sizes.append((pid, int(m.sum())))
        except Exception:
            sizes.append((pid, 0))
    sizes.sort(key=lambda t: t[1], reverse=True)
    top = sizes[: args.canary_n]
    print(f"Top {len(top)} by mask area:")

    for pid, _ in top:
        manifest_row = val_df[val_df["pair_id"] == pid].iloc[0]
        base_path = base_dir / f"{pid}.jpg"
        fine_path = fine_dir / f"{pid}.jpg"
        if not base_path.exists() or not fine_path.exists():
            print(f"  SKIP {pid}: cached outputs missing")
            continue
        base_out = load_rgb(base_path)
        fine_out = load_rgb(fine_path)
        vendor = load_rgb(dataset_root / manifest_row["output"])
        mask = load_mask(dataset_root / manifest_row["mask"])
        out_path = canary_dir / f"{pid}.jpg"
        render_triptych(base_out, vendor, fine_out, mask, out_path)
        print(f"  rendered {pid}")

    print(f"\nUploading to r2:arem-training-data/{args.canary_r2_prefix}/ …")
    res = subprocess.run(
        ["rclone", "copy", str(canary_dir),
         f"r2:arem-training-data/{args.canary_r2_prefix}/", "-q"],
        capture_output=True, text=True, timeout=300,
    )
    if res.returncode != 0:
        print(f"  rclone failed: {res.stderr[:300]}")
    else:
        print("  upload OK")


if __name__ == "__main__":
    main()
