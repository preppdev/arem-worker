"""Guardrail + canary eval for the Stage-2 window-mask pilot.

Two-pass design to fit in 24 GB:
  Pass 1: load BASE model, run inference on val set, dump outputs to disk
  Pass 2: load FINE-TUNED model, run inference on same val set
  Pass 3: compute metrics from the on-disk outputs (no models resident)

Inputs are also capped at 1280px on the long edge before inference —
the guardrail metrics are aggregates so the exact resolution doesn't
move them meaningfully, and full-res 6-MP inputs blow VRAM when paired
with a model that's already 30 M params.


For each model (base may26 NAFNet + fine-tuned window pilot), runs the val
manifest through inference and computes:

  Guardrails (global, must stay within spec bands):
    - val LPIPS
    - val ΔE_mean (full-frame LAB euclidean)
    - val SSIM
  Primary (the metric we trained for):
    - val ΔE_window  (mean LAB inside the mask)

Then for the top N val rows by mask area, renders full-resolution
triptychs:  input | vendor target | fine-tuned output  (with mask outline
overlaid on the output), saves to a local dir, and rclones to R2 so the
dashboard can render them.

Usage on the 3090:
    cd ~/arem-photo-ai-V2
    python eval_window_pilot.py \\
        --dataset-root /tmp/window_pilot_v1 \\
        --base /tmp/may26_int.pth \\
        --finetuned checkpoints/window_fine_tune_pilot_v1/best_de_window.pth \\
        --canary-out /tmp/window_canary_v1 \\
        --canary-n 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import lpips as lpips_lib  # type: ignore
import numpy as np
import pandas as pd
import torch
from PIL import Image  # type: ignore
from pytorch_msssim import ssim as compute_ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.nafnet import NAFNet
from window_loss import _rgb_to_lab


def build_nafnet_from_ckpt(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "arch" in ck and str(ck["arch"]).lower() != "nafnet":
        raise ValueError(f"non-NAFNet ckpt: {ck.get('arch')!r}")
    model = NAFNet(
        in_channels=ck.get("in_channels", 3),
        out_channels=ck.get("out_channels", 3),
        width=ck.get("width", 32),
        middle_blk_num=ck.get("middle_blk_num", 12),
        enc_blk_nums=ck.get("enc_blk_nums", [2, 2, 4, 8]),
        dec_blk_nums=ck.get("dec_blk_nums", [2, 2, 2, 2]),
        use_residual=ck.get("use_residual", True),
        residual_start=ck.get("residual_start", 0),
    ).to(device)
    if ck.get("ema_state_dict"):
        ema = ck["ema_state_dict"]
        sd = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
    else:
        sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


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


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(str(path))
    return (arr > 127).astype(np.uint8)


def cap_long_edge(arr: np.ndarray, max_long: int) -> np.ndarray:
    h, w = arr.shape[:2]
    long = max(h, w)
    if long <= max_long:
        return arr
    scale = max_long / long
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if arr.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


def crop_to_multiple_of_16(arr: np.ndarray, ref_hw: tuple[int, int] | None = None):
    h, w = arr.shape[:2]
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    if ref_hw is not None:
        h16, w16 = ref_hw
    return arr[:h16, :w16]


@torch.no_grad()
def model_infer(model, img_rgb: np.ndarray, device, max_long: int = 1280) -> tuple[np.ndarray, tuple[int, int]]:
    capped = cap_long_edge(img_rgb, max_long)
    capped = crop_to_multiple_of_16(capped)
    x = torch.from_numpy(capped.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
    y = torch.clamp(y.float(), 0, 1)
    out = (y.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    del x, y
    return out, capped.shape[:2]


def lab_diff_mean(pred_rgb: np.ndarray, tgt_rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    p = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    lab_p = _rgb_to_lab(p)
    lab_t = _rgb_to_lab(t)
    diff = ((lab_p - lab_t) ** 2).sum(dim=1, keepdim=True).sqrt().squeeze().numpy()
    if mask is not None:
        return float(diff[mask > 0].mean()) if mask.any() else 0.0
    return float(diff.mean())


def lpips_score(pred_rgb, tgt_rgb, lpips_fn) -> float:
    p = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t = torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return float(lpips_fn(p, t).mean().item())


def ssim_score(pred_rgb, tgt_rgb) -> float:
    p = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return float(compute_ssim(p, t, data_range=1.0, size_average=True).item())


def render_triptych(input_rgb, vendor_rgb, ours_rgb, mask, out_path: Path) -> None:
    """Side-by-side input | vendor | ours, with red mask outline on each."""
    h, w = ours_rgb.shape[:2]
    # Align all to the same shape (model output may be slightly smaller after //16 crop)
    input_c = cv2.resize(input_rgb, (w, h), interpolation=cv2.INTER_AREA)
    vendor_c = cv2.resize(vendor_rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask_c = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Mask outline overlay — dilated boundary
    edge = cv2.dilate(mask_c, np.ones((3, 3), np.uint8), iterations=2) - mask_c
    def overlay(img):
        out = img.copy()
        out[edge > 0] = [255, 64, 64]  # red outline
        return out
    triptych = np.hstack([overlay(input_c), overlay(vendor_c), overlay(ours_rgb)])
    # Labels
    label_band = np.zeros((36, triptych.shape[1], 3), dtype=np.uint8)
    for i, txt in enumerate(("input (stage1)", "vendor target", "ours (fine-tuned)")):
        cv2.putText(label_band, txt, (i * w + 16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    full = np.vstack([label_band, triptych])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dataset", default="pairs.csv")
    ap.add_argument("--base", required=True, help="Path to base may26 NAFNet ckpt")
    ap.add_argument("--finetuned", required=True, help="Path to fine-tuned ckpt")
    ap.add_argument("--canary-out", default="/tmp/window_canary_v1")
    ap.add_argument("--canary-n", type=int, default=10)
    ap.add_argument("--canary-r2-prefix", default="window-pilot/canary-v1")
    args = ap.parse_args()

    device = torch.device("cuda")
    dataset_root = Path(args.dataset_root)
    pairs_df = pd.read_csv(dataset_root / args.dataset)
    _, val_df = shoot_based_split(pairs_df, train_frac=0.9)
    print(f"Val rows: {len(val_df)}")

    lpips_fn = lpips_lib.LPIPS(net="alex").cpu().eval()  # keep on CPU

    # ---- Pass 1: BASE inference, write outputs to /tmp -----------------
    base_dir = Path("/tmp/window_eval_base_outs")
    base_dir.mkdir(parents=True, exist_ok=True)
    print("Pass 1: BASE inference …")
    base = build_nafnet_from_ckpt(args.base, device)
    t0 = time.time()
    for vi, row in val_df.iterrows():
        try:
            inp = load_rgb(dataset_root / row["input"])
            out, _ = model_infer(base, inp, device)
            Image.fromarray(out).save(base_dir / f"{row['pair_id']}.jpg", quality=95)
            torch.cuda.empty_cache()
            if (vi + 1) % 20 == 0:
                print(f"  base [{vi+1}/{len(val_df)}]  {(time.time()-t0)/(vi+1):.1f}s/img", flush=True)
        except Exception as e:
            print(f"  base ERR {row['pair_id']}: {e}")
    del base
    torch.cuda.empty_cache()

    # ---- Pass 2: FINE inference, write outputs to /tmp -----------------
    fine_dir = Path("/tmp/window_eval_fine_outs")
    fine_dir.mkdir(parents=True, exist_ok=True)
    print("Pass 2: FINE-TUNED inference …")
    fine = build_nafnet_from_ckpt(args.finetuned, device)
    t0 = time.time()
    for vi, row in val_df.iterrows():
        try:
            inp = load_rgb(dataset_root / row["input"])
            out, _ = model_infer(fine, inp, device)
            Image.fromarray(out).save(fine_dir / f"{row['pair_id']}.jpg", quality=95)
            torch.cuda.empty_cache()
            if (vi + 1) % 20 == 0:
                print(f"  fine [{vi+1}/{len(val_df)}]  {(time.time()-t0)/(vi+1):.1f}s/img", flush=True)
        except Exception as e:
            print(f"  fine ERR {row['pair_id']}: {e}")
    # Keep `fine` resident for the canary triptychs at the end
    torch.cuda.empty_cache()

    # ---- Pass 3: compute metrics from on-disk outputs ------------------
    print("Pass 3: computing metrics …")
    rows = []
    for vi, row in val_df.iterrows():
        pair_id = row["pair_id"]
        base_pth = base_dir / f"{pair_id}.jpg"
        fine_pth = fine_dir / f"{pair_id}.jpg"
        if not base_pth.exists() or not fine_pth.exists():
            continue
        try:
            tgt_raw = load_rgb(dataset_root / row["output"])
            msk_raw = load_mask(dataset_root / row["mask"])
            base_out = load_rgb(base_pth)
            fine_out = load_rgb(fine_pth)
            # all model outputs were cropped to multiples of 16 after long-edge cap;
            # target and mask come from full-res, so resize them to match outputs.
            h, w = base_out.shape[:2]
            tgt_c = cv2.resize(tgt_raw, (w, h), interpolation=cv2.INTER_AREA)
            msk_c = cv2.resize(msk_raw, (w, h), interpolation=cv2.INTER_NEAREST)

            rows.append({
                "pair_id": pair_id,
                "mask_pixels": int(msk_c.sum()),
                "mask_area_pct": float(100 * msk_c.sum() / (h * w)),
                "base_de_global": lab_diff_mean(base_out, tgt_c),
                "fine_de_global": lab_diff_mean(fine_out, tgt_c),
                "base_de_window": lab_diff_mean(base_out, tgt_c, msk_c),
                "fine_de_window": lab_diff_mean(fine_out, tgt_c, msk_c),
                "base_lpips": lpips_score(base_out, tgt_c, lpips_fn),
                "fine_lpips": lpips_score(fine_out, tgt_c, lpips_fn),
                "base_ssim": ssim_score(base_out, tgt_c),
                "fine_ssim": ssim_score(fine_out, tgt_c),
            })
            if (vi + 1) % 20 == 0:
                print(f"  metrics [{vi+1}/{len(val_df)}]", flush=True)
        except Exception as e:
            print(f"  metrics ERR {pair_id}: {e}")

    eval_df = pd.DataFrame(rows)
    summary = {
        "n_val": len(eval_df),
        "base_de_global_mean": eval_df["base_de_global"].mean(),
        "fine_de_global_mean": eval_df["fine_de_global"].mean(),
        "de_global_delta_pct": (eval_df["fine_de_global"].mean() - eval_df["base_de_global"].mean())
            / eval_df["base_de_global"].mean() * 100,
        "base_de_window_mean": eval_df["base_de_window"].mean(),
        "fine_de_window_mean": eval_df["fine_de_window"].mean(),
        "de_window_delta_pct": (eval_df["fine_de_window"].mean() - eval_df["base_de_window"].mean())
            / eval_df["base_de_window"].mean() * 100,
        "base_lpips_mean": eval_df["base_lpips"].mean(),
        "fine_lpips_mean": eval_df["fine_lpips"].mean(),
        "lpips_delta_pct": (eval_df["fine_lpips"].mean() - eval_df["base_lpips"].mean())
            / eval_df["base_lpips"].mean() * 100,
        "base_ssim_mean": eval_df["base_ssim"].mean(),
        "fine_ssim_mean": eval_df["fine_ssim"].mean(),
    }
    print("\n=== Guardrail eval ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s}  {v:.4f}")
        else:
            print(f"  {k:30s}  {v}")

    # Pass/fail bands per spec
    print("\n=== Spec gate ===")
    ok_de_window = summary["de_window_delta_pct"] <= -40.0
    ok_de_global = abs(summary["de_global_delta_pct"]) <= 5.0
    ok_lpips = abs(summary["lpips_delta_pct"]) <= 3.0
    print(f"  primary ΔE_window ≤ -40%: {summary['de_window_delta_pct']:+.1f}%  "
          f"→ {'PASS' if ok_de_window else 'FAIL'}")
    print(f"  guardrail global ΔE within ±5%: {summary['de_global_delta_pct']:+.1f}%  "
          f"→ {'PASS' if ok_de_global else 'FAIL'}")
    print(f"  guardrail LPIPS within ±3%: {summary['lpips_delta_pct']:+.1f}%  "
          f"→ {'PASS' if ok_lpips else 'FAIL'}")

    eval_df.to_csv("/tmp/window_pilot_eval_per_image.csv", index=False)
    with open("/tmp/window_pilot_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Canary triptychs: pick top N by mask area --------------------
    # Reuse the already-rendered fine outputs from pass 2 (no GPU work here).
    print(f"\n=== Rendering {args.canary_n} canary triptychs ===")
    canary_dir = Path(args.canary_out)
    canary_dir.mkdir(parents=True, exist_ok=True)
    top = eval_df.nlargest(args.canary_n, "mask_pixels")
    for _, row in top.iterrows():
        pair_id = row["pair_id"]
        manifest_row = val_df[val_df["pair_id"] == pair_id].iloc[0]
        inp = load_rgb(dataset_root / manifest_row["input"])
        tgt = load_rgb(dataset_root / manifest_row["output"])
        msk = load_mask(dataset_root / manifest_row["mask"])
        fine_out = load_rgb(fine_dir / f"{pair_id}.jpg")
        h, w = fine_out.shape[:2]
        # resize input/target/mask to match fine_out's dims (which were
        # capped at 1280 long edge and cropped to //16)
        inp_c = cv2.resize(inp, (w, h), interpolation=cv2.INTER_AREA)
        tgt_c = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)
        msk_c = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
        render_triptych(inp_c, tgt_c, fine_out, msk_c, canary_dir / f"{pair_id}.jpg")
        print(f"  rendered {pair_id} (mask {row['mask_area_pct']:.1f}%, "
              f"de_window {row['fine_de_window']:.2f})")

    # ---- Upload to R2 -------------------------------------------------
    print(f"\nUploading triptychs to r2:arem-training-data/{args.canary_r2_prefix}/ …")
    res = subprocess.run(
        ["rclone", "copy", str(canary_dir),
         f"r2:arem-training-data/{args.canary_r2_prefix}/", "-q"],
        capture_output=True, text=True, timeout=300,
    )
    if res.returncode != 0:
        print(f"  rclone failed: {res.stderr[:300]}")
    else:
        print("  upload OK")

    print("\nDone.")


if __name__ == "__main__":
    main()
