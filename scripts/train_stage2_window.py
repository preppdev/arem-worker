"""Stage-2 window-region fine-tune (pilot).

NAFNet w=32 only — this is the production Stage-2 arch (may26). Loads the
production checkpoint as the fine-tune base and reads its arch hyperparams
inline; never falls back to a default arch. The window-aware dataset and
mask-weighted loss focus learning on closing the window-region gap vs the
vendor target without disturbing the rest of the frame.

Usage on the 3090:
    cd ~/arem-photo-ai-V2
    python train_stage2_window.py \\
        --dataset-root /tmp/window_pilot_v1 \\
        --resume /tmp/may26_int.pth \\
        --epochs 5 \\
        --lr 2e-5 \\
        --crop 512,768 \\
        --batch-size 4 \\
        --output-dir window_fine_tune_pilot_v1
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv as csv_module
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pytorch_msssim import ssim as compute_ssim

from models.nafnet import NAFNet
from window_dataset import WindowStage2Dataset
from window_loss import MaskedCombinedLoss, _rgb_to_lab
from ema import EMA


def shoot_based_split(df, train_frac=0.9):
    df = df.copy()
    if "shoot_folder" in df.columns:
        df["_shoot"] = df["shoot_folder"]
    else:
        df["_shoot"] = df["pair_id"].str.rsplit("_", n=1).str[0]
    shoots = sorted(df["_shoot"].unique())
    split = int(len(shoots) * train_frac)
    train_shoots = set(shoots[:split])
    train_df = df[df["_shoot"].isin(train_shoots)].drop(columns=["_shoot"]).reset_index(drop=True)
    val_df = df[~df["_shoot"].isin(train_shoots)].drop(columns=["_shoot"]).reset_index(drop=True)
    return train_df, val_df


def compute_delta_e_window(pred, target, mask):
    """Mean ΔE (LAB euclidean) inside mask, evaluated in fp32."""
    with torch.no_grad():
        lab_p = _rgb_to_lab(pred.float())
        lab_t = _rgb_to_lab(target.float())
        diff = ((lab_p - lab_t) ** 2).sum(dim=1, keepdim=True).sqrt()
        m = mask.float()
        if m.dim() == 3:
            m = m.unsqueeze(1)
        m_sum = m.sum().clamp_min(1.0)
        return (diff * m).sum().item() / m_sum.item()


def build_nafnet_from_ckpt(ckpt_path: str, device):
    """Instantiate NAFNet using the arch hyperparams stored in the ckpt.

    The may26 checkpoints carry `width`, `enc_blk_nums`, `middle_blk_num`,
    `dec_blk_nums`, `use_residual`, and `residual_start` inline. We trust
    those over any defaults — if a key is missing we fall back to the
    documented may26 production config.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "arch" in ck and str(ck["arch"]).lower() != "nafnet":
        raise ValueError(
            f"refusing to fine-tune from a non-NAFNet checkpoint "
            f"(arch={ck['arch']!r}); production Stage-2 is NAFNet only"
        )
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

    # Prefer EMA weights if present (they're the production-shipping weights).
    if ck.get("ema_state_dict"):
        ema = ck["ema_state_dict"]
        sd = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
    else:
        sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)

    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dataset", default="pairs.csv")
    ap.add_argument("--resume", required=True,
                    help="Path to base Stage-2 checkpoint (e.g. interior_full_v1/best_lpips.pth)")
    ap.add_argument("--crop", default="512,768")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=1.0)
    ap.add_argument("--ssim-weight", type=float, default=0.5)
    ap.add_argument("--window-l1-weight", type=float, default=5.0)
    ap.add_argument("--window-chroma-weight", type=float, default=0.5)
    ap.add_argument("--window-chroma-warmup", type=int, default=2)
    ap.add_argument("--mask-overlap-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--output-dir", default="window_fine_tune_pilot_v1")
    args = ap.parse_args()

    device = torch.device("cuda")
    crop_parts = args.crop.split(",")
    crop_size = (int(crop_parts[0]), int(crop_parts[1]))

    run_name = args.output_dir
    ckpt_dir = Path(f"checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- Model (NAFNet, hyperparams from ckpt) -------------------------
    print(f"Loading base NAFNet checkpoint: {args.resume}")
    model, ckpt = build_nafnet_from_ckpt(args.resume, device)
    print(f"  resumed from epoch {ckpt.get('epoch')} "
          f"(metrics: {ckpt.get('metrics')})  "
          f"width={ckpt.get('width', 32)}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # -- Dataset --------------------------------------------------------
    dataset_root = Path(args.dataset_root)
    pairs_df = pd.read_csv(dataset_root / args.dataset)
    print(f"Manifest: {len(pairs_df)} rows")

    train_df, val_df = shoot_based_split(pairs_df, train_frac=0.9)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = WindowStage2Dataset(
        train_df, crop_size=crop_size, augment=True,
        path_prefix=str(dataset_root),
        mask_overlap_frac=args.mask_overlap_frac,
    )
    val_ds = WindowStage2Dataset(
        val_df, crop_size=crop_size, augment=False,
        path_prefix=str(dataset_root),
        mask_overlap_frac=args.mask_overlap_frac,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # -- Loss / optim ---------------------------------------------------
    criterion = MaskedCombinedLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.lpips_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=0.0,
        window_l1_weight=args.window_l1_weight,
        window_chroma_weight=args.window_chroma_weight,
        window_chroma_warmup_epochs=args.window_chroma_warmup,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, decay=0.999) if args.use_ema else None

    # -- Baseline ΔE_window before any training ------------------------
    print("\n=== Computing baseline ΔE_window (before fine-tune) ===")
    model.eval()
    base_deltas = []
    with torch.no_grad():
        for vi, (inp, tgt, msk) in enumerate(val_loader):
            if vi >= 50:
                break
            inp = inp.to(device); tgt = tgt.to(device); msk = msk.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inp)
            base_deltas.append(compute_delta_e_window(out, tgt, msk))
    baseline_de_window = float(np.mean(base_deltas))
    print(f"Baseline ΔE_window mean (50 val crops): {baseline_de_window:.3f}\n")

    # -- Log ------------------------------------------------------------
    Path("logs").mkdir(exist_ok=True)
    log_csv = f"logs/window_pilot_{run_name}_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv_module.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "ssim", "delta_e_window",
            "delta_e_window_delta_pct", "lr", "time_min"])

    print(f"=== Pilot fine-tune ({args.epochs} epochs · {n_params:.1f}M params · "
          f"baseline ΔE_window={baseline_de_window:.3f}) ===")

    # -- Train ----------------------------------------------------------
    best_de = float("inf")
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        criterion.set_epoch(epoch)

        model.train()
        train_losses = []
        for step, (inp, tgt, msk) in enumerate(train_loader):
            inp = inp.to(device); tgt = tgt.to(device); msk = msk.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inp)
                ld = criterion(out, tgt, msk)
            scaler.scale(ld["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            train_losses.append(ld["total"].item())

            if (step + 1) % 25 == 0:
                avg = float(np.mean(train_losses[-25:]))
                print(f"  ep{epoch} step {step+1}/{len(train_loader)} "
                      f"loss={avg:.4f} win_l1={ld.get('window_l1',0):.4f} "
                      f"win_chroma={ld.get('window_chroma',0):.4f}", flush=True)
        scheduler.step()

        # -- Validate ---
        if ema:
            ema.apply(model)
        model.eval()
        val_losses = []
        ssim_scores = []
        deltas = []
        with torch.no_grad():
            for vi, (inp, tgt, msk) in enumerate(val_loader):
                if vi >= 100:
                    break
                inp = inp.to(device); tgt = tgt.to(device); msk = msk.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(inp)
                    ld = criterion(out, tgt, msk)
                val_losses.append(ld["total"].item())
                ssim_scores.append(compute_ssim(
                    out.float(), tgt.float(), data_range=1.0,
                    size_average=True).item())
                deltas.append(compute_delta_e_window(out, tgt, msk))

        avg_val = float(np.mean(val_losses))
        avg_ssim = float(np.mean(ssim_scores))
        de_window = float(np.mean(deltas))
        de_pct = (de_window - baseline_de_window) / baseline_de_window * 100.0
        ep_time = (time.time() - ep_start) / 60.0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs}  train={float(np.mean(train_losses)):.4f}  "
              f"val={avg_val:.4f}  ssim={avg_ssim:.4f}  ΔE_window={de_window:.3f} "
              f"({de_pct:+.1f}% vs baseline)  lr={lr:.6f}  time={ep_time:.1f}min",
              flush=True)

        with open(log_csv, "a", newline="") as f:
            csv_module.writer(f).writerow([
                epoch, float(np.mean(train_losses)), avg_val, avg_ssim,
                de_window, de_pct, lr, ep_time])

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": {
                "val_loss": avg_val, "ssim": avg_ssim,
                "delta_e_window": de_window,
                "baseline_delta_e_window": baseline_de_window,
            },
        }
        if ema:
            save_dict["ema_state_dict"] = ema.state_dict()
        torch.save(save_dict, str(ckpt_dir / "latest.pth"))

        if de_window < best_de:
            best_de = de_window
            torch.save(save_dict, str(ckpt_dir / "best_de_window.pth"))
            print(f"  new best ΔE_window: {best_de:.3f}", flush=True)

        if ema:
            ema.restore(model)

        # Upload epoch to R2
        import subprocess
        subprocess.run(
            ["rclone", "copy", str(ckpt_dir),
             f"r2:arem-training-data/checkpoints/{run_name}/", "-q"],
            timeout=300,
        )

    print(f"\nDone. baseline ΔE_window={baseline_de_window:.3f}  "
          f"best ΔE_window={best_de:.3f}  "
          f"delta={(best_de-baseline_de_window)/baseline_de_window*100:+.1f}%")


if __name__ == "__main__":
    main()
