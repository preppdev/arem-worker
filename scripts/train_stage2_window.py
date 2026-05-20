"""Stage-2 window-region fine-tune.

NAFNet w=32 only — this is the production Stage-2 arch (may26). Loads the
production checkpoint as the fine-tune base and reads its arch hyperparams
inline; never falls back to a default arch. The window-aware dataset and
mask-weighted loss focus learning on closing the window-region gap vs the
vendor target without disturbing the rest of the frame.

The trainer creates a `TrainingRun` row on the dashboard at startup, then
posts a per-epoch `TrainingEpoch` row (with metrics) and N
`TrainingCanaryImage` rows (each with a base|vendor|fine triptych) at each
epoch boundary. Canaries land at:
    r2:arem-training-data/training-runs/<runId>/epoch_<NN>/<pair_id>.jpg

The dashboard renders them at /training/[runId] as the run progresses.

Usage on Vast / 3090:
    cd ~/arem-photo-ai-V2
    set -a; source /root/arem-prod.env; set +a
    python train_stage2_window.py \\
        --dataset-root /root/window_full_v1 \\
        --resume /root/may26_int.pth \\
        --epochs 30 \\
        --lr 4e-5 \\
        --crop 1024,1536 \\
        --batch-size 8 \\
        --output-dir window_fine_tune_full_v1 \\
        --run-name "window_fine_tune_full_v1"
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv as csv_module
import io
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image  # type: ignore
from pytorch_msssim import ssim as compute_ssim

from models.nafnet import NAFNet
from window_dataset import WindowStage2Dataset
from window_loss import MaskedCombinedLoss, _rgb_to_lab
from ema import EMA


DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
CANARY_MAX_LONG = 1600  # long-edge cap for canary inference + triptych


def dashboard_request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{DASHBOARD_URL}{path}",
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "x-worker-token": WORKER_TOKEN,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode() or "{}"
            return json.loads(txt)
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:300]
        raise RuntimeError(f"dashboard {method} {path} -> {e.code}: {msg}")


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
    """Mean ΔE (LAB euclidean) inside the mask, fp32."""
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
    """Instantiate NAFNet using the arch hyperparams stored in the ckpt."""
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
    if ck.get("ema_state_dict"):
        ema_sd = ck["ema_state_dict"]
        sd = ema_sd["shadow"] if isinstance(ema_sd, dict) and "shadow" in ema_sd else ema_sd
    else:
        sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd, strict=False)
    return model, ck


# ----- Canary helpers --------------------------------------------------

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


def crop_to_multiple_of_16(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    return arr[:h16, :w16]


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


@torch.no_grad()
def infer_canary(model, img_rgb: np.ndarray, device) -> tuple[np.ndarray, tuple[int, int]]:
    capped = cap_long_edge(img_rgb, CANARY_MAX_LONG)
    capped = crop_to_multiple_of_16(capped)
    x = torch.from_numpy(capped.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
    y = torch.clamp(y.float(), 0, 1)
    out = (y.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    del x, y
    torch.cuda.empty_cache()
    return out, capped.shape[:2]


def render_triptych(base_rgb, vendor_rgb, fine_rgb, mask, out_path: Path) -> None:
    h, w = fine_rgb.shape[:2]
    base_c = (
        cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_AREA)
        if base_rgb.shape[:2] != (h, w) else base_rgb
    )
    vendor_c = cv2.resize(vendor_rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask_c = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
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


def per_image_metrics(pred_rgb: np.ndarray, tgt_rgb: np.ndarray,
                      mask: np.ndarray, lpips_fn) -> dict:
    """ΔE_global, ΔE_window, ΔE_p95, SSIM, LPIPS, PSNR_Y, lapRatio."""
    p_t = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t_t = torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    lab_p = _rgb_to_lab(p_t)
    lab_t = _rgb_to_lab(t_t)
    diff = ((lab_p - lab_t) ** 2).sum(dim=1, keepdim=True).sqrt().squeeze().numpy()
    de_global = float(diff.mean())
    de_p95 = float(np.percentile(diff, 95))
    de_window = float(diff[mask > 0].mean()) if mask.any() else 0.0
    ssim_v = float(compute_ssim(p_t, t_t, data_range=1.0, size_average=True).item())
    lpips_v = float(lpips_fn(p_t * 2 - 1, t_t * 2 - 1).mean().item())
    # PSNR_Y
    y_p = 0.299 * pred_rgb[..., 0] + 0.587 * pred_rgb[..., 1] + 0.114 * pred_rgb[..., 2]
    y_t = 0.299 * tgt_rgb[..., 0] + 0.587 * tgt_rgb[..., 1] + 0.114 * tgt_rgb[..., 2]
    mse = float(((y_p - y_t) ** 2).mean())
    psnr_y = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-8))
    # Laplacian sharpness ratio
    lap_p = cv2.Laplacian(pred_rgb, cv2.CV_32F).var()
    lap_t = cv2.Laplacian(tgt_rgb, cv2.CV_32F).var()
    lap_ratio = float(lap_p / max(lap_t, 1e-8))
    return {
        "deltaEMean": de_global,
        "deltaEP95": de_p95,
        "deltaEWindow": de_window,
        "ssim": ssim_v,
        "lpips": lpips_v,
        "psnrY": float(psnr_y),
        "lapRatio": lap_ratio,
    }


def rclone_copy_dir(local: Path, dst: str) -> None:
    res = subprocess.run(
        ["rclone", "copy", str(local), dst, "-q", "--s3-no-check-bucket"],
        capture_output=True, text=True, timeout=600,
    )
    if res.returncode != 0:
        print(f"  rclone copy failed: {res.stderr[:200]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--dataset", default="pairs.csv")
    ap.add_argument("--resume", required=True)
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
    ap.add_argument("--run-name", default=None,
                    help="Optional friendly name shown on the dashboard")
    ap.add_argument("--num-canary", type=int, default=8,
                    help="Number of canary triptychs to render per epoch")
    ap.add_argument("--cloud-instance", default=None)
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
          f"(metrics: {ckpt.get('metrics')})  width={ckpt.get('width', 32)}")
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

    # -- Register run with the dashboard --------------------------------
    print("Registering training run with dashboard …")
    config = {
        "crop": args.crop,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "l1_weight": args.l1_weight,
        "lpips_weight": args.lpips_weight,
        "ssim_weight": args.ssim_weight,
        "window_l1_weight": args.window_l1_weight,
        "window_chroma_weight": args.window_chroma_weight,
        "window_chroma_warmup": args.window_chroma_warmup,
        "mask_overlap_frac": args.mask_overlap_frac,
        "base_ckpt": args.resume,
        "base_ckpt_epoch": ckpt.get("epoch"),
        "base_ckpt_metrics": ckpt.get("metrics"),
        "params_m": round(n_params, 2),
    }
    try:
        resp = dashboard_request("POST", "/api/internal/training-runs", {
            "name": args.run_name or run_name,
            "runType": "stage2_window",
            "totalEpochs": args.epochs,
            "trainingPairs": len(train_df),
            "stage1Variant": Path(args.resume).name,
            "config": config,
            "cloudInstance": args.cloud_instance,
        })
        run_id = resp["runId"]
        print(f"  runId={run_id}")
    except Exception as e:
        print(f"  WARN: could not register run on dashboard: {e}")
        run_id = None

    # -- Pick canary set: top N val rows by mask area --------------------
    print("Selecting canary set …")
    canary_rows = []
    for _, row in val_df.iterrows():
        try:
            m = load_mask(dataset_root / row["mask"])
            canary_rows.append({"row": row, "mask_pixels": int(m.sum())})
        except Exception:
            pass
    canary_rows.sort(key=lambda r: r["mask_pixels"], reverse=True)
    canary_rows = canary_rows[: args.num_canary]
    print(f"  picked {len(canary_rows)} canaries (top mask-area val rows)")

    # -- Base outputs once (for triptychs across all epochs) -------------
    base_cache: dict[str, dict] = {}
    print("Caching base outputs for canaries …")
    model.eval()
    for cr in canary_rows:
        row = cr["row"]
        pid = row["pair_id"]
        inp = load_rgb(dataset_root / row["input"])
        out, sh = infer_canary(model, inp, device)
        vendor = load_rgb(dataset_root / row["output"])
        mask = load_mask(dataset_root / row["mask"])
        h, w = sh
        vendor_c = cv2.resize(vendor, (w, h), interpolation=cv2.INTER_AREA)
        mask_c = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        base_cache[pid] = {
            "base_out": out,
            "vendor": vendor_c,
            "mask": mask_c,
            "input": inp,
        }
        print(f"  base canary {pid} cached ({h}x{w})")

    # -- LPIPS for per-canary metrics ------------------------------------
    import lpips as lpips_lib  # type: ignore
    lpips_fn = lpips_lib.LPIPS(net="alex").cpu().eval()

    # -- Baseline ΔE_window via val_loader (mask-biased crops) ----------
    print("\n=== Baseline ΔE_window on val crops ===")
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
    print(f"Baseline ΔE_window (50 val crops): {baseline_de_window:.3f}\n")

    # -- Log ------------------------------------------------------------
    Path("logs").mkdir(exist_ok=True)
    log_csv = f"logs/window_pilot_{run_name}_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv_module.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "ssim", "delta_e_window",
            "delta_e_window_delta_pct", "lr", "time_min"])

    print(f"=== Fine-tune ({args.epochs} epochs · {n_params:.1f}M params · "
          f"baseline ΔE_window={baseline_de_window:.3f}) ===")
    best_de = float("inf")

    # -- Train ----------------------------------------------------------
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

        # -- Validate ---------------------------------------------------
        if ema:
            ema.apply(model)
        model.eval()
        val_losses, ssim_scores, deltas = [], [], []
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

        # -- Render canary triptychs (with EMA still applied) -----------
        canary_dir = ckpt_dir / f"epoch_{epoch:03d}_canary"
        canary_dir.mkdir(parents=True, exist_ok=True)
        canary_metrics = []
        for cr in canary_rows:
            row = cr["row"]
            pid = row["pair_id"]
            cache = base_cache[pid]
            fine_out, _ = infer_canary(model, cache["input"], device)
            # All cached entries are at the same eval-cap resolution.
            h, w = fine_out.shape[:2]
            triptych_path = canary_dir / f"{pid}.jpg"
            render_triptych(
                cache["base_out"], cache["vendor"], fine_out, cache["mask"],
                triptych_path,
            )
            # Also save the raw fine output (smaller, separately addressable)
            raw_path = canary_dir / "raw" / f"{pid}.jpg"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fine_out).save(raw_path, quality=95)
            m = per_image_metrics(fine_out, cache["vendor"], cache["mask"], lpips_fn)
            canary_metrics.append({"pid": pid, **m})

        # Upload canaries for this epoch in one rclone batch
        r2_prefix = f"training-runs/{run_id}/epoch_{epoch:03d}" if run_id else \
                    f"checkpoints/{run_name}/epoch_{epoch:03d}_canary"
        rclone_copy_dir(canary_dir, f"{RCLONE_R2}:{R2_BUCKET}/{r2_prefix}/")

        # POST canary rows + epoch row
        if run_id:
            try:
                for cm in canary_metrics:
                    pid = cm["pid"]
                    dashboard_request("POST", "/api/internal/training-canary", {
                        "runId": run_id,
                        "epoch": epoch,
                        "pairId": pid,
                        "outputR2Path": f"{r2_prefix}/raw/{pid}.jpg",
                        "triptychR2Path": f"{r2_prefix}/{pid}.jpg",
                        "lpips": cm["lpips"],
                        "deltaEMean": cm["deltaEMean"],
                        "deltaEP95": cm["deltaEP95"],
                        "deltaEWindow": cm["deltaEWindow"],
                        "psnrY": cm["psnrY"],
                        "ssim": cm["ssim"],
                        "lapRatio": cm["lapRatio"],
                    })
                dashboard_request("POST", "/api/internal/training-epoch", {
                    "runId": run_id,
                    "epoch": epoch,
                    "trainLossMean": float(np.mean(train_losses)),
                    "valLpips": float(np.mean([cm["lpips"] for cm in canary_metrics])),
                    "valDeltaEMean": float(np.mean([cm["deltaEMean"] for cm in canary_metrics])),
                    "valDeltaEP95": float(np.mean([cm["deltaEP95"] for cm in canary_metrics])),
                    "valDeltaEWindow": de_window,
                    "valSsim": avg_ssim,
                    "valPsnrY": float(np.mean([cm["psnrY"] for cm in canary_metrics])),
                    "valLapRatio": float(np.mean([cm["lapRatio"] for cm in canary_metrics])),
                    "wallTimeSec": (time.time() - ep_start),
                    "checkpointR2Path": f"checkpoints/{run_name}/latest.pth",
                })
            except Exception as e:
                print(f"  WARN: dashboard POST failed: {e}", flush=True)

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
            "arch": "nafnet",
            "width": ckpt.get("width", 32),
            "in_channels": ckpt.get("in_channels", 3),
            "out_channels": ckpt.get("out_channels", 3),
            "middle_blk_num": ckpt.get("middle_blk_num", 12),
            "enc_blk_nums": ckpt.get("enc_blk_nums", [2, 2, 4, 8]),
            "dec_blk_nums": ckpt.get("dec_blk_nums", [2, 2, 2, 2]),
            "use_residual": ckpt.get("use_residual", True),
            "residual_start": ckpt.get("residual_start", 0),
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

        rclone_copy_dir(
            ckpt_dir,
            f"{RCLONE_R2}:{R2_BUCKET}/checkpoints/{run_name}/",
        )

    # -- Mark run completed ---------------------------------------------
    if run_id:
        try:
            dashboard_request("PATCH", f"/api/internal/training-runs/{run_id}", {
                "status": "completed",
                "completedEpochs": args.epochs,
                "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        except Exception as e:
            print(f"  WARN: PATCH completion failed: {e}")

    print(f"\nDone. baseline ΔE_window={baseline_de_window:.3f}  "
          f"best ΔE_window={best_de:.3f}  "
          f"delta={(best_de-baseline_de_window)/baseline_de_window*100:+.1f}%")


if __name__ == "__main__":
    main()
