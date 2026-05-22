"""Stage-2 polish (whole-frame) fine-tune.

Fine-tunes the production Stage-2 NAFNet (may26 w=32) to reproduce a target
style from a paired dataset. Designed for the "polish" step that nudges
pipeline output toward a Lightroom-edited reference. Unlike the window
fine-tune this is a global style transform — no masks, no windowed loss
terms — so the trainer collapses to plain CombinedLoss (L1 + LPIPS + SSIM).

Manifest contract (CSV): pair_id, input, output. Paths are relative to
--dataset-root. Use scripts/build_polish_manifest.py to generate one from
a flat <root>/<file>.jpg + <root>/after/<file>.jpg layout.

Usage on 3090:
    cd ~/arem-photo-ai-V2          # losses.py / ema.py / models/nafnet.py live here
    conda activate arem-photo-ai
    set -a; source /etc/arem-worker.env; set +a
    python train_polish.py \\
        --dataset-root /home/jordan/polish_exterior_v1 \\
        --resume /home/jordan/may26_local_data/models/may26_exterior_w32_b4_4gpu_ep29.pth \\
        --epochs 12 --warmup-epochs 2 --lr 2e-5 \\
        --crop 768 --batch-size 4 \\
        --output-dir polish_exterior_v1 \\
        --run-name "polish_exterior_v1_lr2e5"
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv as csv_module
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
from polish_dataset import PolishStage2Dataset
from losses import CombinedLoss
from ema import EMA


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB→LAB (D65). Self-contained — no window_loss dep."""
    r, g, b = rgb.unbind(dim=1)

    def to_linear(c):
        return torch.where(c <= 0.04045, c / 12.92,
                           ((c.clamp_min(1e-9) + 0.055) / 1.055).pow(2.4))

    rl, gl, bl = to_linear(r), to_linear(g), to_linear(b)
    x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047
    y = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) / 1.0
    z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883

    def f(t):
        d = 6.0 / 29.0
        return torch.where(t > d ** 3, t.clamp_min(1e-9).pow(1.0 / 3.0),
                           t / (3 * d * d) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)


DASHBOARD_URL = os.environ.get(
    "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "arem-training-data")
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
CANARY_MAX_LONG = 1600


def dashboard_request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{DASHBOARD_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json", "x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"dashboard {method} {path} -> {e.code}: {e.read().decode()[:300]}")


def split_pairs(df, train_frac=0.9, seed=42):
    df = df.copy().sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * train_frac)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def compute_delta_e_global(pred, target):
    with torch.no_grad():
        lab_p = _rgb_to_lab(pred.float())
        lab_t = _rgb_to_lab(target.float())
        diff = ((lab_p - lab_t) ** 2).sum(dim=1, keepdim=True).sqrt()
        return diff.mean().item()


def build_nafnet_from_ckpt(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "arch" in ck and str(ck["arch"]).lower() != "nafnet":
        raise ValueError(f"refusing to fine-tune from arch={ck['arch']!r}; NAFNet only")
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
    return cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def crop_to_multiple_of_16(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[: (h // 16) * 16, : (w // 16) * 16]


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def infer_canary(model, img_rgb: np.ndarray, device) -> np.ndarray:
    capped = crop_to_multiple_of_16(cap_long_edge(img_rgb, CANARY_MAX_LONG))
    x = torch.from_numpy(capped.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
    y = torch.clamp(y.float(), 0, 1)
    out = (y.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    del x, y
    torch.cuda.empty_cache()
    return out


def render_triptych(input_rgb, target_rgb, fine_rgb, out_path: Path) -> None:
    h, w = fine_rgb.shape[:2]
    inp_c = cv2.resize(input_rgb, (w, h), interpolation=cv2.INTER_AREA)
    tgt_c = cv2.resize(target_rgb, (w, h), interpolation=cv2.INTER_AREA)
    triptych = np.hstack([inp_c, tgt_c, fine_rgb])
    band = np.zeros((36, triptych.shape[1], 3), dtype=np.uint8)
    for i, txt in enumerate(("input (current pipeline)", "target (Lightroom)", "ours (polish-tuned)")):
        cv2.putText(band, txt, (i * w + 16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack([band, triptych])).save(out_path, quality=95)


def per_image_metrics(pred_rgb: np.ndarray, tgt_rgb: np.ndarray, lpips_fn) -> dict:
    p_t = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t_t = torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    lab_p, lab_t = _rgb_to_lab(p_t), _rgb_to_lab(t_t)
    diff = ((lab_p - lab_t) ** 2).sum(dim=1, keepdim=True).sqrt().squeeze().numpy()
    ssim_v = float(compute_ssim(p_t, t_t, data_range=1.0, size_average=True).item())
    lpips_v = float(lpips_fn(p_t * 2 - 1, t_t * 2 - 1).mean().item())
    y_p = 0.299 * pred_rgb[..., 0] + 0.587 * pred_rgb[..., 1] + 0.114 * pred_rgb[..., 2]
    y_t = 0.299 * tgt_rgb[..., 0] + 0.587 * tgt_rgb[..., 1] + 0.114 * tgt_rgb[..., 2]
    psnr_y = 10.0 * np.log10(255.0 ** 2 / max(float(((y_p - y_t) ** 2).mean()), 1e-8))
    lap_p = cv2.Laplacian(pred_rgb, cv2.CV_32F).var()
    lap_t = cv2.Laplacian(tgt_rgb, cv2.CV_32F).var()
    return {
        "deltaEMean": float(diff.mean()),
        "deltaEP95": float(np.percentile(diff, 95)),
        "ssim": ssim_v,
        "lpips": lpips_v,
        "psnrY": float(psnr_y),
        "lapRatio": float(lap_p / max(lap_t, 1e-8)),
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
    ap.add_argument("--crop", default="768")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--l1-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=1.0)
    ap.add_argument("--ssim-weight", type=float, default=0.5)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--output-dir", default="polish_v1")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--num-canary", type=int, default=8)
    ap.add_argument("--cloud-instance", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    crop_parts = [int(x) for x in str(args.crop).split(",")]
    crop_size = crop_parts[0] if len(crop_parts) == 1 else (crop_parts[0], crop_parts[1])

    run_name = args.output_dir
    ckpt_dir = Path(f"checkpoints/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- Model --
    print(f"Loading base NAFNet checkpoint: {args.resume}")
    model, ckpt = build_nafnet_from_ckpt(args.resume, device)
    print(f"  resumed from epoch {ckpt.get('epoch')} "
          f"(metrics: {ckpt.get('metrics')})  width={ckpt.get('width', 32)}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # -- Dataset --
    dataset_root = Path(args.dataset_root)
    pairs_df = pd.read_csv(dataset_root / args.dataset)
    print(f"Manifest: {len(pairs_df)} rows")
    train_df, val_df = split_pairs(pairs_df, train_frac=0.9)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = PolishStage2Dataset(train_df, crop_size=crop_size, augment=True,
                                   path_prefix=str(dataset_root))
    val_ds = PolishStage2Dataset(val_df, crop_size=crop_size, augment=False,
                                 path_prefix=str(dataset_root))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # -- Loss / optim --
    criterion = CombinedLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.lpips_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    warmup_epochs = max(0, args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, decay=0.999) if args.use_ema else None

    # -- Register run with dashboard --
    config = {
        "crop": args.crop, "batch_size": args.batch_size, "epochs": args.epochs,
        "lr": args.lr, "l1_weight": args.l1_weight, "lpips_weight": args.lpips_weight,
        "ssim_weight": args.ssim_weight, "base_ckpt": args.resume,
        "base_ckpt_epoch": ckpt.get("epoch"), "base_ckpt_metrics": ckpt.get("metrics"),
        "params_m": round(n_params, 2),
    }
    run_id = None
    try:
        resp = dashboard_request("POST", "/api/internal/training-runs", {
            "name": args.run_name or run_name,
            "runType": "stage2_polish",
            "totalEpochs": args.epochs,
            "trainingPairs": len(train_df),
            "stage1Variant": Path(args.resume).name,
            "config": config,
            "cloudInstance": args.cloud_instance,
        })
        run_id = resp.get("runId")
        print(f"  runId={run_id}")
    except Exception as e:
        print(f"  WARN: could not register run on dashboard: {e}")

    # -- Canaries: first N val rows --
    canary_rows = val_df.head(args.num_canary).to_dict("records")
    print(f"Canaries: {len(canary_rows)} rows from val set")

    base_cache: dict[str, dict] = {}
    model.eval()
    for row in canary_rows:
        pid = row["pair_id"]
        inp = load_rgb(dataset_root / row["input"])
        out = infer_canary(model, inp, device)
        tgt = load_rgb(dataset_root / row["output"])
        h, w = out.shape[:2]
        tgt_c = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)
        base_cache[pid] = {"base_out": out, "target": tgt_c, "input": inp}
        print(f"  base canary {pid} cached ({h}x{w})")

    import lpips as lpips_lib  # type: ignore
    lpips_fn = lpips_lib.LPIPS(net="alex").cpu().eval()

    # -- Baseline ΔE_global on val crops --
    print("\n=== Baseline ΔE_global on val crops ===")
    base_deltas = []
    with torch.no_grad():
        for vi, (inp, tgt) in enumerate(val_loader):
            if vi >= 50:
                break
            inp = inp.to(device); tgt = tgt.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inp)
            base_deltas.append(compute_delta_e_global(out, tgt))
    baseline_de = float(np.mean(base_deltas))
    print(f"Baseline ΔE_global (50 val crops): {baseline_de:.3f}\n")

    Path("logs").mkdir(exist_ok=True)
    log_csv = f"logs/polish_{run_name}_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv_module.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "ssim", "delta_e_global",
             "delta_e_global_delta_pct", "lr", "time_min"])

    print(f"=== Polish fine-tune ({args.epochs} epochs · {n_params:.1f}M params · "
          f"baseline ΔE_global={baseline_de:.3f}) ===")
    best_de = float("inf")

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warm_lr = args.lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

        model.train()
        train_losses = []
        for step, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(device); tgt = tgt.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(inp)
                ld = criterion(out, tgt)
            scaler.scale(ld["total"]).backward()
            scaler.step(optimizer); scaler.update()
            if ema: ema.update(model)
            train_losses.append(ld["total"].item())
            if (step + 1) % 25 == 0:
                avg = float(np.mean(train_losses[-25:]))
                print(f"  ep{epoch} step {step+1}/{len(train_loader)} loss={avg:.4f}", flush=True)
        if epoch > warmup_epochs:
            scheduler.step()

        # -- Validate --
        if ema: ema.apply(model)
        model.eval()
        val_losses, ssim_scores, deltas = [], [], []
        with torch.no_grad():
            for vi, (inp, tgt) in enumerate(val_loader):
                if vi >= 100:
                    break
                inp = inp.to(device); tgt = tgt.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(inp)
                    ld = criterion(out, tgt)
                val_losses.append(ld["total"].item())
                ssim_scores.append(compute_ssim(
                    out.float(), tgt.float(), data_range=1.0, size_average=True).item())
                deltas.append(compute_delta_e_global(out, tgt))
        avg_val = float(np.mean(val_losses))
        avg_ssim = float(np.mean(ssim_scores))
        de_global = float(np.mean(deltas))
        de_pct = (de_global - baseline_de) / baseline_de * 100.0

        # -- Canary triptychs --
        canary_dir = ckpt_dir / f"epoch_{epoch:03d}_canary"
        canary_dir.mkdir(parents=True, exist_ok=True)
        canary_metrics = []
        for row in canary_rows:
            pid = row["pair_id"]
            cache = base_cache[pid]
            fine_out = infer_canary(model, cache["input"], device)
            triptych_path = canary_dir / f"{pid}.jpg"
            render_triptych(cache["base_out"], cache["target"], fine_out, triptych_path)
            raw_path = canary_dir / "raw" / f"{pid}.jpg"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fine_out).save(raw_path, quality=95)
            m = per_image_metrics(fine_out, cache["target"], lpips_fn)
            canary_metrics.append({"pid": pid, **m})

        r2_prefix = (f"training-runs/{run_id}/epoch_{epoch:03d}" if run_id
                     else f"checkpoints/{run_name}/epoch_{epoch:03d}_canary")
        rclone_copy_dir(canary_dir, f"{RCLONE_R2}:{R2_BUCKET}/{r2_prefix}/")

        if run_id:
            try:
                for cm in canary_metrics:
                    dashboard_request("POST", "/api/internal/training-canary", {
                        "runId": run_id, "epoch": epoch, "pairId": cm["pid"],
                        "outputR2Path": f"{r2_prefix}/raw/{cm['pid']}.jpg",
                        "triptychR2Path": f"{r2_prefix}/{cm['pid']}.jpg",
                        "lpips": cm["lpips"], "deltaEMean": cm["deltaEMean"],
                        "deltaEP95": cm["deltaEP95"], "deltaEWindow": cm["deltaEMean"],
                        "psnrY": cm["psnrY"], "ssim": cm["ssim"], "lapRatio": cm["lapRatio"],
                    })
                dashboard_request("POST", "/api/internal/training-epoch", {
                    "runId": run_id, "epoch": epoch,
                    "trainLossMean": float(np.mean(train_losses)),
                    "valLpips": float(np.mean([cm["lpips"] for cm in canary_metrics])),
                    "valDeltaEMean": float(np.mean([cm["deltaEMean"] for cm in canary_metrics])),
                    "valDeltaEP95": float(np.mean([cm["deltaEP95"] for cm in canary_metrics])),
                    "valDeltaEWindow": de_global,
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
              f"val={avg_val:.4f}  ssim={avg_ssim:.4f}  ΔE={de_global:.3f} "
              f"({de_pct:+.1f}% vs baseline)  lr={lr:.6f}  time={ep_time:.1f}min", flush=True)
        with open(log_csv, "a", newline="") as f:
            csv_module.writer(f).writerow([
                epoch, float(np.mean(train_losses)), avg_val, avg_ssim,
                de_global, de_pct, lr, ep_time])

        save_dict = {
            "epoch": epoch, "arch": "nafnet",
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
            "metrics": {"val_loss": avg_val, "ssim": avg_ssim,
                        "delta_e_global": de_global,
                        "baseline_delta_e_global": baseline_de},
        }
        if ema:
            save_dict["ema_state_dict"] = ema.state_dict()
        torch.save(save_dict, str(ckpt_dir / "latest.pth"))
        if de_global < best_de:
            best_de = de_global
            torch.save(save_dict, str(ckpt_dir / "best_de_global.pth"))
            print(f"  new best ΔE_global: {best_de:.3f}", flush=True)
        if ema:
            ema.restore(model)

        rclone_copy_dir(ckpt_dir, f"{RCLONE_R2}:{R2_BUCKET}/checkpoints/{run_name}/")

    if run_id:
        try:
            dashboard_request("PATCH", f"/api/internal/training-runs/{run_id}", {
                "status": "completed", "completedEpochs": args.epochs,
                "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        except Exception as e:
            print(f"  WARN: PATCH completion failed: {e}")

    print(f"\nDone. baseline ΔE_global={baseline_de:.3f}  best ΔE_global={best_de:.3f}  "
          f"delta={(best_de-baseline_de)/baseline_de*100:+.1f}%")


if __name__ == "__main__":
    main()
