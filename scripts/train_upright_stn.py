"""Train an image-domain upright-correction model on synthetic rotations
of vendor (correctly-oriented) images.

Why this framing — see the upright-correction status conversation. The
prior scalar-regression model couldn't beat current Hough because (a) it
was trained on post-Hough images where the rotation signal is already
spent and (b) a scalar L1 on θ provides thin gradient. This script
takes a different route:

  - SOURCE OF TRUTH: vendor's 05-Finished-Photos JPGs are correctly
    oriented (they're what we measured everything else against). We
    have ~2900 of these locally on the 3090.
  - SYNTHETIC PAIRS: at every training step, sample θ ~ N(0, 0.8°),
    clipped to ±2.5°. Apply rotation to the vendor image → that's
    the network's input. The target is the original (unrotated) image.
  - LOSS: pixel L1 between the model's de-rotated output and the
    target. Gradient flows through differentiable grid_sample, so the
    backbone learns "predict an angle that, when applied as rotation,
    minimizes pixel error vs the unrotated target."
  - INTEGRATION: at inference we throw away the rotated-image output
    and just keep the predicted angle (scalar). The angle drops into
    the existing pipeline as a swap for Hough's weighted-mean
    estimate. So in production we get the *signal* from the dense
    pixel-domain training without paying the per-pixel inference cost.

Why this might beat scalar regression even though both ultimately
predict an angle:

  - Dense gradient. dL/dθ flows through every pixel, weighted by the
    local vertical gradient magnitude. Images with lots of vertical
    architecture (door frames, wall corners) contribute strong signal;
    flat areas contribute none. The model focuses on what matters.
  - Synthetic ground truth. We *know* the rotation exactly; no SIFT
    measurement noise. Clean labels = lower-variance training.

Run on the 3090:
    DATABASE_URL=... \\
      /home/jordan/miniconda3/envs/arem-photo-ai/bin/python \\
      -m scripts.train_upright_stn --epochs 30 --batch-size 32

Output: /home/jordan/arem-photo-ai-V2/upright_stn_v1.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore
from torchvision import models as tvmodels  # type: ignore
from torchvision import transforms  # type: ignore

DATA_ROOT = Path(os.environ.get("UPRIGHT_TRAIN_ROOT", "/home/jordan/upright-train"))
VENDOR_DIR = DATA_ROOT / "vendor"
OUT_CKPT = Path(os.environ.get(
    "UPRIGHT_STN_OUT",
    "/home/jordan/arem-photo-ai-V2/upright_stn_v1.pth",
))
VERSION = f"v1_{time.strftime('%Y-%m-%d')}"
INPUT_SIZE = 384  # square. Trade-off: smaller = faster, less detail; 384 is enough for the verticals we care about.
# Rotation distribution: empirical observation said mean residual ~0.5°,
# p95 ~1.5°. We train slightly wider so the model handles tail cases
# without extrapolating beyond its training distribution.
ROTATION_STD_DEG = 0.8
ROTATION_CLIP_DEG = 2.5


# ─── Dataset ────────────────────────────────────────────────────────

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# We resize → center-crop the vendor image to INPUT_SIZE², then at
# training time apply a small rotation. The CENTER-CROP is the key
# trick: rotating a SQUARE crop by a tiny angle leaves no black corners
# because the original square is inscribed in a slightly larger square
# that contains all rotated pixels. We add a tiny "rotation margin" by
# letting the source crop be slightly larger than INPUT_SIZE so even
# the rotated version has no border artifacts.
ROT_MARGIN_PX = 24  # source is 384+24=408 square, rotation up to 2.5° → no border


class RotationDataset(Dataset):
    """Each item is one vendor JPG. __getitem__ returns
    (rotated_input_tensor, unrotated_target_tensor, applied_theta_deg)
    where theta is freshly sampled per call. So every epoch sees the
    same images with different rotations — built-in augmentation."""

    def __init__(self, files: list[Path], augment: bool):
        self.files = files
        self.augment = augment
        # Pre-build the transform that operates on the source square
        # BEFORE rotation. We do the rotation ourselves so we have the
        # angle as an explicit label.
        self.resize_crop = transforms.Compose([
            transforms.Resize(INPUT_SIZE + ROT_MARGIN_PX),
            transforms.CenterCrop(INPUT_SIZE + ROT_MARGIN_PX),
        ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            im = Image.new("RGB", (INPUT_SIZE + ROT_MARGIN_PX, INPUT_SIZE + ROT_MARGIN_PX))
        im = self.resize_crop(im)
        # Apply small rotation (or 0 for val determinism).
        if self.augment:
            theta = float(np.clip(np.random.normal(0, ROTATION_STD_DEG),
                                  -ROTATION_CLIP_DEG, ROTATION_CLIP_DEG))
        else:
            # Deterministic val rotations seeded by idx so the val set
            # is reproducible across epochs.
            rng = np.random.default_rng(idx)
            theta = float(np.clip(rng.normal(0, ROTATION_STD_DEG),
                                  -ROTATION_CLIP_DEG, ROTATION_CLIP_DEG))
        # Source target = center crop of im at INPUT_SIZE (the "correct" output)
        target = transforms.functional.center_crop(im, INPUT_SIZE)
        # Rotated input = rotate the larger im by theta, then center crop
        rotated_im = transforms.functional.rotate(
            im, theta, interpolation=transforms.InterpolationMode.BILINEAR,
            expand=False, fill=0,
        )
        rotated_im = transforms.functional.center_crop(rotated_im, INPUT_SIZE)
        return self.to_tensor(rotated_im), self.to_tensor(target), torch.tensor([theta], dtype=torch.float32)


# ─── Model: ResNet18 backbone → 1-d angle head ──────────────────────

class UprightSTN(nn.Module):
    """Predicts a single rotation angle (degrees). At training time we
    apply this rotation via grid_sample to compute the image-domain
    loss; at inference the angle is the only thing the consumer needs."""

    def __init__(self):
        super().__init__()
        backbone = tvmodels.resnet18(weights="DEFAULT")
        feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        # Bias the head toward predicting near-zero (most images are close
        # to upright) so early training doesn't oscillate wildly.
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.zero_()

    def forward(self, x):
        return self.head(self.backbone(x))


def rotate_via_grid_sample(img: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
    """Differentiable rotation via affine grid + grid_sample. img is
    (B, C, H, W); angle_deg is (B, 1) in degrees (positive = CCW).
    Returns the rotated batch at the same resolution."""
    B, C, H, W = img.shape
    theta = angle_deg.squeeze(-1) * math.pi / 180.0
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    # Affine matrix for grid_sample. NOTE: PyTorch's grid_sample uses
    # *inverse* mapping (output → source), so to ROTATE the image by +θ
    # we feed -θ to the affine. We want the model's predicted angle to
    # match the convention "how much do I rotate the input to align with
    # target," so the model predicts +θ and we apply -θ here.
    affine = torch.stack([
        torch.stack([ cos_t, sin_t, torch.zeros_like(cos_t)], dim=1),
        torch.stack([-sin_t, cos_t, torch.zeros_like(cos_t)], dim=1),
    ], dim=1)  # (B, 2, 3)
    grid = F.affine_grid(affine, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ─── Train ──────────────────────────────────────────────────────────

def split(files: list[Path], val_frac: float = 0.1, seed: int = 0):
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    return files[n_val:], files[:n_val]


def train_loop(args, device, train_files, val_files):
    model = UprightSTN().to(device)
    train_ds = RotationDataset(train_files, augment=True)
    val_ds = RotationDataset(val_files, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val_mae = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_pixel_loss = 0.0
        train_angle_mae = 0.0
        n_batches = 0
        for rotated, target, theta in train_loader:
            rotated = rotated.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)
            # Model predicts the angle to UNDO the applied rotation.
            # If input was rotated by +theta, model should output -theta.
            pred = model(rotated)
            corrected = rotate_via_grid_sample(rotated, pred)
            pixel_loss = F.l1_loss(corrected, target)
            optim.zero_grad()
            pixel_loss.backward()
            optim.step()
            train_pixel_loss += float(pixel_loss.item())
            with torch.no_grad():
                # Angle target is -theta (the inverse rotation).
                train_angle_mae += float(torch.abs(pred - (-theta)).mean().item())
            n_batches += 1
        sched.step()

        # Val
        model.eval()
        val_pixel = 0.0
        val_angle_errs: list[float] = []
        n_val_b = 0
        with torch.no_grad():
            for rotated, target, theta in val_loader:
                rotated = rotated.to(device); target = target.to(device); theta = theta.to(device)
                pred = model(rotated)
                corrected = rotate_via_grid_sample(rotated, pred)
                val_pixel += float(F.l1_loss(corrected, target).item())
                # angle error
                val_angle_errs.extend(torch.abs(pred - (-theta)).cpu().flatten().tolist())
                n_val_b += 1
        val_mae = float(np.mean(val_angle_errs))
        val_med = float(np.median(val_angle_errs))
        val_p95 = float(np.percentile(val_angle_errs, 95))
        train_pixel_avg = train_pixel_loss / max(1, n_batches)
        val_pixel_avg = val_pixel / max(1, n_val_b)
        print(f"  ep{ep:02d}  px_train={train_pixel_avg:.4f}  px_val={val_pixel_avg:.4f}  "
              f"ang_train_mae={train_angle_mae/max(1,n_batches):.3f}°  "
              f"val_mae={val_mae:.3f}°  val_med={val_med:.3f}°  val_p95={val_p95:.3f}°  "
              f"t={time.time()-t0:.1f}s", flush=True)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_val_mae


# ─── Main ───────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    files = sorted(p for p in VENDOR_DIR.glob("*.jpg")
                   if p.stat().st_size > 5000)
    print(f"vendor images: {len(files)}")
    if len(files) < 100:
        print(f"ERROR: only {len(files)} vendor images at {VENDOR_DIR}. "
              "Run train_upright_regressor.py first (without --skip-stage) to populate.",
              file=sys.stderr)
        return 2

    train_files, val_files = split(files, val_frac=0.1, seed=0)
    print(f"split: train={len(train_files)} val={len(val_files)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"input size: {INPUT_SIZE}x{INPUT_SIZE}  rotation σ={ROTATION_STD_DEG}° clip ±{ROTATION_CLIP_DEG}°")

    best_state, best_val_mae = train_loop(args, device, train_files, val_files)

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "arch": "upright_stn_resnet18_v1",
        "version": VERSION,
        "input_size": INPUT_SIZE,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "rotation_std_deg": ROTATION_STD_DEG,
        "rotation_clip_deg": ROTATION_CLIP_DEG,
        "val_mae_deg": round(best_val_mae, 4),
    }, str(OUT_CKPT))
    print(f"\nSAVED {OUT_CKPT}  best_val_mae={best_val_mae:.3f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
