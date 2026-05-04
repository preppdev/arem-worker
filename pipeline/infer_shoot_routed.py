"""End-to-end Stage 2 inference with interior/exterior routing.

Pipeline per bracket triplet:
  1. rawpy demosaic (auto-WB, 16-bit sRGB) → 3 TIFFs
  2. Photomatix CLI merge with the production preset → input_autowb.jpg
  3. Classify merged JPG with ResNet-18 (classifier_v2.pth)
  4. Route to matching Stage 2 Restormer (interior or exterior) → pred.jpg

Usage:
    python infer_shoot_routed.py \
        --raw-dir /home/jordan/Desktop/echea \
        --interior-checkpoint ~/checkpoints_remote/interior_full_v1_latest.pth \
        --exterior-checkpoint ~/checkpoints_remote/exterior_full_v1_latest.pth \
        --classifier ~/arem-photo-ai-V2/classifier_v2.pth \
        --output-dir ~/inference_out/echea_routed \
        --preset Interior2
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rawpy
import tifffile
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.restormer import Restormer
from prepare_dataset import find_bracket_triplets

PMTX = os.environ.get("PMTX_STATIC",
    str(Path.home() / "photomatix/PhotomatixCL/PhotomatixCL-static"))


class RestormerStage2(Restormer):
    def forward(self, x):
        residual_input = x
        inp = self.patch_embed(x)
        e1 = self.encoder_1(inp)
        e2 = self.encoder_2(self.down_1(e1))
        e3 = self.encoder_3(self.down_2(e2))
        b = self.bottleneck(self.down_3(e3))
        d3 = self.decoder_3(self.reduce_3(torch.cat([self.up_3(b), e3], dim=1)))
        d2 = self.decoder_2(self.reduce_2(torch.cat([self.up_2(d3), e2], dim=1)))
        d1 = self.decoder_1(self.reduce_1(torch.cat([self.up_1(d2), e1], dim=1)))
        out = self.output(self.refinement(d1))
        return torch.clamp(residual_input + out, 0, 1)


def strip_module_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def rawpy_to_tiff(arw: Path, tiff: Path) -> None:
    with rawpy.imread(str(arw)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False, use_auto_wb=True,
            no_auto_bright=True, output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
        )
    tifffile.imwrite(str(tiff), rgb, photometric="rgb")


def photomatix_merge(tiffs: list[Path], out: Path, preset: str) -> None:
    workdir = out.parent
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.glob("Set003*.jpg"):
        old.unlink(missing_ok=True)
    r = subprocess.run(
        [PMTX, "-x", preset, "-d", str(workdir) + "/", "-s", "jpg",
         *[str(t) for t in tiffs]],
        capture_output=True, text=True, cwd=str(workdir), timeout=180,
    )
    produced = list(workdir.glob("Set003*.jpg"))
    if r.returncode != 0 or not produced:
        raise RuntimeError(
            f"photomatix rc={r.returncode} stderr={r.stderr[-200:]}")
    if produced[0] != out:
        produced[0].rename(out)


def load_stage2(ckpt_path: Path, dim: int, use_ema: bool, device):
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = RestormerStage2(
        in_channels=3, out_channels=3,
        dim=dim, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        use_residual=False,
    ).to(device)
    if use_ema and ck.get("ema_state_dict"):
        ema = ck["ema_state_dict"]
        sd = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
        sd = strip_module_prefix(sd)
        model_keys = set(model.state_dict())
        sd = {k: v for k, v in sd.items() if k in model_keys}
        model.load_state_dict(sd, strict=False)
        print(f"  loaded EMA ({len(sd)} keys), epoch={ck.get('epoch', '?')}")
    else:
        model.load_state_dict(strip_module_prefix(ck["model_state_dict"]))
        print(f"  loaded raw weights, epoch={ck.get('epoch', '?')}")
    model.eval()
    return model


def load_classifier(ckpt_path: Path, device):
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    label_names = state["label_names"]
    num_classes = state["num_classes"]
    print(f"Classifier labels: {label_names}  val_acc: {state['val_acc']:.2f}",
          flush=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, label_names, tfm


def classify(clf_model, tfm, label_names, jpg_path: Path, device) -> tuple[bool, float]:
    """Returns (is_interior, confidence_for_predicted_class)."""
    im = Image.open(jpg_path).convert("RGB")
    x = tfm(im).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf_model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(prob.argmax())
    name = label_names[idx]
    is_interior = name == "interior"
    return is_interior, float(prob[idx])


def infer(model, jpg_path: Path, out_path: Path, device,
          quality: int = 92, tile: int = 1024, overlap: int = 64) -> tuple[int, int]:
    img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    h16, w16 = (h // 16) * 16, (w // 16) * 16
    img_rgb = img_rgb[:h16, :w16]

    # Single-pass cap. With the int32-safe depthwise wrapper in
    # models/restormer.py, the 32-bit indexing limit is gone; the practical
    # cap is now VRAM. On RTX 3090 (24 GB) the model handles ~8 MP single-pass
    # in fp16. Going higher requires a bigger card (H100 80 GB fits 12+ MP).
    fits_single = (h16 * w16 <= 8 * 1024 * 1024)
    if fits_single:
        x = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model(x)
        y = torch.clamp(y, 0, 1)
        pred = (y.squeeze(0).float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(pred).save(out_path, quality=quality)
        torch.cuda.empty_cache()
        return h16, w16

    # Tiled with flat-top feather blending in overlap zones.
    out = np.zeros((h16, w16, 3), dtype=np.float32)
    weight = np.zeros((h16, w16), dtype=np.float32)
    step = tile - overlap
    feather = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
        feather[:overlap] = ramp
        feather[-overlap:] = ramp[::-1]
    win_2d = (feather[:, None] * feather[None, :]).astype(np.float32)
    y_starts = list(range(0, max(h16 - tile, 0) + 1, step))
    if not y_starts or y_starts[-1] != h16 - tile:
        y_starts.append(max(h16 - tile, 0))
    x_starts = list(range(0, max(w16 - tile, 0) + 1, step))
    if not x_starts or x_starts[-1] != w16 - tile:
        x_starts.append(max(w16 - tile, 0))
    for y0 in y_starts:
        for x0 in x_starts:
            patch = img_rgb[y0:y0 + tile, x0:x0 + tile]
            xt = torch.from_numpy(patch.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
            xt = xt.to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yt = model(xt)
            yt = torch.clamp(yt, 0, 1)
            pred = yt.squeeze(0).float().cpu().permute(1, 2, 0).numpy()
            out[y0:y0 + tile, x0:x0 + tile] += pred * win_2d[:, :, None]
            weight[y0:y0 + tile, x0:x0 + tile] += win_2d
            torch.cuda.empty_cache()
    out /= np.maximum(weight[:, :, None], 1e-8)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(out_path, quality=quality)
    return h16, w16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--interior-checkpoint", type=Path, required=True)
    ap.add_argument("--exterior-checkpoint", type=Path, required=True)
    ap.add_argument("--classifier", type=Path,
                    default=Path(__file__).parent / "classifier_v2.pth")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--preset", default="Interior2")
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    ap.add_argument("--keep-tiffs", action="store_true")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-megapixels", type=float, default=0.0,
                    help="If >0, downscale merged JPG to this MP before classify+infer "
                         "(reduces tile count / seam visibility). e.g. 12 for ≤12 MP.")
    ap.add_argument("--tile", type=int, default=1024,
                    help="Tile size for tiled inference (default 1024)")
    ap.add_argument("--overlap", type=int, default=64,
                    help="Tile overlap in pixels (default 64; 256 for big tiles)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiff_dir = args.output_dir / "_tiffs"
    merged_dir = args.output_dir / "input_autowb"
    pred_dir = args.output_dir / "predictions"
    tiff_dir.mkdir(exist_ok=True)
    merged_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)

    triplets = find_bracket_triplets(args.raw_dir)
    if args.limit:
        triplets = triplets[: args.limit]
    print(f"Found {len(triplets)} triplets", flush=True)
    if not triplets:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading interior model...", flush=True)
    int_model = load_stage2(args.interior_checkpoint, args.dim, args.use_ema, device)
    print("Loading exterior model...", flush=True)
    ext_model = load_stage2(args.exterior_checkpoint, args.dim, args.use_ema, device)
    print("Loading classifier...", flush=True)
    clf_model, label_names, clf_tfm = load_classifier(args.classifier, device)

    routing = {"interior": 0, "exterior": 0}
    t0 = time.time()
    for i, (under, mid, over) in enumerate(triplets, 1):
        stem = mid.stem
        merged = merged_dir / f"{stem}_input_autowb.jpg"
        # Note: pred filename always contains route in suffix (-int / -ext)
        pred_int = pred_dir / f"{stem}_pred-int.jpg"
        pred_ext = pred_dir / f"{stem}_pred-ext.jpg"
        if pred_int.exists() or pred_ext.exists():
            print(f"  [{i}/{len(triplets)}] {stem}: skip", flush=True)
            continue

        t_start = time.time()
        tiffs = []
        for arw in (under, mid, over):
            tiff = tiff_dir / f"{arw.stem}.tif"
            if not tiff.exists():
                rawpy_to_tiff(arw, tiff)
            tiffs.append(tiff)
        t_demos = time.time()

        if not merged.exists():
            try:
                photomatix_merge(tiffs, merged, args.preset)
            except Exception as e:
                print(f"  [{i}/{len(triplets)}] {stem}: merge fail: {e}",
                      flush=True)
                continue
        t_merge = time.time()

        is_interior, conf = classify(clf_model, clf_tfm, label_names, merged, device)
        t_clf = time.time()

        # Optional downscale before inference (reduces tile count / seams).
        infer_input = merged
        if args.max_megapixels > 0:
            img = cv2.imread(str(merged), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            mp = (h * w) / 1_000_000
            if mp > args.max_megapixels:
                scale = (args.max_megapixels / mp) ** 0.5
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_ds = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                infer_input = merged_dir / f"{stem}_input_autowb_ds.jpg"
                cv2.imwrite(str(infer_input), img_ds, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

        model = int_model if is_interior else ext_model
        suffix = "int" if is_interior else "ext"
        out_path = pred_dir / f"{stem}_pred-{suffix}.jpg"
        h16, w16 = infer(model, infer_input, out_path, device, args.quality,
                         tile=args.tile, overlap=args.overlap)
        t_infer = time.time()

        routing["interior" if is_interior else "exterior"] += 1

        if not args.keep_tiffs:
            for t in tiffs:
                t.unlink(missing_ok=True)

        print(f"  [{i}/{len(triplets)}] {stem} ({h16}x{w16}) -> {suffix} "
              f"(conf={conf:.2f}): "
              f"demos={t_demos-t_start:.1f}s "
              f"merge={t_merge-t_demos:.1f}s "
              f"clf={t_clf-t_merge:.2f}s "
              f"infer={t_infer-t_clf:.1f}s",
              flush=True)

    if not args.keep_tiffs and tiff_dir.exists():
        try:
            shutil.rmtree(tiff_dir)
        except OSError:
            pass

    print(f"\ndone. {len(triplets)} triplets in {time.time()-t0:.1f}s",
          flush=True)
    print(f"  routing: interior={routing['interior']} exterior={routing['exterior']}",
          flush=True)
    print(f"  predictions: {pred_dir}", flush=True)


if __name__ == "__main__":
    main()
