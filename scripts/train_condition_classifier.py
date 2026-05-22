"""Train a binary condition classifier (per-defect) from ImageReview labels.

Pulls positives + negatives from the production DB, downloads w384 thumbs
from Cloudflare Images, trains a ResNet-18 binary head with light
augmentation, saves a checkpoint to pipeline/classifier_<condition>.pth.

The saved format matches classifier_v2.pth so the inference path in
pipeline.run_pipeline.classify() can load it with --condition just by
swapping the checkpoint path.

Usage:
    python -m scripts.train_condition_classifier --condition shine
    python -m scripts.train_condition_classifier --condition reflection --epochs 15
    python -m scripts.train_condition_classifier --condition all  # train all >=100-label ones

Data definition:
  POSITIVE  = ImageReview where conditions->>'<condition>' = 'true'
  NEGATIVE  = ImageReview with decision='accepted' and the condition NOT flagged
  We balance positives/negatives 1:1 by undersampling negatives.

Env:
    DATABASE_URL  — postgres
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import psycopg2  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torchvision.models as tvmodels  # type: ignore
import torchvision.transforms as T  # type: ignore
from PIL import Image  # type: ignore

CF_HASH = "hfTNewiEIKKL4K9rrhwauQ"
PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
THUMB_CACHE = Path("/tmp/condition-thumbs")
THUMB_CACHE.mkdir(exist_ok=True)

# >= 100 labels — enough to train a usable classifier
TRAINABLE_CONDITIONS = [
    "shine", "sky", "reflection", "glare", "dead-fixture", "color-cast",
]


def log(s: str) -> None:
    print(s, flush=True)


def pull_dataset(condition: str, max_per_class: int = 2000) -> tuple[list[tuple[str, str, int]], dict]:
    """Returns (samples, stats) where samples is [(imageReviewId, cfImageId, label)]
    and label is 1 (positive) or 0 (negative). Balanced 1:1.
    """
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    pos_q = f"""
      SELECT id, "cfImageId" FROM "ImageReview"
      WHERE "cfImageId" IS NOT NULL
        AND conditions IS NOT NULL
        AND conditions->>%s = 'true'
      ORDER BY random()
      LIMIT %s
    """
    cur.execute(pos_q, (condition, max_per_class))
    positives = [(r[0], r[1], 1) for r in cur.fetchall()]

    n_pos = len(positives)
    if n_pos < 20:
        cur.close(); conn.close()
        raise SystemExit(f"too few positives ({n_pos}) for condition '{condition}' — need >=20")

    neg_q = f"""
      SELECT id, "cfImageId" FROM "ImageReview"
      WHERE "cfImageId" IS NOT NULL
        AND decision = 'accepted'
        AND (conditions IS NULL OR conditions->>%s IS DISTINCT FROM 'true')
      ORDER BY random()
      LIMIT %s
    """
    cur.execute(neg_q, (condition, n_pos))
    negatives = [(r[0], r[1], 0) for r in cur.fetchall()]
    cur.close(); conn.close()

    samples = positives + negatives
    random.shuffle(samples)
    return samples, {"n_pos": n_pos, "n_neg": len(negatives)}


def fetch_thumb(cf_id: str) -> Image.Image:
    cache = THUMB_CACHE / f"{cf_id}.jpg"
    if not cache.exists():
        url = f"https://imagedelivery.net/{CF_HASH}/{cf_id}/w384"
        req = urllib.request.Request(url, headers={"User-Agent": "arem-worker/train"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        cache.write_bytes(data)
    return Image.open(cache).convert("RGB")


class ConditionDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rid, cf_id, label = self.samples[idx]
        try:
            img = fetch_thumb(cf_id)
        except Exception:
            # Return a black placeholder; loss for this sample is uninformative.
            img = Image.new("RGB", (384, 384), color=(0, 0, 0))
        x = self.tfm(img)
        return x, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    help="condition key (e.g. shine, reflection) or 'all'")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.condition == "all":
        for c in TRAINABLE_CONDITIONS:
            log(f"\n=== training '{c}' ===")
            run_one(c, args)
        return
    run_one(args.condition, args)


def run_one(condition: str, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    samples, stats = pull_dataset(condition, max_per_class=args.max_per_class)
    log(f"data: {stats}  total={len(samples)}")
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    log(f"train={len(train_samples)} val={len(val_samples)}")

    train_tfm = T.Compose([
        T.Resize((256, 256)),
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ConditionDataset(train_samples, train_tfm)
    val_ds = ConditionDataset(val_samples, val_tfm)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    model = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        # train
        model.train()
        t_loss = 0.0; t_correct = 0; t_total = 0
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (out.argmax(1) == y).sum().item()
            t_total += x.size(0)
        # val
        model.eval()
        v_correct = 0; v_total = 0; tp = 0; fp = 0; fn = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)
                tp += ((pred == 1) & (y == 1)).sum().item()
                fp += ((pred == 1) & (y == 0)).sum().item()
                fn += ((pred == 0) & (y == 1)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        val_acc = v_correct / max(v_total, 1)
        log(f"  epoch {epoch+1:>2}: train_loss={t_loss/t_total:.4f} train_acc={t_correct/t_total:.3f} "
            f"val_acc={val_acc:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
            f"({time.time()-t0:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "labels": ["clean", condition],
                "val_acc": val_acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "stats": stats,
                "condition": condition,
                "epoch": epoch + 1,
                "trained_at": time.time(),
                "version": f"{condition}-v1",
            }

    if best_state is None:
        raise RuntimeError("never improved")

    out_path = PIPELINE_DIR / f"classifier_{condition.replace('-', '_')}.pth"
    torch.save(best_state, out_path)
    log(f"saved {out_path}  best_val_acc={best_val_acc:.3f}")


if __name__ == "__main__":
    main()
