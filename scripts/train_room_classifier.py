"""Train room classifier v2 from reviewer-corrected labels.

Phases (idempotent — re-running picks up where it left off):

  1. Pull manifest: query ImageReview for every row with roomTypeCorrected set,
     resolve the path to the image (prefer outputR2Path → vendorImageR2Path).
     Drop classes with too-few samples (sunroom 40 → kept; not_a_room 3 → dropped).
     Merge utility_laundry_stairs (2 rows) into utility.
  2. Stage images: rclone-download to /tmp/room-train/<class>/<imageReviewId>.jpg.
     Parallel (4 workers). Skip files already present locally.
  3. Train: ResNet18 backbone with Places365 pretrained weights, two-head
     architecture (occupancy + room) matching what arem-worker/pipeline/run_pipeline.py
     load_room_classifier expects. Occupancy is a dummy 1-class head because
     reviewers don't label occupancy — pipeline compat only.
  4. Validate: 10% held-out split, per-class precision/recall, confusion matrix.
  5. Save: torch.save({model_state, room_types, occupancies, version, val_metrics}, ckpt)

Run on the 3090 (jordan@10.2.0.15) from the arem-worker repo root:

    DATABASE_URL=... CLASSIFIER_ROOM_OUT=/home/jordan/arem-photo-ai-V2/room_classifier_v2.pth \\
      /home/jordan/miniconda3/envs/arem-photo-ai/bin/python -m scripts.train_room_classifier \\
        --epochs 25 --batch-size 64

Places365 ResNet18 backbone is downloaded automatically on first run from
the CSAIL mirror; cached at ~/.cache/torch/hub/places365_resnet18.pth.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from PIL import Image  # type: ignore
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # type: ignore
from torchvision import models as tvmodels  # type: ignore
from torchvision import transforms  # type: ignore


# ─── Config ─────────────────────────────────────────────────────────────────

DATA_ROOT = Path(os.environ.get("ROOM_TRAIN_ROOT", "/tmp/room-train"))
PLACES365_CACHE = Path.home() / ".cache" / "torch" / "hub" / "places365_resnet18.pth"
PLACES365_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"

# Labels we accept. Class merges + drops applied during manifest assembly.
# Order matters — index → class — and gets saved in the ckpt so inference uses the
# same mapping.
ROOM_CLASSES = [
    "bathroom", "bedroom", "closet", "dining", "foyer", "garage",
    "hallway", "kitchen", "living", "office", "stairs", "sunroom", "utility",
]
DROPPED_CLASSES = {"not_a_room"}
MERGED_CLASSES = {"utility_laundry_stairs": "utility"}
MIN_PER_CLASS = 30  # below this we still train but the class is flagged at val time

# Occupancy is a dummy single-class head — pipeline compat with _RoomTwoHead.
OCCUPANCIES = ["unknown"]

RCLONE = ["rclone", "--config", str(Path.home() / ".config/rclone/rclone.conf")]
VERSION = f"v2_{time.strftime('%Y-%m-%d')}"


# ─── Manifest ───────────────────────────────────────────────────────────────

def fetch_manifest(db_url: str) -> list[dict]:
    """Pull every (jobId, midStem, roomTypeCorrected, path) from the DB.
    Returns rows with both a labelled class and a resolvable image path.
    """
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT ir.id, ir."jobId", ir."midStem", ir."roomTypeCorrected" AS room,
               COALESCE(ir."outputR2Path", ir."vendorImageR2Path") AS path
        FROM "ImageReview" ir
        WHERE ir."roomTypeCorrected" IS NOT NULL
          AND (ir."outputR2Path" IS NOT NULL OR ir."vendorImageR2Path" IS NOT NULL)
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()

    # Drop / merge classes
    cleaned: list[dict] = []
    for r in rows:
        cls = r["room"]
        if cls in DROPPED_CLASSES:
            continue
        cls = MERGED_CLASSES.get(cls, cls)
        if cls not in ROOM_CLASSES:
            continue
        r["class"] = cls
        cleaned.append(r)
    return cleaned


# ─── Image staging (rclone) ─────────────────────────────────────────────────

def resolve_remote(path: str) -> str:
    p = path.lstrip("/")
    if p.startswith("AREM (Spiro Uploads)"):
        return f"dropbox:{p}"
    return f"r2:arem-training-data/{p}"


def stage_image(row: dict) -> tuple[str, Path | None]:
    """Download one image to DATA_ROOT/<class>/<id>.jpg. Idempotent: existing
    non-empty files are kept as-is."""
    dst = DATA_ROOT / row["class"] / f"{row['id']}.jpg"
    if dst.exists() and dst.stat().st_size > 1000:
        return (row["id"], dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    remote = resolve_remote(row["path"])
    for attempt in (1, 2):
        try:
            r = subprocess.run(
                [*RCLONE, "copyto", remote, str(dst)],
                capture_output=True, timeout=60,
            )
            if r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
                return (row["id"], dst)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            return (row["id"], None)
    return (row["id"], None)


def stage_all(rows: list[dict], workers: int = 4) -> dict[str, Path]:
    print(f"Staging {len(rows)} images to {DATA_ROOT} ({workers} parallel)…", flush=True)
    local: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(stage_image, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rid, p = f.result()
                if p is not None:
                    local[rid] = p
            except Exception:
                pass
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} staged (kept {len(local)})", flush=True)
    return local


# ─── Model ──────────────────────────────────────────────────────────────────

class RoomTwoHead(nn.Module):
    """Matches arem-worker/pipeline/run_pipeline.py _RoomTwoHead. Keep field
    names + forward signature byte-identical so the live worker's loader
    works against this ckpt without code changes."""
    def __init__(self, n_rooms: int, n_occ: int = 1):
        super().__init__()
        self.backbone = tvmodels.resnet18(weights=None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.occ_head = nn.Linear(feat, n_occ)
        self.room_head = nn.Linear(feat, n_rooms)

    def forward(self, x):
        f = self.backbone(x)
        return self.occ_head(f), self.room_head(f)


def load_places365_into(model: RoomTwoHead) -> None:
    """Download Places365 ResNet18 weights (if not cached) and load the
    backbone. Drops Places365's 365-class fc head — we replace it with our
    13-class room head. Does NOT load the (non-existent) occupancy head.
    """
    if not PLACES365_CACHE.exists():
        PLACES365_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Places365 weights → {PLACES365_CACHE} …", flush=True)
        urllib.request.urlretrieve(PLACES365_URL, str(PLACES365_CACHE))
    state = torch.load(str(PLACES365_CACHE), map_location="cpu", weights_only=False)
    # The CSAIL release stores state under "state_dict" with "module.*" prefixes.
    sd = state.get("state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # Drop the final fc layer (365-way) — backbone.fc was replaced by Identity.
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    print(f"Places365 loaded. missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)


# ─── Dataset ────────────────────────────────────────────────────────────────

TRAIN_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class RoomDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        # Robust to truncated / unusual JPGs — fall back to a zero tensor
        # rather than crashing the whole epoch.
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224))
        return self.tfm(im), label


def split_train_val(rows: list[dict], local: dict[str, Path], seed: int = 0,
                    val_frac: float = 0.1):
    """Stratified split: val_frac of each class goes to validation."""
    by_class: dict[str, list[tuple[Path, int]]] = {c: [] for c in ROOM_CLASSES}
    for r in rows:
        p = local.get(r["id"])
        if p is None:
            continue
        by_class[r["class"]].append((p, ROOM_CLASSES.index(r["class"])))
    rng = random.Random(seed)
    train, val = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


# ─── Train + validate ───────────────────────────────────────────────────────

def train(args, device, train_items, val_items):
    model = RoomTwoHead(n_rooms=len(ROOM_CLASSES), n_occ=len(OCCUPANCIES))
    if not args.no_pretrained:
        load_places365_into(model)
    model.to(device)

    # Class-balanced sampling: weights inverse to class frequency.
    labels = [lbl for _, lbl in train_items]
    counts = Counter(labels)
    weights = np.array([1.0 / counts[lbl] for lbl in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(labels),
                                    replacement=True)

    train_ds = RoomDataset(train_items, TRAIN_TFM)
    val_ds = RoomDataset(val_items, VAL_TFM)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    room_loss = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _occ, room_logits = model(x)
            loss = room_loss(room_logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
        sched.step()
        # Val
        model.eval()
        ok = 0
        n = 0
        per_class_ok = Counter()
        per_class_n = Counter()
        confusion = np.zeros((len(ROOM_CLASSES), len(ROOM_CLASSES)), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _occ, logits = model(x)
                pred = logits.argmax(dim=1)
                ok += int((pred == y).sum().item())
                n += int(y.numel())
                for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
                    per_class_n[t] += 1
                    if p == t:
                        per_class_ok[t] += 1
                    confusion[t, p] += 1
        acc = ok / max(1, n)
        per_class = {ROOM_CLASSES[c]: per_class_ok[c] / max(1, per_class_n[c])
                     for c in range(len(ROOM_CLASSES))}
        print(f"  ep{ep:02d}  loss={running/len(train_loader):.4f}  "
              f"val_acc={acc:.4f}  "
              f"t={time.time()-t0:.1f}s", flush=True)
        # Per-class summary every 5 epochs + at end
        if ep % 5 == 0 or ep == args.epochs:
            print("    per-class accuracy: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in per_class.items()), flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_acc, confusion


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4, help="parallel rclone copies")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip Places365 download; train from random init (debug)")
    ap.add_argument("--skip-stage", action="store_true",
                    help="skip rclone staging (assume images are already in /tmp/room-train)")
    args = ap.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 2
    out_path = Path(
        os.environ.get("CLASSIFIER_ROOM_OUT",
                       "/home/jordan/arem-photo-ai-V2/room_classifier_v2.pth")
    )

    # Phase 1 — manifest
    rows = fetch_manifest(db_url)
    print(f"\nmanifest: {len(rows)} labelled images")
    by_class = Counter(r["class"] for r in rows)
    for c in ROOM_CLASSES:
        print(f"  {c:12s} {by_class.get(c, 0)}")

    # Phase 2 — stage
    if args.skip_stage:
        local: dict[str, Path] = {
            r["id"]: DATA_ROOT / r["class"] / f"{r['id']}.jpg"
            for r in rows
            if (DATA_ROOT / r["class"] / f"{r['id']}.jpg").is_file()
        }
        print(f"skip-stage: {len(local)} files already present in {DATA_ROOT}")
    else:
        local = stage_all(rows, workers=args.workers)

    if not local:
        print("ERROR: no images staged", file=sys.stderr)
        return 3

    # Phase 3 — split + train
    train_items, val_items = split_train_val(rows, local)
    print(f"\nsplit: train={len(train_items)} val={len(val_items)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    best_state, best_acc, confusion = train(args, device, train_items, val_items)

    # Phase 4 — save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "room_types": ROOM_CLASSES,
        "occupancies": OCCUPANCIES,
        "version": VERSION,
        "val_acc": round(best_acc, 4),
        "confusion": confusion.tolist(),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "places365_pretrained": not args.no_pretrained,
    }, str(out_path))
    print(f"\nSAVED {out_path}  best_val_acc={best_acc:.4f}")

    # Phase 5 — print confusion table compactly
    print("\nConfusion (rows=true, cols=pred):")
    header = "        " + " ".join(f"{c[:5]:>6s}" for c in ROOM_CLASSES)
    print(header)
    for i, c in enumerate(ROOM_CLASSES):
        print(f"  {c[:6]:6s} " + " ".join(f"{v:>6d}" for v in confusion[i].tolist()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
