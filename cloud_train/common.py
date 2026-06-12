"""Shared pieces for the 2026-06 native-resolution cloud training runs.

Data layout (pulled from r2:arem-training-data/cloud-train/):
  originals/{aligned,may}/<scene>/ours/<slug>.jpg   (full-res ~4200x2800)
  meta/aligned/{room_manifest.jsonl, corrected_*.jsonl}
  meta/may/{manifest.csv, corrected_*.jsonl}
  detector/{ann_train_native.json, ann_val_native.json, classes.json}

Canvas: NATIVE_HW landscape canvas; aspect-preserving resize + letterbox pad
(most frames are exactly 3:2 landscape -> near-identity; verticals pad sides).
"""
import csv, json, os, random
from collections import defaultdict

import torch
import torch.multiprocessing as _tmp
_tmp.set_sharing_strategy("file_system")  # native-res tensors exhaust /dev/shm with the default strategy
from PIL import Image
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF

Image.MAX_IMAGE_PIXELS = None
DATA = os.environ.get("CLOUD_DATA", "/workspace/data")
NATIVE_HW = (2800, 4200)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED = 17


def journal(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("label") is None:
            out.pop(d["slug"], None)
        else:
            out[d["slug"]] = d["label"]
    return out


def load_labeled(journal_name, drop=("discard", "unsure")):
    """[(relpath, label)] for both datasets; relpath under originals/."""
    mani = {}
    for line in open(f"{DATA}/meta/aligned/room_manifest.jsonl"):
        d = json.loads(line)
        mani[d["slug"]] = d["scene"]
    mani2 = {}
    for d in csv.DictReader(open(f"{DATA}/meta/may/manifest.csv")):
        mani2[d["slug"]] = d["scene"]
    items = []
    for tag, m, jp in (("aligned", mani, f"{DATA}/meta/aligned/{journal_name}"),
                       ("may", mani2, f"{DATA}/meta/may/{journal_name}")):
        for slug, label in journal(jp).items():
            if label in drop:
                continue
            folder = m.get(slug)
            if not folder:
                continue
            rel = f"{tag}/{folder}/ours/{slug}.jpg"
            if os.path.exists(f"{DATA}/originals/{rel}"):
                items.append((rel, label))
    return items


def shoot_split(items, val_frac=0.12, seed=SEED):
    """Same shoot-level split logic as every prior run (seed 17)."""
    by_shoot = defaultdict(list)
    for rel, label in items:
        slug = rel.rsplit("/", 1)[-1][:-4]
        parts = slug.split("__")
        shoot = "__".join(parts[:2]) if len(parts) >= 3 else parts[0]
        by_shoot[shoot].append((rel, label))
    rng = random.Random(seed)
    shoots = sorted(by_shoot)
    rng.shuffle(shoots)
    target = int(len(items) * val_frac)
    val_shoots, n = set(), 0
    for s in shoots:
        if n >= target:
            break
        val_shoots.add(s)
        n += len(by_shoot[s])
    train = [(r, l) for s in shoots if s not in val_shoots for r, l in by_shoot[s]]
    val = [(r, l) for s in val_shoots for r, l in by_shoot[s]]
    return train, val


class Letterbox:
    """Aspect-preserving resize into the fixed landscape canvas + pad."""
    def __init__(self, hw=NATIVE_HW):
        self.h, self.w = hw

    def __call__(self, img):
        w, h = img.size
        s = min(self.w / w, self.h / h)
        nw, nh = round(w * s), round(h * s)
        img = TF.resize(img, [nh, nw], antialias=True)
        pl = (self.w - nw) // 2
        pt = (self.h - nh) // 2
        return TF.pad(img, [pl, pt, self.w - nw - pl, self.h - nh - pt])


def train_tfm(hw=NATIVE_HW, augment=True):
    ops = [Letterbox(hw)]
    if augment:
        ops += [T.RandomHorizontalFlip(), T.TrivialAugmentWide()]
    ops += [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


def val_tfm(hw=NATIVE_HW):
    return T.Compose([Letterbox(hw), T.ToImage(),
                      T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
