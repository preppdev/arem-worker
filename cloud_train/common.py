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


# ---- GPU-decode path (B200): workers ship raw JPEG bytes; decode + letterbox
# + augment + normalize happen on-device. Fixes the 31%-GPU-util input
# bottleneck of CPU-decoding 12MP JPEGs (and removes the 141MB/sample IPC).
from torchvision.io import decode_jpeg


class BytesDS(torch.utils.data.Dataset):
    def __init__(self, rows, cls_idx, data_root):
        self.rows, self.cls_idx, self.root = rows, cls_idx, data_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, label = self.rows[i][0], self.rows[i][1]
        with open(f"{self.root}/originals/{rel}", "rb") as f:
            data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
        return data, self.cls_idx[label]


def bytes_collate(batch):
    return [b[0] for b in batch], torch.tensor([b[1] for b in batch])


class GpuPrep:
    """JPEG bytes -> normalized float batch on CUDA, letterboxed to hw."""

    def __init__(self, hw=NATIVE_HW, train=False):
        self.h, self.w = hw
        self.train = train
        self.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        self.aug = T.TrivialAugmentWide() if train else None

    def __call__(self, byte_list):
        import torchvision.transforms.v2.functional as F2
        out = torch.zeros(len(byte_list), 3, self.h, self.w, dtype=torch.uint8, device="cuda")
        for i, data in enumerate(byte_list):
            img = decode_jpeg(data, device="cuda")  # uint8 CHW on GPU
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            c, h, w = img.shape
            s = min(self.w / w, self.h / h)
            nh, nw = round(h * s), round(w * s)
            img = F2.resize(img, [nh, nw], antialias=True)
            if self.train and torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            if self.aug is not None:
                img = self.aug(img)
            top, left = (self.h - nh) // 2, (self.w - nw) // 2
            out[i, :, top:top + nh, left:left + nw] = img
        x = out.float().div_(255.0).sub_(self.mean).div_(self.std)
        return x
