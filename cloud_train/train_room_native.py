#!/usr/bin/env python3
"""Room classifier v4 — ConvNeXt-Base, NATIVE full-frame (2800x4200 letterbox),
B200. Full recipe: EMA, TrivialAugment, mixup, tail-class triple-dup,
sqrt-inverse class weights, label smoothing, hflip TTA, shoot-split seed 17."""
import copy, json, os, random, time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from common import DATA, NATIVE_HW, load_labeled, shoot_split, train_tfm, val_tfm

OUT = os.environ.get("OUT", "/workspace/runs/room_native_b")
EPOCHS = int(os.environ.get("EPOCHS", "16"))
BATCH = int(os.environ.get("BATCH", "4"))
ACCUM = int(os.environ.get("ACCUM", "6"))
LR = float(os.environ.get("LR", "3e-4"))
WORKERS = int(os.environ.get("WORKERS", "24"))
DUP3_BELOW, DUP2_BELOW = 350, 800
EMA_DECAY, MIXUP_ALPHA, MIXUP_P = 0.999, 0.2, 0.5
os.makedirs(OUT, exist_ok=True)
random.seed(17); torch.manual_seed(17)

items = load_labeled("corrected_room.jsonl")
classes = sorted({l for _, l in items})
cls_idx = {c: i for i, c in enumerate(classes)}
train_items, val_items = shoot_split(items)
counts = Counter(l for _, l in train_items)
dups = []
for rel, l in train_items:
    if counts[l] < DUP3_BELOW: dups += [(rel, l, True), (rel, l, False)]
    elif counts[l] < DUP2_BELOW: dups += [(rel, l, True)]
train_aug = [(r, l, False) for r, l in train_items] + dups
print(f"{len(items)} imgs {len(classes)} classes | train {len(train_items)} (+{len(dups)} dups) / val {len(val_items)}", flush=True)
print("dist:", dict(counts.most_common()), flush=True)

TR, VA = train_tfm(), val_tfm()

class DS(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if self.train:
            rel, label, flip = self.rows[i]
        else:
            rel, label = self.rows[i]; flip = False
        img = Image.open(f"{DATA}/originals/{rel}").convert("RGB")
        if flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return (TR if self.train else VA)(img), cls_idx[label]

train_dl = DataLoader(DS(train_aug, True), batch_size=BATCH, shuffle=True,
                      num_workers=WORKERS, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
val_dl = DataLoader(DS(val_items, False), batch_size=BATCH, shuffle=False,
                    num_workers=WORKERS, pin_memory=True, persistent_workers=True)
dev = "cuda"
model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes))
model.to(dev)
ema = copy.deepcopy(model).eval()
for p in ema.parameters(): p.requires_grad_(False)

@torch.no_grad()
def ema_update():
    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point: e.mul_(EMA_DECAY).add_(m, alpha=1 - EMA_DECAY)
        else: e.copy_(m)

w = torch.tensor([1.0 / counts[c] ** 0.5 for c in classes], dtype=torch.float32)
w = (w / w.mean()).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=(EPOCHS * len(train_dl)) // ACCUM + EPOCHS)
scaler = torch.amp.GradScaler()

@torch.no_grad()
def evaluate(net):
    net.eval()
    conf = torch.zeros(len(classes), len(classes), dtype=torch.long)
    with torch.amp.autocast("cuda"):
        for x, y in val_dl:
            x = x.to(dev, non_blocking=True)
            logits = net(x) + net(torch.flip(x, dims=[3]))
            p = logits.argmax(1).cpu()
            for t_, q in zip(y, p): conf[t_, q] += 1
    acc = conf.diag().sum().item() / conf.sum().item()
    return acc, conf

best = 0.0
for ep in range(1, EPOCHS + 1):
    model.train(); t0, seen, lsum = time.time(), 0, 0.0
    opt.zero_grad(set_to_none=True)
    for bi, (x, y) in enumerate(train_dl):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda"):
            if random.random() < MIXUP_P:
                lam = random.betavariate(MIXUP_ALPHA, MIXUP_ALPHA); lam = max(lam, 1 - lam)
                idx = torch.randperm(x.size(0), device=dev)
                logits = model(lam * x + (1 - lam) * x[idx])
                loss = lam * F.cross_entropy(logits, y, weight=w, label_smoothing=0.1) + \
                       (1 - lam) * F.cross_entropy(logits, y[idx], weight=w, label_smoothing=0.1)
            else:
                loss = F.cross_entropy(model(x), y, weight=w, label_smoothing=0.1)
        scaler.scale(loss / ACCUM).backward()
        if (bi + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step(); ema_update()
        seen += y.numel(); lsum += loss.item() * y.numel()
    acc, conf = evaluate(ema)
    per = {classes[i]: round((conf[i, i] / max(conf[i].sum().item(), 1)).item(), 3) for i in range(len(classes))}
    print(f"ep{ep:02d} loss {lsum/seen:.3f} EMA+TTA acc {acc:.4f} ({time.time()-t0:.0f}s) {per}", flush=True)
    if acc > best:
        best = acc
        torch.save({"model": ema.state_dict(), "classes": classes, "input_hw": NATIVE_HW,
                    "acc": acc, "arch": "convnext_base", "tta": "hflip", "letterbox": True,
                    "version": "v4_native_b200"}, f"{OUT}/best.pt")
        json.dump({"classes": classes, "matrix": conf.tolist(), "acc": acc}, open(f"{OUT}/confusion.json", "w"))
        print(f"  saved best {best:.4f}", flush=True)
print(f"ROOM_DONE best {best:.4f}", flush=True)
