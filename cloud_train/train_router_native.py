#!/usr/bin/env python3
"""Interior/exterior router v4 — ResNet-18, NATIVE full-frame letterbox, B200.
Drop-in for pipeline CLASSIFIER_PATH ({label_names, num_classes, model_state,
val_acc} dict, same as classifier_v2/v3). Shoot-split seed 17."""
import json, os, random, time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from common import DATA, NATIVE_HW, load_labeled, shoot_split, BytesDS, bytes_collate, GpuPrep

OUT = os.environ.get("OUT", "/workspace/runs/router_native")
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "3e-4"))
WORKERS = int(os.environ.get("WORKERS", "24"))
os.makedirs(OUT, exist_ok=True)
random.seed(17); torch.manual_seed(17)

items = load_labeled("corrected_scene.jsonl")
classes = sorted({l for _, l in items})  # ["exterior", "interior"]
cls_idx = {c: i for i, c in enumerate(classes)}
train_items, val_items = shoot_split(items)
print(f"{len(items)} imgs {Counter(l for _, l in items)} | train {len(train_items)} / val {len(val_items)}", flush=True)

train_dl = DataLoader(BytesDS(train_items, cls_idx, DATA), batch_size=BATCH, shuffle=True,
                      num_workers=WORKERS, collate_fn=bytes_collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
val_dl = DataLoader(BytesDS(val_items, cls_idx, DATA), batch_size=BATCH, shuffle=False,
                    num_workers=WORKERS, collate_fn=bytes_collate, persistent_workers=True)
dev = "cuda"
prep_train, prep_val = GpuPrep(train=True), GpuPrep(train=False)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(train_dl))
crit = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler()

def evaluate():
    model.eval()
    conf = torch.zeros(2, 2, dtype=torch.long)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for bl, y in val_dl:
            p = model(prep_val(bl)).argmax(1).cpu()
            for t_, q in zip(y, p): conf[t_, q] += 1
    acc = conf.diag().sum().item() / conf.sum().item()
    rec = {classes[i]: round(conf[i, i].item() / max(1, conf[i].sum().item()), 4) for i in range(2)}
    print(f"  val acc {acc:.4f} recall {rec} conf {conf.tolist()}", flush=True)
    return acc

best = 0.0
for ep in range(1, EPOCHS + 1):
    model.train(); t0, seen, lsum = time.time(), 0, 0.0
    for bl, y in train_dl:
        x, y = prep_train(bl), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        seen += y.numel(); lsum += loss.item() * y.numel()
    acc = evaluate()
    print(f"ep{ep}/{EPOCHS} loss {lsum/seen:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if acc > best:
        best = acc
        torch.save({"label_names": classes, "num_classes": 2, "model_state": model.state_dict(),
                    "val_acc": best, "input_hw": NATIVE_HW, "letterbox": True,
                    "version": "v4_native_b200"}, f"{OUT}/router_v4_native.pth")
        print(f"  saved best {best:.4f}", flush=True)
print(f"ROUTER_DONE best {best:.4f}", flush=True)
