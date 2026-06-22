#!/usr/bin/env python3
"""Train a FAST interior/exterior router (mobilenet_v3_small @ 384) to replace
the 12 MP resnet18 in the preview path. Distilled from the teacher's labels.
Target: match the teacher (~99%) but run in ~5-10ms instead of ~280ms.

  train_fast_router.py <data_dir> <out_ckpt>
"""
import os, sys, random
from pathlib import Path
import torch, torch.nn as nn, numpy as np, cv2
import torchvision.models as tvm

cv2.setNumThreads(0)  # fork-safe DataLoader workers
DATA, OUT = sys.argv[1], sys.argv[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["exterior", "interior"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def letterbox(im, size=384):
    h, w = im.shape[:2]; s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(im, (nw, nh))
    canvas = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas

class DS(torch.utils.data.Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        im = letterbox(im, 384)
        if self.train and random.random() < 0.5:
            im = im[:, ::-1]
        t = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return (t - MEAN) / STD, y

items = []
for yi, lbl in enumerate(LABELS):
    for p in Path(DATA, lbl).glob("*.jpg"):
        items.append((str(p), yi))
random.Random(0).shuffle(items)
nval = max(300, len(items) // 10)
val, train = items[:nval], items[nval:]
print(f"train {len(train)} val {len(val)} (ext+int)", flush=True)

model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
model = model.to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
dl = torch.utils.data.DataLoader(DS(train, True), batch_size=64, shuffle=True, num_workers=8, drop_last=True)
vdl = torch.utils.data.DataLoader(DS(val, False), batch_size=128, num_workers=8)

def evaluate():
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for x, y in vdl:
            p = model(x.to(DEV)).argmax(1).cpu()
            ok += (p == y).sum().item(); tot += len(y)
    model.train(); return ok / tot

best = 0.0
for ep in range(12):
    for x, y in dl:
        loss = nn.functional.cross_entropy(model(x.to(DEV)), y.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = evaluate()
    print(f"epoch {ep} val_acc {acc:.4f}", flush=True)
    if acc > best:
        best = acc
        torch.save({"model_state": model.state_dict(), "arch": "mobilenet_v3_small",
                    "label_names": LABELS, "input": 384, "val_acc": acc}, OUT)
        print(f"  saved best {acc:.4f}", flush=True)
print(f"ROUTER DONE best {best:.4f}", flush=True)
