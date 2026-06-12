#!/usr/bin/env python3
"""Camera/tripod detector v3 — Faster R-CNN R50-FPN v2 at NATIVE resolution
(min 2800 / max 4200; annotations pre-scaled to native space). Self-contained
mAP50 eval matching the v2 harness convention. B200."""
import json, os, time
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.v2 import functional as TF

Image.MAX_IMAGE_PIXELS = None
DATA = os.environ.get("CLOUD_DATA", "/workspace/data")
OUT = os.environ.get("OUT", "/workspace/runs/detector_native")
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = int(os.environ.get("BATCH", "4"))
LR = float(os.environ.get("LR", "0.005"))
WORKERS = int(os.environ.get("WORKERS", "16"))
MIN_SIZE, MAX_SIZE = 2800, 4200
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(17)

class DetDS(Dataset):
    def __init__(self, ann_path, train):
        self.recs = json.load(open(ann_path))
        self.train = train
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(f"{DATA}/originals/{r['path']}").convert("RGB")
        boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(r["labels"], dtype=torch.int64)
        if self.train and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                W = img.size[0]
                boxes = torch.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], dim=1)
        x = TF.to_dtype(TF.to_image(img), torch.float32, scale=True)
        return x, {"boxes": boxes, "labels": labels}

def collate(b): return tuple(zip(*b))

train_dl = DataLoader(DetDS(f"{DATA}/detector/ann_train_native.json", True), batch_size=BATCH,
                      shuffle=True, num_workers=WORKERS, collate_fn=collate, persistent_workers=True, prefetch_factor=4)
val_ds = DetDS(f"{DATA}/detector/ann_val_native.json", False)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS, collate_fn=collate, persistent_workers=True)
dev = "cuda"
model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=MIN_SIZE, max_size=MAX_SIZE)
in_feat = model.roi_heads.box_predictor.cls_score.in_features
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 3)  # bg + camera + tripod
model.to(dev)
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler()

def iou(a, b):
    inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

@torch.no_grad()
def evaluate():
    """AP50 per class (11-pt-free AUC over score-ranked dets), mean of classes."""
    model.eval()
    gts = defaultdict(list); dets = defaultdict(list)
    img_id = 0
    for xs, ts in val_dl:
        outs = model([x.to(dev) for x in xs])
        for t, o in zip(ts, outs):
            for b, l in zip(t["boxes"].tolist(), t["labels"].tolist()):
                gts[l].append((img_id, b))
            for b, l, s in zip(o["boxes"].cpu().tolist(), o["labels"].cpu().tolist(), o["scores"].cpu().tolist()):
                dets[l].append((s, img_id, b))
            img_id += 1
    aps = {}
    for cls in (1, 2):
        gt = gts[cls]
        gt_by_img = defaultdict(list)
        for iid, b in gt: gt_by_img[iid].append([b, False])
        d = sorted(dets[cls], reverse=True)
        tp, fp = [], []
        for s, iid, b in d:
            matched = False
            for g in gt_by_img.get(iid, []):
                if not g[1] and iou(b, g[0]) >= 0.5:
                    g[1] = True; matched = True; break
            tp.append(1 if matched else 0); fp.append(0 if matched else 1)
        npos = len(gt)
        if npos == 0: aps[cls] = float("nan"); continue
        ctp = cfp = 0; prec_rec = []
        for t_, f_ in zip(tp, fp):
            ctp += t_; cfp += f_
            prec_rec.append((ctp / (ctp + cfp), ctp / npos))
        ap, last_r = 0.0, 0.0
        for p, r in prec_rec:
            ap += p * (r - last_r); last_r = r
        aps[cls] = ap
    m = sum(v for v in aps.values() if v == v) / 2
    print(f"  mAP50 {m:.4f} (camera {aps.get(1, 0):.3f} / tripod {aps.get(2, 0):.3f})", flush=True)
    return m

best = 0.0
for ep in range(1, EPOCHS + 1):
    model.train(); t0, lsum, n = time.time(), 0.0, 0
    for xs, ts in train_dl:
        xs = [x.to(dev) for x in xs]
        ts = [{k: v.to(dev) for k, v in t.items()} for t in ts]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            losses = model(xs, ts)
            loss = sum(losses.values())
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        lsum += loss.item(); n += 1
    sched.step()
    m = evaluate()
    print(f"ep{ep:02d}/{EPOCHS} loss {lsum/max(n,1):.3f} ({time.time()-t0:.0f}s)", flush=True)
    if m > best:
        best = m
        torch.save({"model_state": model.state_dict(), "num_classes": 3, "metrics": {"mAP50": best},
                    "min_size": MIN_SIZE, "max_size": MAX_SIZE, "version": "v3_native_b200"}, f"{OUT}/best.pth")
        print(f"  saved best {best:.4f}", flush=True)
print(f"DETECTOR_DONE best {best:.4f}", flush=True)
