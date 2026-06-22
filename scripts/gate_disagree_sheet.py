#!/usr/bin/env python3
"""For a job, find frames where the fast student and the teacher-at-1600 disagree,
and build a labeled contact sheet so a human can judge who's actually right
(student vs degraded-teacher-at-preview-res). Each tile: 's=<student> t1600=<teacher>'."""
import os, sys
from pathlib import Path
import torch, torch.nn as nn, numpy as np, cv2
from PIL import Image
import torchvision.models as tvm
BASE = Path(os.path.expanduser("~/arem-worker")); sys.path.insert(0, str(BASE))
from pipeline import run_pipeline as R
import boto3
DEV = "cuda"
teacher, tlabels, tfm_full = R.load_classifier(BASE / "pipeline" / "classifier_v4.pth", DEV)
ck = torch.load(os.path.expanduser("~/router_train/fast_router.pt"), map_location=DEV)
fast = tvm.mobilenet_v3_small(); fast.classifier[-1] = nn.Linear(fast.classifier[-1].in_features, 2)
fast.load_state_dict(ck["model_state"]); fast = fast.to(DEV).eval()
FL = ck["label_names"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def lb(im, size=384):
    h, w = im.shape[:2]; s = size / max(h, w); nw, nh = max(1, round(w*s)), max(1, round(h*s))
    r = cv2.resize(im, (nw, nh)); c = np.zeros((size, size, 3), np.uint8); t, l = (size-nh)//2, (size-nw)//2; c[t:t+nh, l:l+nw] = r; return c
def fp(bgr):
    t = torch.from_numpy(lb(bgr)[:, :, ::-1].copy()).permute(2, 0, 1).float()/255.0
    with torch.no_grad(): p = torch.softmax(fast(((t-MEAN)/STD).unsqueeze(0).to(DEV)), 1).cpu().numpy()[0]
    return FL[int(p.argmax())]
S3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
B = "arem-training-data"
keys = [o["Key"] for o in S3.list_objects_v2(Bucket=B, Prefix=f"training-stage1/{sys.argv[1]}/").get("Contents", [])][:80]
tiles = []
for k in keys:
    bgr = cv2.imdecode(np.frombuffer(S3.get_object(Bucket=B, Key=k)["Body"].read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]; s = min(1, 1600/max(h, w)); prev = cv2.resize(bgr, (round(w*s), round(h*s)))
    im = Image.fromarray(prev[:, :, ::-1]); xt = tfm_full(im).unsqueeze(0).to(DEV)
    with torch.no_grad(): tl = tlabels[int(torch.softmax(teacher(xt), 1).argmax())]
    fl = fp(prev)
    if fl != tl:
        th = cv2.resize(prev, (300, 200)); cv2.putText(th, f"s={fl[:3]} t={tl[:3]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(th)
        if len(tiles) >= 12: break
if tiles:
    while len(tiles) % 4: tiles.append(np.zeros((200, 300, 3), np.uint8))
    rows = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite("/tmp/disagree_sheet.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {len([t for t in tiles if t.any()])} disagreement tiles")
else:
    print("no disagreements")
