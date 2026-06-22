#!/usr/bin/env python3
"""Gate the fast router before deploy: agreement with the 12MP teacher on
PREVIEW-resolution frames (the real deploy scenario — 1600px preview -> 384
student vs 1600px -> 2800 letterbox teacher), plus timing + the confidence of
any disagreements (to calibrate a fallback gate)."""
import os, sys, time
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

def letterbox(im, size=384):
    h, w = im.shape[:2]; s = size / max(h, w); nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(im, (nw, nh)); c = np.zeros((size, size, 3), np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2; c[t:t + nh, l:l + nw] = r; return c

def fast_pred(bgr):
    t = torch.from_numpy(letterbox(bgr)[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    x = ((t - MEAN) / STD).unsqueeze(0).to(DEV)
    with torch.no_grad():
        p = torch.softmax(fast(x), 1).cpu().numpy()[0]
    return FL[int(p.argmax())], float(p.max())

S3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
B = "arem-training-data"
keys = [o["Key"] for o in S3.list_objects_v2(Bucket=B, Prefix=f"training-stage1/{sys.argv[1]}/").get("Contents", [])][:80]
agree = n = 0; tfast = 0.0; dis = []
for k in keys:
    bgr = cv2.imdecode(np.frombuffer(S3.get_object(Bucket=B, Key=k)["Body"].read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]; s = min(1, 1600 / max(h, w)); prev = cv2.resize(bgr, (round(w * s), round(h * s)))
    im = Image.fromarray(prev[:, :, ::-1]); xt = tfm_full(im).unsqueeze(0).to(DEV)
    with torch.no_grad():
        tlab = tlabels[int(torch.softmax(teacher(xt), 1).argmax())]
    a = time.time(); fl, conf = fast_pred(prev); tfast += time.time() - a
    if fl == tlab:
        agree += 1
    else:
        dis.append(round(conf, 2))
    n += 1
print(f"teacher-agreement on {n} preview frames: {agree}/{n} = {agree/n:.3f}")
print(f"fast router avg {int(tfast/n*1000)}ms  (teacher ~280ms)")
print(f"disagreement confidences: {sorted(dis)}")
