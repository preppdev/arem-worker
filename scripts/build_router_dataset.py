#!/usr/bin/env python3
"""Build the fast-router distillation dataset: pull each labeled Stage-1 frame
from R2, downscale to 384 long-side, save under data/<label>/. Labels come from
the existing 12MP classifier's verdict (ImageReview.isInteriorWorker) — i.e. we
distill the slow teacher into a small fast student. Threaded downloads.

  build_router_dataset.py <labels.tsv> <out_dir>
"""
import os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, boto3

TSV, OUT = sys.argv[1], sys.argv[2]
S3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
B = "arem-training-data"
for lbl in ("interior", "exterior"):
    Path(OUT, lbl).mkdir(parents=True, exist_ok=True)
rows = [l.rstrip("\n").split("\t") for l in open(TSV) if "\t" in l]

def work(arg):
    i, (lbl, key) = arg
    out = Path(OUT, lbl, f"{i}.jpg")
    if out.exists():
        return "cached"
    try:
        im = cv2.imdecode(np.frombuffer(S3.get_object(Bucket=B, Key=key)["Body"].read(), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return "decodefail"
        h, w = im.shape[:2]; s = 384 / max(h, w)
        im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return "ok"
    except Exception as e:
        return f"err {str(e)[:60]}"

done = ok = 0
with ThreadPoolExecutor(max_workers=24) as ex:
    for res in ex.map(work, enumerate(rows)):
        done += 1
        if res in ("ok", "cached"):
            ok += 1
        if done % 500 == 0:
            print(f"{done}/{len(rows)} ok={ok}", flush=True)
print("counts:", {lbl: len(list(Path(OUT, lbl).glob("*.jpg"))) for lbl in ("interior", "exterior")}, flush=True)
print(f"DATASET DONE {ok}/{len(rows)}", flush=True)
