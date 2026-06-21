#!/usr/bin/env python3
"""Batch 50 recent drone DNGs through the production NAFNet v2 + clarity finish
and upload the 12 MP results to a Dropbox review folder. Provenance (shoot +
stem) is baked into each output filename so the reviewer knows the source.

  batch_drone_review.py <recent_shoots.txt> [N]
"""
import sys, os, subprocess, tempfile
import numpy as np, cv2, torch
sys.path.insert(0, os.path.expanduser("~/arem-worker"))
sys.path.insert(0, os.path.expanduser("~/arem-worker/pipeline"))
from pipeline.drone_finish import develop, to_12mp, nafnet_finish
from models.nafnet import NAFNet

SHOOTS = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
REVIEW = "dropbox:AREM (Spiro Uploads)/_drone-nafnet-v2-review"
MODEL = os.path.expanduser("~/polish-cache/models/drone-finish/nafnet_v2.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def rc(args, t=300):
    return subprocess.run(["rclone"] + args, capture_output=True, text=True, timeout=t)

# load model
ck = torch.load(MODEL, map_location=DEV, weights_only=False)
net = NAFNet(in_channels=3, out_channels=3, width=32, middle_blk_num=12,
             enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
             use_residual=True, residual_start=0).to(DEV).eval()
net.load_state_dict(ck["model"])
print(f"model val_l1={ck.get('val_l1'):.4f} dev={DEV}", flush=True)

# collect up to N drone DNGs across recent shoots
shoots = [l.strip() for l in open(SHOOTS) if l.strip() and l.startswith("AREM (")]
jobs = []  # (shoot_path, dng_name)
for sp in shoots:
    if len(jobs) >= N:
        break
    r = rc(["lsf", f"dropbox:{sp}/04-RAW-Drone-Other", "--include", "*.DNG", "--include", "*.dng"], t=60)
    dngs = [x for x in r.stdout.split("\n") if x.lower().endswith(".dng")]
    for d in dngs:
        if len(jobs) >= N:
            break
        jobs.append((sp, d))
print(f"collected {len(jobs)} drone DNGs from {len(set(j[0] for j in jobs))} shoots", flush=True)

def short(sp):
    b = sp.rstrip("/").split("/")[-1]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in b)[:60]

done = 0
for i, (sp, dng) in enumerate(jobs, 1):
    stem = os.path.splitext(dng)[0]
    outname = f"{i:02d}__{short(sp)}__{stem}.jpg"
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "x.dng")
        if rc(["copyto", f"dropbox:{sp}/04-RAW-Drone-Other/{dng}", lp], t=300).returncode != 0:
            print(f"  [{i}/{len(jobs)}] miss {stem}", flush=True); continue
        try:
            base = to_12mp(develop(lp))
            out = nafnet_finish(net, base, DEV)
            op = os.path.join(td, outname)
            cv2.imwrite(op, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rc(["copyto", op, f"{REVIEW}/{outname}", "--progress=false"], t=180)
            done += 1
            print(f"  [{i}/{len(jobs)}] done {outname} ({out.shape[1]}x{out.shape[0]})", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(jobs)}] ERR {stem}: {str(e)[:120]}", flush=True)
print(f"BATCH DONE: {done}/{len(jobs)} uploaded to {REVIEW}", flush=True)
