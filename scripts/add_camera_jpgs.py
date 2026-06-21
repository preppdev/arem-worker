#!/usr/bin/env python3
"""Pair the review batch: for the same 50 recent drone DNGs, upload the CAMERA
JPG that sits beside each DNG into the review folder, named to sort right next to
its processed counterpart, so they can be A/B'd at full res in Dropbox.

Processed name: NN__<shoot>__<stem>.jpg
Camera   name: NN__<shoot>__<stem>__CAMERA.jpg   (sorts adjacent)
"""
import sys, os, subprocess, tempfile
SHOOTS = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
REVIEW = "dropbox:AREM (Spiro Uploads)/_drone-nafnet-v2-review"
SAVE_FIRST = os.path.expanduser("~/drone_review/pairs")  # keep first few locally for sampling
os.makedirs(SAVE_FIRST, exist_ok=True)

def rc(args, t=300):
    return subprocess.run(["rclone"] + args, capture_output=True, text=True, timeout=t)

def short(sp):
    b = sp.rstrip("/").split("/")[-1]
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in b)[:60]

# reproduce the EXACT same collection order as batch_drone_review.py
shoots = [l.strip() for l in open(SHOOTS) if l.strip() and l.startswith("AREM (")]
jobs = []
for sp in shoots:
    if len(jobs) >= N:
        break
    r = rc(["lsf", f"dropbox:{sp}/04-RAW-Drone-Other", "--include", "*.DNG", "--include", "*.dng"], t=60)
    for d in [x for x in r.stdout.split("\n") if x.lower().endswith(".dng")]:
        if len(jobs) >= N:
            break
        jobs.append((sp, d))

done = 0
for i, (sp, dng) in enumerate(jobs, 1):
    stem = os.path.splitext(dng)[0]
    src_dir = f"dropbox:{sp}/04-RAW-Drone-Other"
    name = f"{i:02d}__{short(sp)}__{stem}__CAMERA.jpg"
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "cam.jpg")
        ok = False
        for ext in ("JPG", "jpg", "JPEG", "jpeg"):
            if rc(["copyto", f"{src_dir}/{stem}.{ext}", lp], t=180).returncode == 0:
                ok = True; break
        if not ok:
            print(f"  [{i}/{len(jobs)}] no camera JPG for {stem}", flush=True); continue
        rc(["copyto", lp, f"{REVIEW}/{name}", "--progress=false"], t=180)
        if i <= 4:  # keep the first few locally to send as full-res samples
            import shutil; shutil.copy2(lp, os.path.join(SAVE_FIRST, f"{i:02d}_camera.jpg"))
            rc(["copyto", f"{REVIEW}/{i:02d}__{short(sp)}__{stem}.jpg", os.path.join(SAVE_FIRST, f"{i:02d}_processed.jpg")], t=180)
        done += 1
        print(f"  [{i}/{len(jobs)}] paired camera {name}", flush=True)
print(f"CAMERA PAIRING DONE: {done}/{len(jobs)} camera JPGs uploaded to {REVIEW}", flush=True)
