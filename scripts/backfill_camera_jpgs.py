#!/usr/bin/env python3
"""Backfill non-delivered shoots' drone edited/ folders with the CAMERA JPG beside
each DNG (matches the live bypass), so tomorrow's deliveries ship camera-original
drone frames instead of the old model output. Only touches shoots in the input
list (undelivered + status=done) that already have a drone edited/ folder.

  backfill_camera_jpgs.py <shoots.txt>
"""
import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.expanduser("~/arem-worker"))
sys.path.insert(0, os.path.expanduser("~/arem-worker/pipeline"))
import cv2
from pipeline.drone_finish import develop, to_12mp

SHOOTS = sys.argv[1]

def rc(args, t=300):
    return subprocess.run(["rclone"] + args, capture_output=True, text=True, timeout=t)

shoots = [l.strip() for l in open(SHOOTS) if l.strip().startswith("AREM (")]
print(f"{len(shoots)} candidate shoots", flush=True)
tot_shoots = tot_cam = tot_dev = 0
for sp in shoots:
    drone = f"dropbox:{sp}/04-RAW-Drone-Other"
    edited = f"{drone}/edited"
    dngs = [x for x in rc(["lsf", drone, "--include", "*.DNG", "--include", "*.dng"], t=60).stdout.split("\n") if x.lower().endswith(".dng")]
    if not dngs:
        continue
    # only backfill shoots already processed (edited/ exists) — others get camera
    # JPGs automatically when the live bypass processes them.
    if rc(["lsf", edited, "--include", "*.jpg"], t=60).returncode != 0:
        print(f"  SKIP (no edited/ yet) {sp.split('/')[-1]}", flush=True); continue
    # available camera JPGs beside the DNGs
    cams = set(os.path.splitext(x)[0] for x in rc(["lsf", drone, "--include", "*.JPG", "--include", "*.jpg", "--include", "*.JPEG", "--include", "*.jpeg"], t=60).stdout.split("\n") if x.strip())
    cam_n = dev_n = 0
    for dng in dngs:
        stem = os.path.splitext(dng)[0]
        if stem in cams:  # camera JPG beside DNG -> server-side copy into edited/
            ext = "JPG"
            for e in ("JPG", "jpg", "JPEG", "jpeg"):
                if rc(["lsf", drone, "--include", f"{stem}.{e}"], t=30).stdout.strip():
                    ext = e; break
            if rc(["copyto", f"{drone}/{stem}.{ext}", f"{edited}/{stem}.jpg", "--progress=false"], t=180).returncode == 0:
                cam_n += 1
        else:  # DNG-only -> clean develop at 12MP
            with tempfile.TemporaryDirectory() as td:
                lp = os.path.join(td, "x.dng"); op = os.path.join(td, f"{stem}.jpg")
                if rc(["copyto", f"{drone}/{dng}", lp], t=300).returncode != 0:
                    continue
                try:
                    cv2.imwrite(op, to_12mp(develop(lp)), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rc(["copyto", op, f"{edited}/{stem}.jpg", "--progress=false"], t=180)
                    dev_n += 1
                except Exception as e:
                    print(f"    ERR develop {stem}: {str(e)[:80]}", flush=True)
    tot_shoots += 1; tot_cam += cam_n; tot_dev += dev_n
    print(f"  {sp.split('/')[-1]}: {cam_n} camera-jpg + {dev_n} clean-develop -> edited/", flush=True)
print(f"BACKFILL DONE: {tot_shoots} shoots, {tot_cam} camera JPGs + {tot_dev} develops written", flush=True)
