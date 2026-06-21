#!/usr/bin/env python3
"""Assemble the drone retrain dataset: good_develop(DNG) -> dngoutput target.

Pulls each DNG + its dngoutput finished JPG from Dropbox, develops the DNG with
the production recipe, downscales BOTH to a 2048 long side (plenty for learning
a tone/style mapping; keeps storage + training fast and pixel-aligned), and
writes <out>/<stem>_in.jpg + <stem>_tgt.jpg. Resumable (skips done stems).

  build_drone_dataset.py <out_dir> [limit]
"""
import os, sys, subprocess, tempfile, glob
import numpy as np, cv2, rawpy

OUT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
os.makedirs(OUT, exist_ok=True)
SRC = "dropbox:AREM (Spiro Uploads)/training-data/Drone Dng"
LONG = 2048

def rc(args, timeout=180):
    return subprocess.run(["rclone"] + args, capture_output=True, text=True, timeout=timeout)

def good_develop(dng):
    with rawpy.imread(dng) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0,
                              output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5),
                              output_bps=8, user_flip=-1)
    return rgb[:, :, ::-1].copy()  # BGR

def down(img):
    h, w = img.shape[:2]; s = LONG / max(h, w)
    return cv2.resize(img, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA) if s < 1 else img

# stems that have BOTH a DNG and a dngoutput target
dng_list = [x for x in rc(["lsf", SRC, "--include", "*.DNG"]).stdout.split("\n") if x.endswith(".DNG")]
tgt_set = set(x for x in rc(["lsf", f"{SRC}/dngoutput", "--include", "*.jpg"]).stdout.split("\n") if x.endswith(".jpg"))
stems = [d[:-4] for d in dng_list if f"{d[:-4]}.jpg" in tgt_set]
print(f"{len(stems)} stems with DNG+dngoutput target", flush=True)

done = ok = 0
for i, stem in enumerate(sorted(stems)):
    if ok >= LIMIT: break
    ip, tp = os.path.join(OUT, f"{stem}_in.jpg"), os.path.join(OUT, f"{stem}_tgt.jpg")
    if os.path.isfile(ip) and os.path.isfile(tp):
        done += 1; ok += 1; continue
    with tempfile.TemporaryDirectory() as td:
        dng = os.path.join(td, "x.dng"); tgt = os.path.join(td, "t.jpg")
        if rc(["copyto", f"{SRC}/{stem}.DNG", dng], timeout=300).returncode != 0:
            print(f"  miss DNG {stem}", flush=True); continue
        if rc(["copyto", f"{SRC}/dngoutput/{stem}.jpg", tgt], timeout=180).returncode != 0:
            print(f"  miss tgt {stem}", flush=True); continue
        try:
            dev = down(good_develop(dng))
            t = cv2.imread(tgt)
            if t.shape[:2] != dev.shape[:2]:
                t = cv2.resize(t, (dev.shape[1], dev.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.imwrite(ip, dev, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(tp, t, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok += 1
        except Exception as e:
            print(f"  ERR {stem}: {str(e)[:100]}", flush=True); continue
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(stems)} done (ok={ok}, cached={done})", flush=True)
print(f"DATASET DONE: {ok} pairs in {OUT}", flush=True)
