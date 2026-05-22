"""Build the local dataset + manifest for the Stage-2 window fine-tune pilot.

For each accepted window-mask row (with stage1R2Path populated), downloads:
  - Stage-1 output  (input)   stored as JPEG at common resolution
  - Vendor JPG      (target)  stored as JPEG at common resolution
  - Window mask PNG (mask)    stored as PNG  at common resolution
…all resized to match the vendor's native resolution (since that's what we
want the model to produce). Mask is resized with nearest-neighbour.

Writes a CSV manifest with columns:
    pair_id, shoot_folder, input, output, mask

Usage on the 3090 box:
    cd ~/arem-photo-ai-V2
    python build_window_pilot_manifest.py \\
        --limit 1000 \\
        --out-dir /tmp/window_pilot_v1 \\
        --workers 8

Env:
    DATABASE_URL — read from ~/arem-worker/.env.local
    R2 access    — via rclone (existing 'r2:' remote)
"""
from __future__ import annotations

import argparse
import csv as csv_module
import io
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore
from PIL import Image  # type: ignore

R2_TRAINING_BUCKET = "arem-training-data"
# All three artifacts (stage1 output, vendor target, window mask) live in
# the training bucket. Keys in the DB are bare R2 keys (no bucket prefix).
RCLONE_R2 = os.environ.get("RCLONE_R2", "r2")
JPEG_QUALITY = 95

_lock = Lock()
def log(s: str) -> None:
    with _lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)


def fetch_candidates(conn, limit: int) -> list[dict]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT ir.id, ir."midStem", ir."jobId",
               ir."stage1R2Path", ir."vendorImageR2Path", ir."windowMaskR2Path",
               j."jobFolderName" AS shoot_folder
        FROM "ImageReview" ir
        JOIN "Job" j ON j.id = ir."jobId"
        WHERE ir."windowMaskVerdict" = 'accepted'
          AND ir."stage1R2Path" LIKE 'training-stage1/%%'
          AND ir."vendorImageR2Path" LIKE 'vendor-mirrors/%%'
          AND ir."windowMaskR2Path" IS NOT NULL
        ORDER BY random()
        LIMIT %s
        """,
        (limit,),
    )
    rows = list(cur.fetchall())
    cur.close()
    return rows


def rclone_pull(bucket: str, key: str, dest: Path) -> None:
    """Pull a single R2 object to local. Raises on failure."""
    src = f"{RCLONE_R2}:{bucket}/{key.lstrip('/')}"
    res = subprocess.run(
        ["rclone", "copyto", src, str(dest), "--s3-no-check-bucket", "--no-traverse"],
        capture_output=True, text=True, timeout=60,
    )
    if res.returncode != 0:
        raise RuntimeError(f"rclone failed for {src}: {res.stderr[:200]}")
    # rclone exits 0 with "nothing to transfer" when the source 404s, so
    # verify the local file actually landed.
    if not dest.exists() or dest.stat().st_size == 0:
        raise FileNotFoundError(f"rclone returned 0 but {src} did not produce a local file")


def load_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".jxl":
        from imagecodecs import jpegxl_decode  # type: ignore
        with open(path, "rb") as f:
            arr = jpegxl_decode(f.read())
        return Image.fromarray(arr)
    return Image.open(path).convert("RGB")


def process_one(row: dict, out_dir: Path) -> dict:
    pair_id = row["midStem"] or row["id"]
    shoot_folder = row["shoot_folder"] or "unknown"
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="winpilot_") as td:
        td = Path(td)
        # 1. Pull all three
        vendor_local = td / "vendor.jpg"
        mask_local = td / "mask.png"
        stage1_local = td / f"stage1{Path(row['stage1R2Path']).suffix}"
        rclone_pull(R2_TRAINING_BUCKET, row["stage1R2Path"], stage1_local)
        rclone_pull(R2_TRAINING_BUCKET, row["vendorImageR2Path"], vendor_local)
        rclone_pull(R2_TRAINING_BUCKET, row["windowMaskR2Path"], mask_local)

        # 2. Load + canonical resolution = vendor target
        vendor_img = load_image(vendor_local)
        target_w, target_h = vendor_img.size
        stage1_img = load_image(stage1_local).resize(
            (target_w, target_h), Image.BICUBIC
        )
        mask_img = Image.open(mask_local).convert("L").resize(
            (target_w, target_h), Image.NEAREST
        )

        # 3. Save into the pilot dir
        inp_out = out_dir / "inputs" / f"{pair_id}.jpg"
        tgt_out = out_dir / "outputs" / f"{pair_id}.jpg"
        msk_out = out_dir / "masks" / f"{pair_id}.png"
        stage1_img.save(inp_out, "JPEG", quality=JPEG_QUALITY)
        vendor_img.save(tgt_out, "JPEG", quality=JPEG_QUALITY)
        mask_img.save(msk_out, "PNG", optimize=True)

    return {
        "pair_id": pair_id,
        "shoot_folder": shoot_folder,
        "input": f"inputs/{pair_id}.jpg",
        "output": f"outputs/{pair_id}.jpg",
        "mask": f"masks/{pair_id}.png",
        "ms": int((time.time() - t0) * 1000),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv-name", default="pairs.csv")
    args = ap.parse_args()

    if not os.environ.get("DATABASE_URL"):
        sys.exit("missing env: DATABASE_URL")

    out_dir = Path(args.out_dir)
    for sub in ("inputs", "outputs", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    rows = fetch_candidates(conn, args.limit)
    conn.close()
    log(f"selected {len(rows)} accepted rows")

    csv_path = out_dir / args.csv_name
    done = err = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["pair_id", "shoot_folder", "input", "output", "mask"])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, r, out_dir): r for r in rows}
            t0 = time.time()
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    r = fut.result()
                    writer.writerow([
                        r["pair_id"], r["shoot_folder"],
                        r["input"], r["output"], r["mask"],
                    ])
                    f.flush()
                    done += 1
                except Exception as e:
                    err += 1
                    row = futs[fut]
                    log(f"  ERR {row.get('midStem')}: {str(e)[:200]}")
                if i % 50 == 0:
                    rate = i / max(1, time.time() - t0)
                    eta = (len(rows) - i) / max(rate, 0.01) / 60
                    log(f"  [{i}/{len(rows)}] done={done} err={err} rate={rate:.1f}/s eta={eta:.0f}min")

    log(f"finished: done={done} err={err} csv={csv_path}")


if __name__ == "__main__":
    main()
