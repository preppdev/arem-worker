"""AREM editing pipeline worker.

Polls the dashboard /api/jobs/claim endpoint for queued shoots, pulls ARWs
from Dropbox, runs the production pipeline (Photomatix merge → Stage 2
Restormer → auto-upright + EXIF), uploads the finished JPEGs back to
Dropbox under /08-Test-Edit, and reports status to the dashboard.

Designed to be runnable both locally (current host) and in a Docker
container on RunPod. Stateless except for the local scratch dir that
gets cleaned per job.

Env vars:
  WORKER_TOKEN          — required. Sent in x-worker-token header.
  WORKER_ID             — optional. Defaults to hostname.
  DASHBOARD_URL         — defaults to https://arem-editing-dashboard.vercel.app
  RCLONE_DROPBOX        — rclone remote name. Default 'dropbox'
  PMTX_STATIC           — path to PhotomatixCL-static binary
  AREM_REPO             — path to arem-photo-ai-V2 repo (for inference scripts)
  UPRIGHT_REPO          — path to upright_test (for auto_upright.py)
  CHECKPOINT_INTERIOR   — path to interior Stage 2 checkpoint
  CHECKPOINT_EXTERIOR   — path to exterior Stage 2 checkpoint
  CLASSIFIER_PATH       — path to classifier_v2.pth
  PYTHON_BIN            — python interpreter to run pipeline scripts
                          (default: same interpreter as worker)
  WORK_ROOT             — scratch dir for downloaded ARWs and outputs.
                          default /tmp/arem-worker
  DROPBOX_OUTPUT_FOLDER — name of the output subfolder under the job.
                          default '08-Test-Edit' (warmup); switch to
                          '05-Finished-Photos' once production-ready.

Usage:
  python worker.py --once          # claim and process one job, then exit
  python worker.py                 # loop forever, polling every IDLE_SLEEP s
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import urllib.request
import urllib.error
import urllib.parse

# ---- config ----
DASHBOARD_URL = os.environ.get("DASHBOARD_URL",
    "https://arem-editing-dashboard.vercel.app").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")
WORKER_ID = os.environ.get("WORKER_ID", f"local-{socket.gethostname()}")
RCLONE_REMOTE = os.environ.get("RCLONE_DROPBOX", "dropbox")
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/tmp/arem-worker"))
DROPBOX_OUTPUT_FOLDER = os.environ.get("DROPBOX_OUTPUT_FOLDER", "08-Test-Edit")
IDLE_SLEEP = int(os.environ.get("IDLE_SLEEP", "60"))
HEARTBEAT_SLEEP = int(os.environ.get("HEARTBEAT_SLEEP", "30"))

PMTX_STATIC = os.environ.get("PMTX_STATIC",
    str(Path.home() / "photomatix/PhotomatixCL/PhotomatixCL-static"))
AREM_REPO = Path(os.environ.get("AREM_REPO",
    str(Path.home() / "arem-photo-ai-V2")))
UPRIGHT_REPO = Path(os.environ.get("UPRIGHT_REPO",
    str(Path.home() / "upright_test")))

CHECKPOINT_INTERIOR = Path(os.environ.get("CHECKPOINT_INTERIOR",
    str(Path.home() / "checkpoints_remote/interior_full_v1_latest.pth")))
CHECKPOINT_EXTERIOR = Path(os.environ.get("CHECKPOINT_EXTERIOR",
    str(Path.home() / "checkpoints_remote/exterior_full_v1_latest.pth")))
CLASSIFIER_PATH = Path(os.environ.get("CLASSIFIER_PATH",
    str(AREM_REPO / "classifier_v2.pth")))
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def log(msg: str):
    print(f"[worker {WORKER_ID}] {msg}", flush=True)


# ---- HTTP helpers ----
def _post(path: str, body: dict | None = None) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "x-worker-token": WORKER_TOKEN}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} -> HTTP {e.code}: {body_text[:300]}")
    except Exception as e:
        raise RuntimeError(f"POST {path} failed: {e}")


def claim_job() -> dict | None:
    r = _post("/api/jobs/claim", {"workerId": WORKER_ID})
    return r.get("job")


def post_status(job_id: str, status: str, **fields) -> dict:
    body = {"status": status, **fields}
    return _post(f"/api/jobs/{job_id}/status", body)


def _get(path: str) -> dict:
    url = f"{DASHBOARD_URL}{path}"
    req = urllib.request.Request(
        url, method="GET",
        headers={"x-worker-token": WORKER_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {path} -> HTTP {e.code}: {body_text[:300]}")
    except Exception as e:
        raise RuntimeError(f"GET {path} failed: {e}")


def _http_download(url: str, dest: Path) -> int:
    """Stream-download a presigned URL to a local file. Returns bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="GET")
    total = 0
    with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as fh:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    return total


def _http_put(url: str, src: Path) -> None:
    """Stream-upload a local file via PUT to a presigned URL."""
    size = src.stat().st_size
    with src.open("rb") as fh:
        req = urllib.request.Request(
            url, data=fh.read(), method="PUT",
            headers={"Content-Length": str(size)},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                # Drain to release the connection
                r.read()
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"PUT {url[:80]}... -> HTTP {e.code}: {body_text[:300]}")


# ---- Dropbox path resolution ----
def resolve_dropbox_source(stored_path: str) -> str:
    """Translate the stored dropboxPath into an rclone remote path.

    Stored format examples:
      'Jordan DiCaprio/AREM (Spiro Uploads)/2026/Q2/Blair/123 Vine St'
      'AREM (Spiro Uploads)/2026/Q2/Blair/123 Vine St'

    The personal Dropbox is rooted at the user's drive — we strip
    'Jordan DiCaprio/' if present and prepend the rclone remote.
    """
    p = stored_path.strip().rstrip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/"):]
    return f"{RCLONE_REMOTE}:{p}"


def rclone(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    cmd = ["rclone", *args]
    log(f"  rclone {' '.join(args[:2])} ...")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---- pipeline steps ----
# All AREM shoots use this fixed folder structure; we no longer try
# fallback names. If a folder doesn't have 01-RAW-Photos/, that's a data
# problem to surface — not a routing question.
RAW_SUBFOLDER = "01-RAW-Photos"
# Image formats we accept as bracket inputs. RAW formats hit rawpy +
# lensfun (Sony ARW, Nikon NEF, Canon CR2/CR3, plus generic DNG/RAF/RW2).
# Everything else (JPG/JPEG/TIFF/PNG/JXL/WEBP) gets decoded directly via
# PIL (lens correction skipped — the in-camera JPG already has it baked).
ACCEPTED_EXTS = ["ARW", "arw", "NEF", "nef", "CR2", "cr2", "CR3", "cr3",
                 "DNG", "dng", "RAF", "raf", "RW2", "rw2",
                 "JPG", "jpg", "JPEG", "jpeg",
                 "TIF", "tif", "TIFF", "tiff", "PNG", "png",
                 "JXL", "jxl", "WEBP", "webp"]


def _include_flags(exts: list[str]) -> list[str]:
    out = []
    for e in exts:
        out += ["--include", f"*.{e}"]
    return out


def download_raws(dropbox_remote_path: str, local_raw_dir: Path) -> tuple[int, int, str]:
    """Pull RAW_SUBFOLDER contents to local. Returns (file_count, bytes, used_subfolder).

    AREM shoots have a fixed structure — `01-RAW-Photos/` always exists on
    a real shoot. Missing or empty means the photographer hasn't finished
    uploading; that's a data state to report up, not a path to retry.
    """
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    include_flags = _include_flags(ACCEPTED_EXTS)
    src = f"{dropbox_remote_path}/{RAW_SUBFOLDER}"
    ls = rclone(["lsf", src, *include_flags], timeout=60)
    if ls.returncode != 0:
        raise RuntimeError(
            f"{RAW_SUBFOLDER} not found under {dropbox_remote_path} "
            f"(rc={ls.returncode}): {ls.stderr[-200:]}")
    if not ls.stdout.strip():
        raise RuntimeError(
            f"{RAW_SUBFOLDER} is empty (no files matching {ACCEPTED_EXTS}) "
            f"under {dropbox_remote_path} — shoot upload may be incomplete")
    r = rclone(["copy", src, str(local_raw_dir),
                *include_flags,
                "--transfers", "8", "--checkers", "16",
                "--progress=false"])
    if r.returncode != 0:
        raise RuntimeError(f"rclone copy failed rc={r.returncode}: {r.stderr[-300:]}")
    files = []
    for ext in ACCEPTED_EXTS:
        files += list(local_raw_dir.glob(f"*.{ext}"))
    total = sum(p.stat().st_size for p in files)
    return len(files), total, RAW_SUBFOLDER


def run_inference(local_raw_dir: Path, pred_root: Path) -> Path:
    """Run the canonical Stage 1 (NAFNet) + Stage 2 (Restormer routed) pipeline.

    Calls pipeline/run_pipeline.py via subprocess to keep deps isolated.
    Returns the directory containing the Stage 2 JPGs (named
    <stem>_stage2-{int|ext}.jpg).
    """
    pred_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, str(AREM_REPO / "run_pipeline.py"),
        "--raw-dir", str(local_raw_dir),
        "--output-dir", str(pred_root),
    ]
    # CHECKPOINT_STAGE1 / CHECKPOINT_INTERIOR / CHECKPOINT_EXTERIOR /
    # CLASSIFIER_PATH come from the environment (Dockerfile defaults).
    log(f"  inference: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=4 * 3600)
    log(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        raise RuntimeError(f"inference failed rc={r.returncode}: {r.stderr[-500:]}")
    pred_dir = pred_root / "stage2"
    if not pred_dir.exists():
        raise RuntimeError(f"no stage2/ dir under {pred_root}")
    return pred_dir


def run_upright(pred_dir: Path, upright_root: Path, raw_dir: Path) -> Path:
    """Run auto_upright on the predicted JPGs. Returns the directory of finals.

    auto_upright expects shoot folders under in-dir; we synthesize that layout
    and symlink the downloaded ARWs into a flat dir for EXIF lookup.
    """
    upright_root.mkdir(parents=True, exist_ok=True)
    in_dir = upright_root / "_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    # Flat ARW dir for AREM_ARW_FLAT_LAYOUT=1
    arw_layout = upright_root / "_arws"
    if arw_layout.exists():
        shutil.rmtree(arw_layout, ignore_errors=True)
    arw_layout.mkdir(parents=True, exist_ok=True)
    # Symlink every input image (ARW or standard format) so auto_upright
    # can find the source for EXIF-passthrough. exifread reads EXIF from
    # ARW + JPG + TIFF identically.
    for ext in ("ARW", "arw", "JPG", "jpg", "JPEG", "jpeg",
                "TIF", "tif", "TIFF", "tiff", "JXL", "jxl"):
        for src in raw_dir.glob(f"*.{ext}"):
            try:
                (arw_layout / src.name).symlink_to(src)
            except FileExistsError:
                pass
    shoot_in = in_dir / "shoot"
    if shoot_in.exists():
        shutil.rmtree(shoot_in)
    shoot_in.mkdir(parents=True)
    n_in = 0
    for j in pred_dir.glob("*.jpg"):
        # run_pipeline.py emits '<stem>_stage2-{int|ext}.jpg'. auto_upright
        # already strips '_stage2-int'/'_stage2-ext' (see _mid_stem_from_input),
        # so we copy as-is and let it produce clean output names.
        target = shoot_in / j.name
        shutil.copy2(j, target)
        n_in += 1
    if n_in == 0:
        # Surface why the pipeline produced no output. run_pipeline.py
        # writes _meta.json with both per-triplet skip reasons (e.g.
        # lens_excluded, demosaic_fail) and group-grouping failures
        # (e.g. no valid bracket window). Either path produces 0 JPGs;
        # show whichever bucket has data so the dashboard error category
        # is actually actionable.
        import json as _json
        meta_path = pred_dir.parent / "_meta.json"
        diag = ""
        n_arws = n_triplets = 0
        if meta_path.is_file():
            try:
                meta = _json.loads(meta_path.read_text())
                n_arws = meta.get("n_arws", 0)
                n_triplets = meta.get("n_triplets", 0)
                from collections import Counter
                skips = [t.get("skip", "") for t in meta.get("triplets", []) if t.get("skip")]
                if skips:
                    c = Counter(skips)
                    diag = "; ".join(f"{n}× {r}" for r, n in c.most_common())
                else:
                    grp = meta.get("group_failures", [])
                    if grp:
                        c = Counter(g.get("reason", "?") for g in grp)
                        diag = "no valid brackets — " + "; ".join(
                            f"{n}× {r}" for r, n in c.most_common())
            except Exception:
                pass
        if n_triplets == 0:
            msg = (f"no triplets formed from {n_arws} input files; "
                   f"check shooting order / EXIF EV tags")
        else:
            msg = "no predicted JPGs — every triplet was skipped/failed"
        if diag:
            msg += f" ({diag})"
        raise RuntimeError(msg)

    cmd = [PYTHON_BIN, str(UPRIGHT_REPO / "auto_upright.py"),
           "--in-dir", str(in_dir),
           "--out-dir", str(upright_root / "_out"),
           "--workers", "8"]
    log(f"  upright: {' '.join(cmd)}")
    upright_env = os.environ.copy()
    # Tell auto_upright where to find the ARWs we downloaded for this job.
    upright_env["AREM_ARW_ROOT"] = str((upright_root / "_arws").resolve())
    upright_env["AREM_ARW_FLAT_LAYOUT"] = "1"
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2 * 3600,
                       env=upright_env)
    log(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        raise RuntimeError(f"upright failed rc={r.returncode}: {r.stderr[-500:]}")
    out = upright_root / "_out" / "shoot"
    if not out.exists():
        raise RuntimeError(f"no upright output at {out}")
    return out


def upload_outputs(local_dir: Path, dropbox_job_path: str) -> int:
    dst = f"{dropbox_job_path}/{DROPBOX_OUTPUT_FOLDER}"
    r = rclone(["copy", str(local_dir), dst,
                "--include", "*.jpg",
                "--transfers", "8", "--progress=false"])
    if r.returncode != 0:
        raise RuntimeError(f"rclone upload failed rc={r.returncode}: {r.stderr[-300:]}")
    return len(list(local_dir.glob("*.jpg")))


def download_manual_inputs(inputs: list[dict], local_raw_dir: Path) -> tuple[int, int]:
    """Manual-upload ingest: fetch each presigned R2 URL into local_raw_dir.

    inputs: [{ key, name, url }, ...] from claim response.
    Returns (file_count, total_bytes). Filename comes from `name` so
    EXIF-driven find_triplets sees the same names the operator dropped.
    """
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    count = 0
    for i, item in enumerate(inputs, 1):
        name = item.get("name") or f"file_{i:04d}"
        url = item["url"]
        dest = local_raw_dir / name
        log(f"  manual: fetch {i}/{len(inputs)} {name}")
        total += _http_download(url, dest)
        count += 1
    return count, total


def upload_manual_outputs(local_dir: Path, output_prefix: str) -> tuple[int, list[str]]:
    """Manual-upload egress: PUT each *.jpg under local_dir to R2 via
    a presigned URL minted by the dashboard. Returns (count, keys).
    """
    jpgs = sorted(local_dir.glob("*.jpg"))
    keys: list[str] = []
    for p in jpgs:
        signed = _get(f"/api/jobs/claim?outputUploadFor={urllib.parse.quote(output_prefix, safe='/')}"
                      f"&filename={urllib.parse.quote(p.name)}")
        _http_put(signed["url"], p)
        keys.append(signed["key"])
    return len(jpgs), keys


# ---- per-job orchestrator ----
def process_job(job: dict) -> dict:
    job_id = job["id"]
    stored_path = job["dropboxPath"]
    manual = job.get("manualUpload")  # None for normal Dropbox jobs

    work = WORK_ROOT / job_id
    raw_dir = work / "raws"
    pred_root = work / "predictions"
    upright_root = work / "uprighted"

    t0 = time.time()
    if manual:
        inputs = manual.get("inputs", []) or []
        log(f"job {job_id}: manual upload, {len(inputs)} files")
        log("  [1/4] fetching inputs from R2")
        n_arws, total_bytes = download_manual_inputs(inputs, raw_dir)
        raw_subfolder = "manual_upload"
    else:
        dropbox_path = resolve_dropbox_source(stored_path)
        log(f"job {job_id}: {dropbox_path}")
        log("  [1/4] downloading ARWs")
        n_arws, total_bytes, raw_subfolder = download_raws(dropbox_path, raw_dir)

    if n_arws == 0:
        raise RuntimeError("no input files found")
    post_status(job_id, "processing", fileCount=n_arws, totalBytes=total_bytes)
    log(f"    downloaded {n_arws} inputs from {raw_subfolder} ({total_bytes/1e9:.2f} GB)")

    log("  [2/4] inference (rawpy → photomatix → restormer)")
    pred_dir = run_inference(raw_dir, pred_root)

    log("  [3/4] auto-upright + EXIF/branding")
    upright_dir = run_upright(pred_dir, upright_root, raw_dir)

    if manual:
        output_prefix = manual.get("outputPrefix") or f"manual_uploads/{job_id}/outputs/"
        log(f"  [4/4] uploading outputs to R2 ({output_prefix})")
        n_jpg, output_keys = upload_manual_outputs(upright_dir, output_prefix)
    else:
        log(f"  [4/4] uploading to {DROPBOX_OUTPUT_FOLDER}")
        n_jpg = upload_outputs(upright_dir, dropbox_path)
        output_keys = None

    # Read _meta.json from the inference step to surface grouping failures
    # and peak VRAM. Each anchor that couldn't be paired is a data-hygiene
    # issue. Peak VRAM tells us how much sensor-resolution headroom we have.
    import json as _json
    grouping_warnings: list[dict] = []
    peak_vram_gb: float | None = None
    gpu_total_gb: float | None = None
    meta_path = pred_root / "_meta.json"
    if meta_path.is_file():
        try:
            meta = _json.loads(meta_path.read_text())
            grouping_warnings = meta.get("group_failures", []) or []
            peak_vram_gb = meta.get("peak_vram_gb")
            gpu_total_gb = meta.get("gpu_total_gb")
        except Exception:
            pass

    runtime_sec = round(time.time() - t0, 1)
    log(f"  done. {n_jpg} JPGs in {runtime_sec}s; {len(grouping_warnings)} grouping warnings"
        + (f"; peak_vram={peak_vram_gb} GB / {gpu_total_gb} GB" if peak_vram_gb else ""))

    # ── Upload 512px thumbnails to R2 + report per-image classification ─────
    # run_pipeline.py wrote per-image classification into _meta.json and
    # 512px thumbnails into <pred_root>/_thumbnails/. Push thumbs to
    # r2:arem-training-data/image-thumbnails/<jobId>/<stem>.jpg, then
    # POST one classification record per image to the dashboard.
    # All best-effort — a thumbnail/classification failure must NOT
    # prevent the job from being reported done.
    try:
        thumbs_dir = pred_root / "_thumbnails"
        triplets = (meta or {}).get("triplets", []) if "meta" in locals() else []
        if thumbs_dir.is_dir() and triplets:
            r2_prefix = f"r2:arem-training-data/image-thumbnails/{job_id}"
            r2_upload = rclone(
                ["copy", str(thumbs_dir), r2_prefix,
                 "--transfers", "8", "--checkers", "8"],
                timeout=600,
            )
            if r2_upload.returncode != 0:
                log(f"  WARN thumbnail upload rc={r2_upload.returncode}: {r2_upload.stderr[-200:]}")
            n_cls = 0
            for t in triplets:
                stem = t.get("stem")
                cls = t.get("classification")
                if not stem or not cls:
                    continue
                thumb_key = f"image-thumbnails/{job_id}/{stem}.jpg"
                try:
                    _post("/api/internal/image-classification", {
                        "jobId": job_id,
                        "midStem": stem,
                        "isInteriorWorker": cls.get("isInterior"),
                        "roomTypeWorker": cls.get("roomType"),
                        "roomConfidenceWorker": cls.get("roomConfidence"),
                        "classifierModelVersions": cls.get("modelVersions"),
                        "thumbnailR2Path": thumb_key if (thumbs_dir / f"{stem}.jpg").is_file() else None,
                    })
                    n_cls += 1
                except Exception as e:
                    log(f"  WARN classification POST for {stem}: {str(e)[:200]}")
            log(f"  classification: posted {n_cls}/{len(triplets)} records to dashboard")
    except Exception as e:
        log(f"  WARN thumbnail/classification step: {str(e)[:200]}")

    # Cleanup local scratch — keep raws if it failed earlier (won't reach here)
    try:
        shutil.rmtree(work)
    except Exception:
        pass

    result = {
        "jpegCount": n_jpg,
        "rawCount": n_arws,
        "totalBytes": total_bytes,
        "runtimeSec": runtime_sec,
        "rawSubfolder": raw_subfolder,
        "outputFolder": (
            manual.get("outputPrefix") if manual
            else f"{stored_path}/{DROPBOX_OUTPUT_FOLDER}"
        ),
        "groupingWarnings": grouping_warnings,
        "peakVramGb": peak_vram_gb,
        "gpuTotalGb": gpu_total_gb,
    }
    if manual and output_keys:
        result["r2OutputKeys"] = output_keys
    return result


def loop_once() -> bool:
    """Returns True if a job was processed, False if queue empty."""
    job = claim_job()
    if not job:
        return False
    job_id = job["id"]
    try:
        result = process_job(job)
        post_status(job_id, "done", output=result, fileCount=result["rawCount"],
                    totalBytes=result["totalBytes"])
        return True
    except Exception as e:
        tb = traceback.format_exc()
        log(f"ERROR on {job_id}: {e}\n{tb}")
        try:
            post_status(job_id, "error", errorMessage=f"{type(e).__name__}: {e}")
        except Exception as e2:
            log(f"  (also failed to report error to dashboard: {e2})")
        return True  # we did try to process, just hit an error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="claim and process at most one job, then exit")
    ap.add_argument("--idle-sleep", type=int, default=IDLE_SLEEP,
                    help="seconds to wait between polls when queue empty")
    args = ap.parse_args()

    if not WORKER_TOKEN:
        sys.exit("ERROR: WORKER_TOKEN env var is required")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"starting; dashboard={DASHBOARD_URL}; once={args.once}")

    while True:
        try:
            processed = loop_once()
        except Exception as e:
            log(f"poll loop error: {e}")
            processed = False

        if args.once:
            return

        time.sleep(0 if processed else args.idle_sleep)


if __name__ == "__main__":
    main()
