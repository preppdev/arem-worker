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
RAW_SUBFOLDER_CANDIDATES = ["01-RAW-Photos", "01-RAW-Files"]
# Image formats we accept as bracket inputs. Raw formats hit rawpy + lensfun;
# everything else gets decoded directly via PIL (lens correction skipped).
ACCEPTED_EXTS = ["ARW", "arw", "JPG", "jpg", "JPEG", "jpeg",
                 "TIF", "tif", "TIFF", "tiff", "PNG", "png",
                 "JXL", "jxl", "WEBP", "webp"]


def _include_flags(exts: list[str]) -> list[str]:
    out = []
    for e in exts:
        out += ["--include", f"*.{e}"]
    return out


def download_raws(dropbox_remote_path: str, local_raw_dir: Path) -> tuple[int, int, str]:
    """Pull the RAW folder contents to local. Returns (file_count, bytes, used_subfolder).

    Accepts any standard image format (ARW + JPG + TIFF + PNG + JXL + WEBP).
    Tries known RAW subfolder names in order; uses whichever exists.
    """
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    include_flags = _include_flags(ACCEPTED_EXTS)
    last_err = ""
    for candidate in RAW_SUBFOLDER_CANDIDATES:
        src = f"{dropbox_remote_path}/{candidate}"
        ls = rclone(["lsf", src, *include_flags], timeout=60)
        if ls.returncode != 0 or not ls.stdout.strip():
            last_err = f"{candidate}: rc={ls.returncode} {ls.stderr[-200:]}"
            continue
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
        return len(files), total, candidate
    raise RuntimeError(f"no input images found under {dropbox_remote_path} "
                       f"(tried {RAW_SUBFOLDER_CANDIDATES}); last error: {last_err}")


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
        # writes _meta.json with skip reasons (e.g. lens_excluded for
        # FE 12-24mm GM — no lensfun profile available).
        import json as _json
        meta_path = pred_dir.parent / "_meta.json"
        skip_summary = ""
        if meta_path.is_file():
            try:
                meta = _json.loads(meta_path.read_text())
                skips = [t.get("skip", "") for t in meta.get("triplets", []) if t.get("skip")]
                if skips:
                    from collections import Counter
                    counts = Counter(skips)
                    skip_summary = "; ".join(f"{c}× {r}" for r, c in counts.most_common())
            except Exception:
                pass
        msg = f"no predicted JPGs — every triplet was skipped/failed."
        if skip_summary:
            msg += f" Reasons: {skip_summary}"
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


# ---- per-job orchestrator ----
def process_job(job: dict) -> dict:
    job_id = job["id"]
    stored_path = job["dropboxPath"]
    dropbox_path = resolve_dropbox_source(stored_path)
    log(f"job {job_id}: {dropbox_path}")

    work = WORK_ROOT / job_id
    raw_dir = work / "raws"
    pred_root = work / "predictions"
    upright_root = work / "uprighted"

    t0 = time.time()
    log("  [1/4] downloading ARWs")
    n_arws, total_bytes, raw_subfolder = download_raws(dropbox_path, raw_dir)
    if n_arws == 0:
        raise RuntimeError("no ARW files found")
    post_status(job_id, "processing", fileCount=n_arws, totalBytes=total_bytes)
    log(f"    downloaded {n_arws} ARWs from {raw_subfolder} ({total_bytes/1e9:.2f} GB)")

    log("  [2/4] inference (rawpy → photomatix → restormer)")
    pred_dir = run_inference(raw_dir, pred_root)

    log("  [3/4] auto-upright + EXIF/branding")
    upright_dir = run_upright(pred_dir, upright_root, raw_dir)

    log(f"  [4/4] uploading to {DROPBOX_OUTPUT_FOLDER}")
    n_jpg = upload_outputs(upright_dir, dropbox_path)

    # Diagnostic: also upload Stage 1 outputs to <output>/stage1/ when
    # UPLOAD_STAGE1_DIAG=1. Used to isolate which inference stage produces
    # high-res artifacts (NAFNet vs Restormer).
    if os.environ.get("UPLOAD_STAGE1_DIAG", "") in ("1", "true", "yes"):
        stage1_local = pred_root / "stage1"
        if stage1_local.is_dir():
            dst = f"{dropbox_path}/{DROPBOX_OUTPUT_FOLDER}/stage1"
            log(f"    diag: uploading {len(list(stage1_local.glob('*.jpg')))} stage1 JPGs to {dst}")
            r = rclone(["copy", str(stage1_local), dst,
                        "--include", "*.jpg",
                        "--transfers", "8", "--progress=false"])
            if r.returncode != 0:
                log(f"    diag: stage1 upload rc={r.returncode}: {r.stderr[-200:]}")

    # Read _meta.json from the inference step to surface grouping failures.
    # Each anchor that couldn't be paired into a complete bracket is a
    # data-hygiene issue the photographer / labeler should look at.
    import json as _json
    grouping_warnings: list[dict] = []
    meta_path = pred_root / "_meta.json"
    if meta_path.is_file():
        try:
            meta = _json.loads(meta_path.read_text())
            grouping_warnings = meta.get("group_failures", []) or []
        except Exception:
            pass

    runtime_sec = round(time.time() - t0, 1)
    log(f"  done. {n_jpg} JPGs in {runtime_sec}s; {len(grouping_warnings)} grouping warnings")

    # Cleanup local scratch — keep raws if it failed earlier (won't reach here)
    try:
        shutil.rmtree(work)
    except Exception:
        pass

    return {
        "jpegCount": n_jpg,
        "rawCount": n_arws,
        "totalBytes": total_bytes,
        "runtimeSec": runtime_sec,
        "rawSubfolder": raw_subfolder,
        "outputFolder": f"{stored_path}/{DROPBOX_OUTPUT_FOLDER}",
        "groupingWarnings": grouping_warnings,
    }


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
