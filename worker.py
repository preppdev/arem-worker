"""AREM editing pipeline worker.

Polls the dashboard /api/jobs/claim endpoint for queued shoots, pulls ARWs
from Dropbox, runs the production pipeline (rawpy → NAFNet Stage 1 →
classifier → NAFNet Stage 2 → auto-upright + EXIF), uploads the finished
JPEGs back to Dropbox under /08-Test-Edit, and reports status to the
dashboard.

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

import requests  # type: ignore  # for Cloudflare Images multipart upload

from pipeline import cc_ingest  # type: ignore  # AREM CC media-ingest POST
from pipeline import pretagger  # type: ignore  # local arem-pretagger inference client

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
# Set UPLOAD_STAGE1=1 to upload each completed job's Stage-1 NAFNet
# outputs to R2 (training-stage1/<jobId>/<midStem>.jpg) for downstream
# fine-tuning training-pair assembly. Stage 1 outputs are produced by
# run_pipeline.py at <pred_root>/stage1/ and otherwise discarded with
# the scratch dir.
UPLOAD_STAGE1 = os.environ.get("UPLOAD_STAGE1", "") in ("1", "true", "yes")

# Production-output bucket. After every Dropbox upload succeeds, the
# worker also pushes each final JPG to
#   r2:<R2_OUTPUT_BUCKET>/tours/<jobId>/photos/<NNNN>-<midStem>.jpg
# and POSTs (sortOrder, productionR2Path, cfImageId) per-image back to
# the dashboard so the virtual-tour platform + media-delivery surfaces
# can read from R2 (archive) + Cloudflare Images (delivery CDN).
# Empty R2_OUTPUT_BUCKET disables the entire dual-write pathway.
R2_OUTPUT_BUCKET = os.environ.get("R2_OUTPUT_BUCKET", "")

# Cloudflare Images upload — secondary delivery surface. Each upload
# returns a cfImageId; downstream consumers render via
#   https://imagedelivery.net/<account_hash>/<cfImageId>/<variant>
# (variants already provisioned: w384/w640/w828/w1080/w1200/w1920/w2400).
# Empty CLOUDFLARE_API_TOKEN disables only the CF Images upload (R2
# dual-write still runs).
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

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
# Local cache for Stage 3 polish checkpoints fetched on-demand from R2.
# Keyed by R2 key (full path under arem-training-data) to ensure
# different ckpts coexist. Wiped only when the box restarts; we never
# evict — total footprint stays small (one ~250-450 MB ckpt per style).
POLISH_CACHE_ROOT = Path(os.environ.get("POLISH_CACHE_ROOT",
    str(Path.home() / "polish-cache")))
POLISH_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
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


# ---- Stage 3 polish ckpt resolution ----
def _get_settings() -> dict:
    """Fetch the merged AppSetting JSON from the dashboard. Returns the
    inner `settings` map or an empty dict on transport failure (caller
    treats missing keys as "Stage 3 off")."""
    try:
        r = _get("/api/internal/settings")
        return r.get("settings") or {}
    except Exception as e:
        log(f"  WARN /api/internal/settings fetch failed: {str(e)[:200]} — "
            f"assuming Stage 3 polish off")
        return {}


def _ensure_polish_ckpt(r2_key: str) -> Path | None:
    """Download a polish checkpoint from R2 into POLISH_CACHE_ROOT if
    missing locally, return the local path. Returns None if the
    download fails (caller skips Stage 3 for that route).
    """
    # Cache layout mirrors the R2 key so different ckpts coexist and the
    # parent dir name (used as stage3Variant) is preserved.
    local = POLISH_CACHE_ROOT / r2_key
    if local.is_file() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    src = f"r2:arem-training-data/{r2_key}"
    log(f"  Stage 3: fetching {src} → {local}")
    try:
        r = subprocess.run(
            ["rclone", "copyto", src, str(local), "--s3-no-check-bucket"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            log(f"  WARN polish ckpt rclone rc={r.returncode}: "
                f"{r.stderr[-200:]}")
            return None
        if not local.is_file() or local.stat().st_size == 0:
            log(f"  WARN polish ckpt empty after fetch: {local}")
            return None
        return local
    except Exception as e:
        log(f"  WARN polish ckpt fetch failed: {str(e)[:200]}")
        return None


def resolve_polish_ckpts(style: str | None) -> dict[str, Path | None]:
    """Resolve Stage 3 polish checkpoints for the given style.

    Looks up AppSetting `polish.ckpts` (a JSON map of
    styleName -> {exterior, interior}), falling back to
    `polish.defaultStyle` when `style` is null. Downloads each referenced
    R2 key to POLISH_CACHE_ROOT on first use. Returns
    {"interior": Path|None, "exterior": Path|None} — a None value means
    "skip Stage 3 for this route".
    """
    settings = _get_settings()
    ckpts = settings.get("polish.ckpts") or {}
    default_style = settings.get("polish.defaultStyle") or "default"
    effective_style = style or default_style
    entry = ckpts.get(effective_style) or ckpts.get(default_style) or {}
    out: dict[str, Path | None] = {"interior": None, "exterior": None}
    for route in ("interior", "exterior"):
        key = entry.get(route)
        if not key:
            continue
        path = _ensure_polish_ckpt(key)
        out[route] = path
    if out["interior"] or out["exterior"]:
        log(f"  Stage 3 style='{effective_style}' resolved: "
            f"ext={out['exterior'].name if out['exterior'] else 'off'} "
            f"int={out['interior'].name if out['interior'] else 'off'}")
    else:
        log(f"  Stage 3 style='{effective_style}': no ckpts (Stage 3 disabled)")
    return out


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


def run_inference(local_raw_dir: Path, pred_root: Path,
                  polish_ckpts: dict[str, Path | None] | None = None,
                  skip_mid_stems: set[str] | None = None) -> Path:
    """Run the canonical Stage 1 (NAFNet) + Stage 2 (NAFNet routed) +
    Stage 3 (NAFNet polish, optional) pipeline.

    Calls pipeline/run_pipeline.py via subprocess to keep deps isolated.
    `polish_ckpts` is the resolved {"interior", "exterior"} dict from
    resolve_polish_ckpts(); None values disable Stage 3 for that route.
    `skip_mid_stems` is the make-up optimisation: midStems whose JPGs
    the parent already produced are excluded from the inference loop,
    saving the Stage 1/2/3 GPU work. Empty/None = full reprocess.

    Returns the directory containing the final JPGs (still named
    <stem>_stage2-{int|ext}.jpg — Stage 3 overwrites Stage 2 in place
    so downstream paths don't change).
    """
    pred_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN, str(AREM_REPO / "run_pipeline.py"),
        "--raw-dir", str(local_raw_dir),
        "--output-dir", str(pred_root),
    ]
    # Stage 3 polish ckpts — pass via flags so the subprocess doesn't
    # also need to talk to the dashboard. Omitting a flag = no polish on
    # that route.
    if polish_ckpts:
        if polish_ckpts.get("interior"):
            cmd += ["--polish-interior-ckpt", str(polish_ckpts["interior"])]
        if polish_ckpts.get("exterior"):
            cmd += ["--polish-exterior-ckpt", str(polish_ckpts["exterior"])]
    # Make-up skip set. Comma-separated to match the CLI flag's parsing.
    if skip_mid_stems:
        cmd += ["--skip-mid-stems", ",".join(sorted(skip_mid_stems))]
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
        #
        # Message must accurately describe WHAT was detected and avoid
        # blaming the input when the cause is worker-side strictness
        # (the makeup grouper rejecting brackets the parent accepted is
        # the canonical example). Specifically:
        #   - "no triplets formed" + skipped_stems non-empty
        #     → this is a makeup retry; the parent already produced
        #       output from these files. Phrase as a worker-side decision.
        #   - "no triplets formed" + no skipped_stems
        #     → genuine first-pass failure; report cluster + reason counts.
        #   - "triplets formed but all skipped"
        #     → Stage 1/2/3 errors on otherwise-valid brackets.
        import json as _json
        meta_path = pred_dir.parent / "_meta.json"
        diag = ""
        n_arws = n_triplets = 0
        n_groups_attempted = 0
        is_makeup_retry = False
        n_skipped_in_makeup = 0
        if meta_path.is_file():
            try:
                meta = _json.loads(meta_path.read_text())
                n_arws = meta.get("n_arws", 0)
                n_triplets = meta.get("n_triplets", 0)
                n_groups_attempted = len(meta.get("group_failures", []))
                skipped = meta.get("skipped_stems", []) or []
                is_makeup_retry = bool(skipped)
                n_skipped_in_makeup = len(skipped)
                from collections import Counter
                skips = [t.get("skip", "") for t in meta.get("triplets", [])
                         if t.get("skip")]
                if skips:
                    c = Counter(skips)
                    diag = "; ".join(f"{n}× {r}" for r, n in c.most_common(4))
                else:
                    grp = meta.get("group_failures", [])
                    if grp:
                        c = Counter(g.get("reason", "?") for g in grp)
                        diag = "; ".join(f"{n}× {r}" for r, n in c.most_common(4))
            except Exception:
                pass

        if n_triplets == 0:
            if is_makeup_retry:
                # The parent run already produced output from these files.
                # Naming it explicitly: the grouper is the bottleneck, not
                # the input.
                msg = (
                    f"makeup retry produced no triplets: the bracket grouper "
                    f"rejected all {n_groups_attempted} candidate bracket "
                    f"window(s) from {n_arws} input frame(s). "
                    f"Parent job already produced output for "
                    f"{n_skipped_in_makeup} of these frames; this retry's "
                    f"stricter grouping rules found no recoverable brackets "
                    f"in the remainder. This is a worker-side regression, "
                    f"not a shooting-order or EXIF issue with the photos."
                )
            else:
                msg = (
                    f"bracket grouper rejected all {n_groups_attempted} "
                    f"candidate bracket window(s) from {n_arws} input "
                    f"frame(s) — no valid (under/mid/over) sequence found. "
                    f"Likely cause: bracketing pattern in EXIF EV doesn't "
                    f"match the detector's expected layout."
                )
        else:
            n_triplet_total = len(meta.get("triplets", [])) if meta_path.is_file() else n_triplets
            msg = (
                f"all {n_triplet_total} formed triplets were skipped or "
                f"failed during Stage 1/2/3 — no output JPGs produced."
            )
        if diag:
            msg += f" Top reasons: {diag}."
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


def _strip_stage2_suffix(stem: str) -> str:
    """Strip the _stage2-int / _stage2-ext suffix from a worker-emitted
    JPG basename to recover the bare midStem that ImageReview joins on."""
    for suffix in ("_stage2-int", "_stage2-ext"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _cf_images_upload(local_path: Path, *, job_id: str, mid_stem: str,
                      sort_order: int) -> str | None:
    """Upload one JPG to Cloudflare Images. Returns the cfImageId on
    success, None on failure. Failures are logged but never raise — CF
    Images is the delivery copy, not the durable one (R2 already
    succeeded by the time we get here)."""
    if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
        return None
    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{CLOUDFLARE_ACCOUNT_ID}/images/v1")
    meta = json.dumps({
        "src": "arem-worker",
        "jobId": job_id,
        "midStem": mid_stem,
        "sortOrder": sort_order,
    })
    try:
        with open(local_path, "rb") as f:
            files = {
                "file": (local_path.name, f, "image/jpeg"),
                "metadata": (None, meta, "application/json"),
            }
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
                files=files,
                timeout=60,
            )
    except Exception as e:
        log(f"  WARN CF Images upload {mid_stem}: {str(e)[:200]}")
        return None
    if resp.status_code != 200:
        log(f"  WARN CF Images upload {mid_stem}: HTTP {resp.status_code} "
            f"{resp.text[:200]}")
        return None
    try:
        return resp.json()["result"]["id"]
    except Exception as e:
        log(f"  WARN CF Images parse {mid_stem}: {str(e)[:200]}")
        return None


def _enqueue_sky_swap_for_shoot(
    *,
    production_records: list[dict],
    triplets: list[dict],
    dropbox_path: str,
    job_id: str,
) -> None:
    """Queue one sky-swap job per exterior frame in the just-finished shoot.

    All frames in the shoot get the SAME plate so the deliverable folder
    reads as one consistent listing. Plate is picked from the approved
    SkyImage library — random per shoot, seeded by job_id for stability
    across retries (a retried job picks the same plate).

    Best-effort: any failure is raised to the caller, which logs and
    moves on.
    """
    import hashlib as _hashlib
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    # Stem → isInterior from the run's classifications. Falls back to
    # treating unknown as interior (skip) — safer than mistakenly
    # rewriting an indoor shot.
    is_exterior_by_stem: dict[str, bool] = {}
    for t in triplets:
        stem = t.get("stem")
        cls = t.get("classification") or {}
        if stem is not None and "isInterior" in cls:
            is_exterior_by_stem[stem] = cls.get("isInterior") is False

    exterior_recs = [
        r for r in production_records
        if r.get("midStem") and is_exterior_by_stem.get(r["midStem"], False)
    ]
    if not exterior_recs:
        log("  sky-swap enqueue: 0 exteriors in shoot, skipping")
        return

    # Pull approved plates from the dashboard.
    dashboard_url = _os.environ.get(
        "DASHBOARD_URL", "https://arem-editing-dashboard.vercel.app"
    ).rstrip("/")
    worker_token = _os.environ.get("WORKER_TOKEN", "")
    try:
        req = _urlreq.Request(
            f"{dashboard_url}/api/internal/sky-library?status=approved&limit=60",
            headers={"x-worker-token": worker_token},
        )
        with _urlreq.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except (_urlerr.URLError, _urlerr.HTTPError) as e:
        log(f"  sky-swap enqueue: list-plates fetch failed ({e}); skipping")
        return
    plates = [p for p in (data.get("items") or [])
              if p.get("status") == "approved" and p.get("r2Path")]
    if not plates:
        log("  sky-swap enqueue: 0 approved plates in library, skipping")
        return

    # Pick one plate, seeded by job_id so retries are stable.
    seed = int(_hashlib.sha256(job_id.encode()).hexdigest()[:8], 16)
    plate = plates[seed % len(plates)]
    log(f"  sky-swap enqueue: shoot plate "
        f"{plate.get('category', '?')}/{plate['id'][:8]} for "
        f"{len(exterior_recs)} exterior frames")

    # Resolve imageReviewId per stem (best-effort: the dashboard's
    # classification POST has already created the rows, but we don't have
    # ids handy here; the daemon can update by jobId+midStem if needed,
    # which we include).
    enqueued = 0
    for rec in exterior_recs:
        stem = rec["midStem"]
        src_key = rec.get("productionR2Path")
        if not src_key:
            continue
        rid = f"{job_id}-{stem}-{int(time.time())}"
        request = {
            "id": rid,
            "sourceBucket": "arem-production-edit-jobs",
            "sourceR2Path": src_key,
            "plateBucket": "arem-training-data",
            "plateR2Path": plate["r2Path"],
            "skyImageId": plate["id"],
            "dropboxJobPath": dropbox_path,
            "dropboxSubfolder": "08-Test-Edit/sky-swap",
            "stem": stem,
            "jobId": job_id,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        req_local = Path(f"/tmp/sky_swap_req_{rid}.json")
        req_local.write_text(_json.dumps(request))
        try:
            r = _subprocess.run(
                ["rclone", "copyto", str(req_local),
                 f"r2:arem-training-data/sky-swap-jobs/requests/{rid}.json"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                enqueued += 1
            else:
                log(f"  sky-swap enqueue: rclone fail {rid}: {r.stderr.strip()[:160]}")
        finally:
            try:
                req_local.unlink()
            except OSError:
                pass
    log(f"  sky-swap enqueue: queued {enqueued}/{len(exterior_recs)} jobs")


def upload_outputs_production_r2(
    local_dir: Path,
    job_id: str,
    *,
    sort_order_start: int = 1,
    skip_mid_stems: set[str] | None = None,
) -> list[dict]:
    """Push the final JPGs to the production output bucket + Cloudflare
    Images and return one record per uploaded image.

    R2 key shape (new spec):
      tours/<jobId>/photos/<sortOrder:04d>-<midStem>.jpg

    Args:
      sort_order_start: 1-based starting ordinal. Make-up jobs pass the
        parent's next-available value here so new stems land after the
        existing ones.
      skip_mid_stems: stems already covered on the parent (make-up only);
        we don't re-upload these even though inference re-ran on them.
        Saves Cloudflare Images quota and avoids cfImageId duplicates.

    Returns: list of {midStem, sortOrder, productionR2Path, cfImageId}.
    Best-effort: an exception in this function is reported up so the
    caller can log+swallow; the job itself stays "done" since Dropbox
    already delivered.
    """
    if not R2_OUTPUT_BUCKET:
        return []
    skip = skip_mid_stems or set()
    # Sort by stripped mid-stem so sortOrder follows alphabetical order
    # of the FINAL frame names; the make-up's filter excludes anything
    # that already has a productionR2Path on the parent.
    jpgs = sorted(local_dir.glob("*.jpg"), key=lambda p: _strip_stage2_suffix(p.stem))
    new_jpgs = [p for p in jpgs if _strip_stage2_suffix(p.stem) not in skip]
    if not new_jpgs:
        if skip:
            log(f"    production R2: all {len(jpgs)} stems already covered — nothing to upload")
        return []

    records: list[dict] = []
    for offset, p in enumerate(new_jpgs):
        idx = sort_order_start + offset
        mid_stem = _strip_stage2_suffix(p.stem)
        target_name = f"{idx:04d}-{mid_stem}.jpg"
        r2_key = f"tours/{job_id}/photos/{target_name}"
        r2_url = f"r2:{R2_OUTPUT_BUCKET}/{r2_key}"
        cp = rclone(["copyto", str(p), r2_url,
                     "--progress=false"], timeout=180)
        if cp.returncode != 0:
            log(f"  WARN r2 production copy {mid_stem}: rc={cp.returncode} "
                f"{cp.stderr[-200:].strip()}")
            continue
        cf_id = _cf_images_upload(p, job_id=job_id, mid_stem=mid_stem,
                                  sort_order=idx)
        records.append({
            "midStem": mid_stem,
            "sortOrder": idx,
            "productionR2Path": r2_key,
            "cfImageId": cf_id,
        })
    return records


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

    # Make-up jobs (source=makeup, created by the watchdog) carry a
    # synthetic dropboxPath. The claim endpoint resolves it for us and
    # returns:
    #   makeup = {
    #     parentJobId,            # all per-image artifacts go here
    #     inputPathOverride,      # real Dropbox folder to download from
    #     alreadyDoneMidStems[],  # skip these at upload time
    #     nextSortOrder,          # append-after-highest for new R2 keys
    #     attempt,
    #   }
    # The make-up's own job.id stays the status-update target so we
    # can track the audit trail; everything else routes to parentJobId.
    makeup = job.get("makeup") or None
    # Source=makeup MUST come with an unwrap envelope from the claim
    # endpoint. If the envelope is missing or only contains an error
    # field (e.g. a Vercel function instance ran stale claim code
    # during a deploy roll-out), refuse to process — falling through
    # to the else: branch would use the synthetic _makeup/<parent>/<N>
    # path and trash the Job with a misleading "Dropbox folder not
    # found" error.
    if job.get("source") == "makeup":
        if not makeup or makeup.get("error") or not makeup.get("inputPathOverride"):
            raise RuntimeError(
                f"make-up Job claimed without a valid envelope "
                f"(got makeup={makeup!r}). "
                f"Dashboard claim endpoint may have served stale code; retry."
            )
    effective_job_id = makeup["parentJobId"] if makeup else job_id
    skip_mid_stems: set[str] = (
        set(makeup.get("alreadyDoneMidStems") or []) if makeup else set()
    )
    sort_order_start = (makeup.get("nextSortOrder") or 1) if makeup else 1

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
    elif makeup:
        # Pull RAWs from the parent's real Dropbox folder. Output also
        # lands in the parent's 08-Test-Edit (rclone copy is content-
        # aware, so re-uploading existing JPGs is a no-op).
        dropbox_path = resolve_dropbox_source(makeup["inputPathOverride"])
        log(f"job {job_id}: make-up #{makeup.get('attempt')} of parent "
            f"{effective_job_id}; src={dropbox_path}; "
            f"skip={len(skip_mid_stems)} already-done stems; "
            f"nextSortOrder={sort_order_start}")
        log("  [1/4] downloading ARWs")
        n_arws, total_bytes, raw_subfolder = download_raws(dropbox_path, raw_dir)
    else:
        dropbox_path = resolve_dropbox_source(stored_path)
        log(f"job {job_id}: {dropbox_path}")
        log("  [1/4] downloading ARWs")
        n_arws, total_bytes, raw_subfolder = download_raws(dropbox_path, raw_dir)

    if n_arws == 0:
        raise RuntimeError("no input files found")
    post_status(job_id, "processing", fileCount=n_arws, totalBytes=total_bytes)
    log(f"    downloaded {n_arws} inputs from {raw_subfolder} ({total_bytes/1e9:.2f} GB)")

    log("  [2/4] inference (rawpy → NAFNet stage1 → NAFNet stage2 → Stage 3 polish)")
    # Stage 3 polish ckpt selection. Job.polishStyle (set per-shoot;
    # eventually wired to a customer-facing style picker) overrides the
    # AppSetting default. Resolver downloads ckpts to local cache on
    # first use, returns {} if /api/settings is unreachable so the
    # pipeline cleanly degrades to "no Stage 3".
    polish_ckpts = resolve_polish_ckpts(job.get("polishStyle"))
    pred_dir = run_inference(raw_dir, pred_root,
                             polish_ckpts=polish_ckpts,
                             skip_mid_stems=skip_mid_stems)

    log("  [3/4] auto-upright + EXIF/branding")
    upright_dir = run_upright(pred_dir, upright_root, raw_dir)

    if manual:
        output_prefix = manual.get("outputPrefix") or f"manual_uploads/{job_id}/outputs/"
        log(f"  [4/4] uploading outputs to R2 ({output_prefix})")
        n_jpg, output_keys = upload_manual_outputs(upright_dir, output_prefix)
        production_records: list[dict] = []
    else:
        log(f"  [4/4] uploading to {DROPBOX_OUTPUT_FOLDER}")
        # Dropbox upload is content-aware (rclone copy skips identical
        # files), so make-ups re-uploading already-done stems is a no-op.
        n_jpg = upload_outputs(upright_dir, dropbox_path)
        output_keys = None

        # Dual-write the same JPGs to the production output bucket +
        # Cloudflare Images. For make-up jobs:
        #   - effective_job_id = parent's id (R2 prefix routes there)
        #   - skip_mid_stems = stems already covered on the parent
        #   - sort_order_start = parent's next-available sortOrder
        # so the make-up's new stems append cleanly. Best-effort: an
        # exception here is logged as a warning but does NOT roll back
        # the Dropbox upload or fail the job.
        production_records = []
        if R2_OUTPUT_BUCKET:
            try:
                production_records = upload_outputs_production_r2(
                    upright_dir,
                    effective_job_id,
                    sort_order_start=sort_order_start,
                    skip_mid_stems=skip_mid_stems,
                )
                n_cf = sum(1 for r in production_records if r.get("cfImageId"))
                log(f"    production dual-write: {len(production_records)} files → "
                    f"r2:{R2_OUTPUT_BUCKET}/tours/{effective_job_id}/photos/  "
                    f"cf_images={n_cf}/{len(production_records)}")
            except Exception as e:
                log(f"  WARN production dual-write: {str(e)[:200]}")

    # Read _meta.json from the inference step to surface grouping failures
    # and peak VRAM. Each anchor that couldn't be paired is a data-hygiene
    # issue. Peak VRAM tells us how much sensor-resolution headroom we have.
    import json as _json
    grouping_warnings: list[dict] = []
    peak_vram_gb: float | None = None
    gpu_total_gb: float | None = None
    n_triplets: int = 0
    meta_path = pred_root / "_meta.json"
    if meta_path.is_file():
        try:
            meta = _json.loads(meta_path.read_text())
            grouping_warnings = meta.get("group_failures", []) or []
            peak_vram_gb = meta.get("peak_vram_gb")
            gpu_total_gb = meta.get("gpu_total_gb")
            n_triplets = int(meta.get("n_triplets") or 0)
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
            # Make-up: route thumbnails + classification POSTs to the
            # parent's job id. Also skip stems that already exist on
            # the parent (we don't want to overwrite the reviewer-
            # corrected classification fields on already-finished rows).
            r2_prefix = f"r2:arem-training-data/image-thumbnails/{effective_job_id}"
            r2_upload = rclone(
                ["copy", str(thumbs_dir), r2_prefix,
                 "--transfers", "8", "--checkers", "8"],
                timeout=600,
            )
            if r2_upload.returncode != 0:
                log(f"  WARN thumbnail upload rc={r2_upload.returncode}: {r2_upload.stderr[-200:]}")
            # Map midStem -> production record so each per-image POST
            # can also carry the dual-write outputs (productionR2Path,
            # sortOrder, cfImageId).
            production_by_stem = {r["midStem"]: r for r in production_records}
            n_cls = 0
            n_skipped_existing = 0
            for t in triplets:
                stem = t.get("stem")
                cls = t.get("classification")
                if not stem or not cls:
                    continue
                if stem in skip_mid_stems:
                    n_skipped_existing += 1
                    continue
                thumb_key = f"image-thumbnails/{effective_job_id}/{stem}.jpg"
                prod = production_by_stem.get(stem) or {}
                try:
                    _post("/api/internal/image-classification", {
                        "jobId": effective_job_id,
                        "midStem": stem,
                        "isInteriorWorker": cls.get("isInterior"),
                        "roomTypeWorker": cls.get("roomType"),
                        "roomConfidenceWorker": cls.get("roomConfidence"),
                        "classifierModelVersions": cls.get("modelVersions"),
                        "thumbnailR2Path": thumb_key if (thumbs_dir / f"{stem}.jpg").is_file() else None,
                        "productionR2Path": prod.get("productionR2Path"),
                        "sortOrder": prod.get("sortOrder"),
                        "cfImageId": prod.get("cfImageId"),
                        # Stage 3 polish variant applied to this image
                        # (null = Stage 3 skipped on this route). Passed
                        # explicitly even when null so re-POSTs from the
                        # backfill path can correct a previously-set value.
                        "stage3Variant": cls.get("stage3Variant"),
                    })
                    n_cls += 1
                except Exception as e:
                    log(f"  WARN classification POST for {stem}: {str(e)[:200]}")
            log(f"  classification: posted {n_cls}/{len(triplets)} records to dashboard"
                + (f" (skipped {n_skipped_existing} already-done stems)" if n_skipped_existing else ""))
    except Exception as e:
        log(f"  WARN thumbnail/classification step: {str(e)[:200]}")

    # ── Pre-tag each deliverable via the local arem-pretagger service ──
    # Best-effort: if the pretagger is down or unconfigured (no
    # PRETAG_TOKEN), this whole block is a no-op and the reviewer just
    # doesn't see suggestions for these images. Never blocks the
    # pipeline. ImageReview.preTagConfidence is written from the
    # dashboard side via /api/internal/pretag-result.
    if pretagger.pretagger_enabled():
        try:
            n_pre_ok = 0
            n_pre_skip = 0
            for prod in production_records:
                stem = prod.get("midStem")
                if not stem:
                    n_pre_skip += 1
                    continue
                local_file = upright_dir / (
                    f"{stem}_stage2-int.jpg"
                    if (upright_dir / f"{stem}_stage2-int.jpg").is_file()
                    else f"{stem}_stage2-ext.jpg"
                )
                if not local_file.is_file():
                    n_pre_skip += 1
                    continue
                resp = pretagger.pretag_image(local_file)
                if not resp:
                    n_pre_skip += 1
                    continue
                payload = pretagger.to_dashboard_payload(effective_job_id, stem, resp)
                try:
                    _post("/api/internal/pretag-result", payload)
                    n_pre_ok += 1
                except Exception as e:
                    n_pre_skip += 1
                    log(f"  WARN pretag-result POST for {stem}: {str(e)[:200]}")
            log(f"  pretag: posted {n_pre_ok}/{len(production_records)} (skipped {n_pre_skip})")
        except Exception as e:
            log(f"  WARN pretag step: {str(e)[:200]}")
    else:
        log("  pretag: disabled (PRETAG_TOKEN not set)")

    # ── Sky quality analysis → conditional sky-swap ────────────────────
    # The auto-hook fires only when the shoot's exteriors actually need a
    # new sky (dull / overcast / hazy). Clear-blue and golden-hour shoots
    # are left alone. Per-shoot decision is in pipeline.sky_quality;
    # exterior locals from upright_dir.
    #
    # Override knobs (env):
    #   AREM_SKY_SWAP_FORCE=1  bypass the quality gate and trigger on
    #                          every shoot — useful for QA sweeps. Pair
    #                          with AREM_SKY_SWAP_DROPBOX_SUBFOLDER if
    #                          you want results in a separate folder.
    #   AREM_SKY_SWAP_OFF=1    never trigger sky-swap. Hard kill switch.
    #
    # Best-effort — failures here are logged and never block the job.
    if not manual and production_records:
        try:
            _force = os.environ.get("AREM_SKY_SWAP_FORCE", "").lower() in ("1", "true", "yes")
            _off   = os.environ.get("AREM_SKY_SWAP_OFF",   "").lower() in ("1", "true", "yes")
            if _off:
                log("  sky-swap: AREM_SKY_SWAP_OFF set — skipping")
            elif _force:
                log("  sky-swap: AREM_SKY_SWAP_FORCE set — bypassing quality gate, triggering on every shoot")
                _enqueue_sky_swap_for_shoot(
                    production_records=production_records,
                    triplets=(meta or {}).get("triplets", []),
                    dropbox_path=dropbox_path,
                    job_id=effective_job_id,
                )
            else:
                from pipeline.sky_quality import should_swap_shoot  # type: ignore
                # Map midStem → isExterior from triplets
                _is_ext_by_stem: dict[str, bool] = {}
                for t in (meta or {}).get("triplets", []) or []:
                    _stem = t.get("stem")
                    _cls = t.get("classification") or {}
                    if _stem is not None and "isInterior" in _cls:
                        _is_ext_by_stem[_stem] = _cls.get("isInterior") is False
                _ext_stems = [r["midStem"] for r in production_records
                               if r.get("midStem") and _is_ext_by_stem.get(r["midStem"], False)]
                _shoot_subdir = Path(dropbox_path).name if dropbox_path else None
                _ext_paths: list[Path] = []
                if _shoot_subdir:
                    for _ms in _ext_stems:
                        _p = upright_dir / _shoot_subdir / f"{_ms}.jpg"
                        if _p.is_file():
                            _ext_paths.append(_p)
                if _ext_paths:
                    _trigger, _summary = should_swap_shoot(_ext_paths)
                    log(f"  sky-quality: trigger={_trigger}  "
                        f"mean_dull={_summary.get('weighted_mean_dull')}  "
                        f"frac_high={_summary.get('frac_high_dull')}  "
                        f"n_with_sky={_summary.get('n_with_sky')}/{_summary.get('n_frames')}")
                    if _trigger:
                        _enqueue_sky_swap_for_shoot(
                            production_records=production_records,
                            triplets=(meta or {}).get("triplets", []),
                            dropbox_path=dropbox_path,
                            job_id=effective_job_id,
                        )
                    else:
                        log("  sky-swap: skipped (sky conditions look good)")
                else:
                    log("  sky-quality: no exterior locals — skipping analysis")
        except Exception as e:
            log(f"  WARN sky-quality / sky-swap step: {str(e)[:200]}")

    # ── POST per-shoot asset batch to AREM Command Center ──────────────
    # Only photos with both productionR2Path AND cfImageId qualify —
    # those are the two CC-required fields for kind=photo. Anything
    # missing either gets a later reconciliation pass. Best-effort:
    # transport / 4xx / 5xx are logged and silently swallowed.
    try:
        if production_records:
            classifier_by_stem = {
                t["stem"]: (t.get("classification") or {})
                for t in (meta or {}).get("triplets", [])
                if t.get("stem")
            } if "meta" in locals() else {}

            cc_assets = []
            for prod in production_records:
                if not prod.get("cfImageId") or not prod.get("productionR2Path"):
                    continue
                local_file = upright_dir / (f"{prod['midStem']}_stage2-int.jpg"
                                            if (upright_dir / f"{prod['midStem']}_stage2-int.jpg").is_file()
                                            else f"{prod['midStem']}_stage2-ext.jpg")
                width, height = cc_ingest.jpeg_dims(local_file) if local_file.is_file() else (None, None)
                size_bytes = local_file.stat().st_size if local_file.is_file() else None
                cls = classifier_by_stem.get(prod["midStem"], {})
                cc_assets.append({
                    "kind": "photo",
                    "sortOrder": prod["sortOrder"],
                    "r2Bucket": R2_OUTPUT_BUCKET,
                    "r2Key": prod["productionR2Path"],
                    "cfImageId": prod["cfImageId"],
                    "mimeType": "image/jpeg",
                    "width": width,
                    "height": height,
                    "sizeBytes": size_bytes,
                    "room": cls.get("roomType"),
                    "isHero": False,
                    "altText": None,
                    "caption": None,
                    "checksum": None,
                })
            if cc_assets:
                # CC's Shoot ↔ our Job.id are separate spaces — always
                # POST to /new with an ensureJob block. CC dedups via
                # editorJobId (most reliable, exact match on parent's
                # job id); falls back to dropboxPath; then address.
                # Make-ups use the parent's identity (editorJobId =
                # parent's id, dropboxPath = parent's real folder) so
                # the new assets attach to the existing CC shoot.
                ensure_job = cc_ingest.build_ensure_job(
                    editor_job_id=effective_job_id,
                    dropbox_path=(
                        makeup["inputPathOverride"] if makeup else job.get("dropboxPath")
                    ),
                    photographer=job.get("photographer"),
                    completed_at=(job.get("completedAt") if isinstance(job.get("completedAt"), str)
                                  else None),
                )
                result = cc_ingest.post_media(
                    assets=cc_assets, shoot_key="new", ensure_job=ensure_job,
                )
                if "error" in result:
                    log(f"  CC ingest: {result['error']}")
                else:
                    accepted = len(result.get("accepted") or [])
                    skipped = len(result.get("skipped") or [])
                    errors = len(result.get("errors") or [])
                    log(f"  CC ingest: shootId={result.get('shootId')} "
                        f"resolvedFrom={result.get('resolvedFrom')} "
                        f"accepted={accepted} skipped={skipped} errors={errors}")
    except Exception as e:
        log(f"  WARN CC ingest step: {str(e)[:200]}")

    # ── Upload Stage-1 NAFNet outputs to R2 (for fine-tune training pairs) ──
    # Gated by UPLOAD_STAGE1=1. run_pipeline.py wrote each pre-Stage-2
    # intermediate at <pred_root>/stage1/<midStem>_stage1.jpg. We strip
    # the _stage1 suffix on upload so R2 keys match the midStem convention
    # used by other backfilled artifacts (image-thumbnails/, vendor-mirrors/).
    # All best-effort: a Stage-1 upload failure must NOT prevent the job
    # from being reported done.
    if UPLOAD_STAGE1:
        try:
            stage1_dir = pred_root / "stage1"
            if stage1_dir.is_dir():
                renamed_dir = pred_root / "stage1_renamed"
                renamed_dir.mkdir(parents=True, exist_ok=True)
                n_renamed = 0
                for p in stage1_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
                        stem = p.stem
                        if stem.endswith("_stage1"):
                            stem = stem[: -len("_stage1")]
                        shutil.copyfile(p, renamed_dir / f"{stem}.jpg")
                        n_renamed += 1
                if n_renamed > 0:
                    # Make-ups: route stage1 R2 keys + DB posts to the
                    # parent's id so training-pair assembly joins
                    # cleanly with the parent's existing stage1 outputs.
                    r2_prefix = f"r2:arem-training-data/training-stage1/{effective_job_id}"
                    r = rclone(
                        ["copy", str(renamed_dir), r2_prefix,
                         "--transfers", "8", "--checkers", "8"],
                        timeout=900,
                    )
                    if r.returncode != 0:
                        log(f"  WARN stage1 upload rc={r.returncode}: {r.stderr[-200:]}")
                    else:
                        log(f"  stage1: uploaded {n_renamed} JPGs to R2")
                        # POST stage1R2Path per image — same endpoint as the
                        # classification step, partial-PATCH semantics
                        # mean only this field gets updated per row.
                        n_posted = 0
                        for p in renamed_dir.iterdir():
                            if not p.is_file():
                                continue
                            stem = p.stem
                            if stem in skip_mid_stems:
                                continue  # parent already has this stage1 row
                            try:
                                _post("/api/internal/image-classification", {
                                    "jobId": effective_job_id,
                                    "midStem": stem,
                                    "stage1R2Path": f"training-stage1/{effective_job_id}/{stem}.jpg",
                                })
                                n_posted += 1
                            except Exception as e:
                                log(f"  WARN stage1 POST for {stem}: {str(e)[:200]}")
                        log(f"  stage1: posted {n_posted}/{n_renamed} paths to dashboard")
        except Exception as e:
            log(f"  WARN stage1 upload step: {str(e)[:200]}")

    # Cleanup local scratch — keep raws if it failed earlier (won't reach here)
    try:
        shutil.rmtree(work)
    except Exception:
        pass

    result = {
        "jpegCount": n_jpg,
        "rawCount": n_arws,
        # Authoritative count of valid (under/mid/over) brackets the
        # grouper found in this shoot — the cap on physically-recoverable
        # output. The watchdog uses this in preference to ceil(fileCount/3)
        # to avoid creating doomed makeup retries on shoots where the
        # photographer simply didn't have enough valid brackets in the
        # input set (missing frames, bad EXIF EV, 2-frame fragments, etc.).
        "bracketsFound": n_triplets,
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
