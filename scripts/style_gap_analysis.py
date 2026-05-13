#!/usr/bin/env python3
"""
style_gap_analysis.py

Quantify the systematic style gap between our pipeline's 08-Test-Edit
outputs and AutoHDR's 05-Finished-Photos targets, across recent
production shoots.

For each sampled (ours, vendor) image pair, computes per-image stats
(luminance, contrast, per-channel means, R/G + B/G ratios for WB
temp/tint, saturation), then aggregates deltas across the whole
dataset. Output tells you whether the gap is a uniform offset (a flat
constant correction at end-of-pipeline closes it) or has significant
per-image variance (a small regression head is warranted).

Usage:
    # On the 3090 worker box, with creds loaded:
    set -a && . /etc/arem-worker.env && set +a
    python3 scripts/style_gap_analysis.py --jobs 30 --per-job 5

    # Or from a dev laptop with creds set the same way:
    set -a && . /tmp/db.env && . /tmp/vercel-prod.env && set +a
    python3 scripts/style_gap_analysis.py

Args:
    --jobs N         Sample the N most-recent done jobs       (default 30)
    --per-job M      Max paired frames per job                (default 5)
    --since DAYS     Only consider jobs done in last N days   (default 30)
    --csv FILE       Also write per-pair stats CSV
    --resize PX      Thumbnail side before stats (perf)       (default 600)

Notes:
- Pairing uses the same offset-histogram fuzzy match as
  app/api/review/[jobId]/route.ts (post-2026-05-12 fix that
  tie-breaks by smallest |offset|).
- Skips archived jobs and make-up children (parentJobId).
- Stats are computed on resized thumbnails — global metrics
  (mean/std/per-channel) are scale-invariant; this keeps the
  analyzer running in seconds even on 25MP source JPGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import psycopg2
from PIL import Image


# ─── Dropbox client (token refresh + list + content download) ───────────────

_TOK: dict[str, object] = {"access": os.environ.get("DROPBOX_ACCESS_TOKEN")}
_TOK_LOCK = threading.Lock()  # serialize refresh across worker threads


def _refresh_token() -> str:
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    key = os.environ.get("DROPBOX_APP_KEY")
    secret = os.environ.get("DROPBOX_APP_SECRET")
    if not (refresh and key and secret):
        raise RuntimeError(
            "DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET required"
        )
    with _TOK_LOCK:
        # Re-check inside the lock — another thread may have just refreshed.
        # Hmm, but we don't track expiry here; OK to occasionally double-refresh
        # since the upstream call already serializes them in practice.
        data = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": key,
                "client_secret": secret,
            }
        ).encode()
        req = urllib.request.Request("https://api.dropboxapi.com/oauth2/token", data=data)
        with urllib.request.urlopen(req, timeout=10) as r:
            j = json.loads(r.read().decode())
        _TOK["access"] = j["access_token"]
        return j["access_token"]


def _dropbox_headers() -> dict[str, str]:
    tok = _TOK.get("access") or _refresh_token()
    h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    m = os.environ.get("DROPBOX_TEAM_MEMBER_ID")
    if m:
        h["Dropbox-API-Select-User"] = m
    return h


def _api(endpoint: str, body: dict, retry: bool = True) -> dict:
    req = urllib.request.Request(
        f"https://api.dropboxapi.com/2/{endpoint}",
        data=json.dumps(body).encode(),
        headers=_dropbox_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401 and retry:
            _refresh_token()
            return _api(endpoint, body, retry=False)
        raise


def _normalize(p: str) -> str:
    p = p.strip("/")
    if p.startswith("Jordan DiCaprio/"):
        p = p[len("Jordan DiCaprio/") :]
    return f"/{p}" if p else ""


def list_folder(path: str) -> list[dict]:
    api_path = _normalize(path)
    entries: list[dict] = []
    cursor: str | None = None
    for _ in range(20):
        if cursor:
            r = _api("files/list_folder/continue", {"cursor": cursor})
        else:
            r = _api(
                "files/list_folder",
                {"path": api_path, "recursive": False, "include_non_downloadable_files": False},
            )
        entries.extend(r.get("entries", []))
        if not r.get("has_more"):
            break
        cursor = r.get("cursor")
    return entries


def download_file(path: str, retry: bool = True) -> bytes:
    """Download via Dropbox content API. Returns raw bytes."""
    headers = {
        "Authorization": f"Bearer {_TOK.get('access') or _refresh_token()}",
        "Dropbox-API-Arg": json.dumps({"path": _normalize(path)}),
    }
    m = os.environ.get("DROPBOX_TEAM_MEMBER_ID")
    if m:
        headers["Dropbox-API-Select-User"] = m
    req = urllib.request.Request(
        "https://content.dropboxapi.com/2/files/download", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 401 and retry:
            _refresh_token()
            return download_file(path, retry=False)
        raise


# ─── Fuzzy pairing (ported from app/api/review/[jobId]/route.ts) ────────────

_NUM_RE = re.compile(r"(\d+)$")
_JPG_RE = re.compile(r"\.jpe?g$", re.I)
HIST_WIDE = 10
PAIR_TIGHT = 1


def _stem(name: str) -> str:
    return _JPG_RE.sub("", name)


def pair_ours_to_vendor(ours_names: list[str], vendor_names: list[str]) -> list[tuple[str, str]]:
    """Returns [(ours_name, vendor_name), ...] using the dashboard's
    offset-histogram fuzzy match (post-2026-05-12 fix: tie-break by
    smallest |offset|)."""

    def with_num(names: list[str]) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for n in names:
            m = _NUM_RE.search(_stem(n))
            if m:
                out.append((n, int(m.group(1))))
        return sorted(out, key=lambda x: x[1])

    ours = with_num(ours_names)
    vendor = with_num(vendor_names)
    if not ours or not vendor:
        return []

    # Histogram of all (ours - vendor) offsets within ±HIST_WIDE.
    counts: dict[int, int] = defaultdict(int)
    for _, on in ours:
        for _, vn in vendor:
            d = on - vn
            if abs(d) <= HIST_WIDE:
                counts[d] += 1
    if not counts:
        return []

    # Tie-break by smallest |d| when counts equal.
    best_off, best_votes = 0, -1
    for d, c in counts.items():
        if c > best_votes or (c == best_votes and abs(d) < abs(best_off)):
            best_off, best_votes = d, c

    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for o_name, on in ours:
        target = on - best_off
        best: tuple[str, int] | None = None
        best_dist = float("inf")
        for v_name, vn in vendor:
            if v_name in used:
                continue
            dist = abs(vn - target)
            if dist <= PAIR_TIGHT and dist < best_dist:
                best, best_dist = (v_name, vn), dist
        if best:
            pairs.append((o_name, best[0]))
            used.add(best[0])
    return pairs


# ─── Image stats ────────────────────────────────────────────────────────────

# Linear-ish primary metrics. We keep things in [0, 1] sRGB space (not
# linearized) because the vendor's outputs are also display-referred
# sRGB — the comparison is meaningful in the space the reviewer sees.

METRICS = ["luminance", "contrast", "r_mean", "g_mean", "b_mean",
           "rg_ratio", "bg_ratio", "saturation"]


def image_stats(jpg_bytes: bytes, resize_px: int) -> dict[str, float]:
    img = Image.open(BytesIO(jpg_bytes)).convert("RGB")
    img.thumbnail((resize_px, resize_px))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Luma per BT.601 — what most viewers/editors call "brightness".
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    s = {
        "luminance": float(Y.mean()),
        "contrast": float(Y.std()),
        "r_mean": float(R.mean()),
        "g_mean": float(G.mean()),
        "b_mean": float(B.mean()),
        "rg_ratio": float(R.mean() / max(G.mean(), 1e-6)),  # WB temp proxy
        "bg_ratio": float(B.mean() / max(G.mean(), 1e-6)),  # WB tint proxy
    }
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    s["saturation"] = float(hsv[..., 1].mean())
    return s


# ─── Main driver ────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--per-job", type=int, default=5)
    ap.add_argument("--since", type=int, default=30, help="Days back")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--resize", type=int, default=600)
    ap.add_argument(
        "--interior-only", action="store_true",
        help="Filter to ImageReviews with enrichment.roomType != 'exterior' (and not null)",
    )
    ap.add_argument(
        "--target", type=int, default=None,
        help="Stop iterating jobs once total analyzed pairs >= this number",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent Dropbox downloads per job (default 4)",
    )
    args = ap.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL not set. Source /tmp/db.env or /etc/arem-worker.env first.")

    since_dt = datetime.now(timezone.utc) - timedelta(days=args.since)
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, "taskNumber", "jobFolderName", photographer, "dropboxPath"
          FROM "Job"
         WHERE status = 'done'
           AND "completedAt" > %s
           AND "archivedAt" IS NULL
           AND "parentJobId" IS NULL
         ORDER BY "completedAt" DESC
         LIMIT %s
        """,
        (since_dt, args.jobs),
    )
    jobs = cur.fetchall()
    print(f"# Found {len(jobs)} done jobs in last {args.since} days\n")
    if args.interior_only:
        print("# Filter: interior-only (ImageReview.enrichment.roomType != 'exterior')\n")
    if args.target:
        print(f"# Target: stop after >= {args.target} pairs analyzed\n")

    rows: list[dict] = []
    room_counts: Counter[str] = Counter()

    def _analyze_pair(o_name: str, v_name: str, o_path: str, v_path: str,
                      task_label: str, room: str | None) -> dict | None:
        try:
            o_bytes = download_file(o_path)
            v_bytes = download_file(v_path)
            o_s = image_stats(o_bytes, args.resize)
            v_s = image_stats(v_bytes, args.resize)
        except Exception as e:
            print(f"    skip {o_name}: {str(e)[:120]}")
            return None
        return {
            "job": task_label,
            "ours_name": o_name,
            "vendor_name": v_name,
            "room_type": room or "",
            **{f"ours_{k}": o_s[k] for k in METRICS},
            **{f"vendor_{k}": v_s[k] for k in METRICS},
            **{f"d_{k}": o_s[k] - v_s[k] for k in METRICS},
        }

    for ji, (jid, task, folder, photog, dpath) in enumerate(jobs, 1):
        if args.target and len(rows) >= args.target:
            print(f"# Reached target ({len(rows)} >= {args.target}) — stopping.\n")
            break

        print(f"[{ji}/{len(jobs)}] T-{task} · {folder!r}", flush=True)
        try:
            ours_entries = list_folder(f"{dpath}/08-Test-Edit")
            vendor_entries = list_folder(f"{dpath}/05-Finished-Photos")
        except Exception as e:
            print(f"  skip — list_folder failed: {str(e)[:120]}")
            continue

        ours_paths = {
            e["name"]: e["path_display"]
            for e in ours_entries
            if e.get(".tag") == "file" and _JPG_RE.search(e["name"])
        }
        vendor_paths = {
            e["name"]: e["path_display"]
            for e in vendor_entries
            if e.get(".tag") == "file" and _JPG_RE.search(e["name"])
        }
        pairs = pair_ours_to_vendor(list(ours_paths.keys()), list(vendor_paths.keys()))
        if not pairs:
            print(f"  no pairs (ours={len(ours_paths)}, vendor={len(vendor_paths)})")
            continue

        # Pull this job's ImageReview enrichment to map midStem -> roomType.
        room_by_stem: dict[str, str | None] = {}
        if args.interior_only:
            cur.execute(
                """SELECT "midStem", enrichment
                     FROM "ImageReview"
                    WHERE "jobId" = %s""",
                (jid,),
            )
            for stem, enrichment in cur.fetchall():
                if isinstance(enrichment, dict):
                    room_by_stem[stem] = enrichment.get("roomType")
                else:
                    room_by_stem[stem] = None

            before = len(pairs)
            pairs = [
                (o, v) for (o, v) in pairs
                if (rt := room_by_stem.get(_stem(o))) and rt != "exterior"
            ]
            print(f"  interior filter: {before} → {len(pairs)} pairs")
            if not pairs:
                continue

        # Evenly sample pairs across the (post-filter) shoot.
        if len(pairs) > args.per_job:
            idxs = np.linspace(0, len(pairs) - 1, args.per_job, dtype=int)
            pairs = [pairs[i] for i in idxs]
        print(f"  {len(pairs)} pairs sampled (of {len(ours_paths)} ours / {len(vendor_paths)} vendor)")

        # If we have a target, don't overshoot it on this job.
        if args.target:
            remaining = args.target - len(rows)
            if remaining > 0 and len(pairs) > remaining:
                pairs = pairs[:remaining]

        # Parallel download/analyze for this job (token refresh is lock-protected).
        task_label = f"T-{task}"
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _analyze_pair,
                    o, v, ours_paths[o], vendor_paths[v], task_label,
                    room_by_stem.get(_stem(o)),
                )
                for (o, v) in pairs
            ]
            for fut in as_completed(futures):
                row = fut.result()
                if row:
                    rows.append(row)
                    if row["room_type"]:
                        room_counts[row["room_type"]] += 1

    conn.close()

    if not rows:
        sys.exit("\nNo pairs analyzed — nothing to aggregate.")

    n_jobs = len({r["job"] for r in rows})
    print(
        f"\n# Analyzed {len(rows)} paired frames across {n_jobs} jobs.\n"
        f"# (Aggregating deltas in sRGB space, ours - vendor.)\n"
    )
    if room_counts:
        top = sorted(room_counts.items(), key=lambda kv: -kv[1])
        breakdown = ", ".join(f"{k}={v}" for k, v in top)
        print(f"# Room-type breakdown: {breakdown}\n")

    # ─── Aggregate table ───────────────────────────────────────────────────
    hdr = f"{'metric':<14}  {'ours_mean':>10}  {'vendor_mean':>10}  {'Δ mean':>9}  {'Δ median':>9}  {'Δ stdev':>9}"
    print(hdr)
    print("-" * len(hdr))
    agg: dict[str, dict[str, float]] = {}
    for m in METRICS:
        ours_arr = np.array([r[f"ours_{m}"] for r in rows])
        vendor_arr = np.array([r[f"vendor_{m}"] for r in rows])
        delta = ours_arr - vendor_arr
        agg[m] = {
            "ours_mean": float(ours_arr.mean()),
            "vendor_mean": float(vendor_arr.mean()),
            "d_mean": float(delta.mean()),
            "d_median": float(np.median(delta)),
            "d_std": float(delta.std()),
        }
        print(
            f"{m:<14}  {agg[m]['ours_mean']:>10.4f}  {agg[m]['vendor_mean']:>10.4f}  "
            f"{agg[m]['d_mean']:>+9.4f}  {agg[m]['d_median']:>+9.4f}  {agg[m]['d_std']:>9.4f}"
        )

    # ─── Suggested constant corrections ────────────────────────────────────
    print("\n# Suggested constant corrections to apply at end of pipeline:\n")
    # Exposure in stops (EV). +/− = brighten / darken.
    ev = float(np.log2(max(agg["luminance"]["vendor_mean"], 1e-6)
                       / max(agg["luminance"]["ours_mean"], 1e-6)))
    print(f"  Exposure:      {ev:+.3f} EV  ({'brighten' if ev > 0 else 'darken'} our output)")
    # Contrast as a relative multiplier on the luma std-dev.
    rel_contrast = (agg["contrast"]["vendor_mean"] - agg["contrast"]["ours_mean"]) / max(agg["contrast"]["ours_mean"], 1e-6)
    print(f"  Contrast:      {rel_contrast:+.3f} (relative, e.g. +0.05 = scale Y about its mean by 1.05)")
    # WB temp/tint via channel ratios. R/G > vendor → too warm → cool down.
    rg = agg["rg_ratio"]["ours_mean"] - agg["rg_ratio"]["vendor_mean"]
    bg = agg["bg_ratio"]["ours_mean"] - agg["bg_ratio"]["vendor_mean"]
    print(f"  WB temp (R/G): {-rg:+.4f}  ({'cool down' if rg > 0 else 'warm up'} our output)")
    print(f"  WB tint (B/G): {-bg:+.4f}  ({'less magenta' if bg < 0 else 'less green'})")
    sat = agg["saturation"]["ours_mean"] - agg["saturation"]["vendor_mean"]
    print(f"  Saturation:    {-sat:+.4f}  ({'desaturate' if sat > 0 else 'saturate'})")

    # Variance hint — when stdev of Δ is comparable to mean, a constant
    # correction won't cut it and you should reach for a per-image model.
    print("\n# Δ stdev / |Δ mean| ratios — high = per-image variance dominates the bias:")
    for m in METRICS:
        mean_d = agg[m]["d_mean"]
        std_d = agg[m]["d_std"]
        ratio = std_d / max(abs(mean_d), 1e-6)
        verdict = "constant fix viable" if ratio < 1.5 else "needs per-image model"
        print(f"  {m:<14}  ratio={ratio:>6.2f}   ({verdict})")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n# Wrote per-pair CSV: {args.csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
