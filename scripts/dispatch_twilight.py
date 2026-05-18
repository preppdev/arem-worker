"""Drain pending TwilightRequest rows by calling Google Nano Banana directly.

Parallel worker, runs from the 3090 box. Reads pending rows from main DB,
downloads source from CF Images, calls Google AI Studio, uploads result to
CF Images, updates the DB row. Loops until no pending remain.

Env required (typically sourced from .env.local + an injected GOOGLE_AI_API_KEY):
    DATABASE_URL
    GOOGLE_AI_API_KEY
    CLOUDFLARE_ACCOUNT_ID
    CLOUDFLARE_API_TOKEN

Usage:
    python -m scripts.dispatch_twilight                  # process all pending
    python -m scripts.dispatch_twilight --workers 5      # parallelism
    python -m scripts.dispatch_twilight --limit 10       # only N
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore

CF_HASH = "hfTNewiEIKKL4K9rrhwauQ"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def log(s: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)


DEFAULT_PROMPT = """Transform this daytime exterior real-estate photo into a professional DAY-TO-DUSK photograph showing the SAME EXACT BUILDING AND SCENERY at sunset.

CRITICAL — BUILDING IDENTITY: This is the EXACT building from the source photo. Do NOT generate a different house. Do NOT change the architecture, roofline, story count, window arrangement, door placement, materials, siding color, masonry, or any structural element. Do NOT modify the landscape or scenery. The output must show the IDENTICAL building and IDENTICAL scenery from the IDENTICAL camera angle and framing — only the sky, lighting, and time-of-day cues change.

SKY: Warm coral-orange sunset. Deep navy at top, transitioning through purple-mauve to vivid coral and bright orange near the horizon. Pale orange dominates the lower half with bright golden-orange highlights right at the horizon line. Warm-dominant and orange-leaning. Soft layered horizontal streaks, not puffy cumulus clouds.

LIGHTING — SUBTLE dusk only: The scene transitions to dusk light levels through a modest reduction in overall brightness and slightly lifted shadows. The foreground (lawn, driveway, siding, masonry, walkway) RETAINS its natural color balance — apply only a LIGHT, GENTLE warm tint, NOT a heavy orange wash. Grass must still read as green. White siding must still read as white. Pavement must still read as gray. Every visible light fixture (interior windows, porch lights, sconces) emits warm golden glow.

WINDOW CONTENTS: For every window, ONLY re-color and re-light what is already visible in the source. DO NOT invent new curtains, furniture, lamps, plants, or decor.

DO NOT ADD ANYTHING: No new lamp posts, garden lights, vehicles, signs, plants, or architectural elements. Only allowed changes: sky, existing light fixtures' glow, warm window glow, and a subtle natural dusk tone on existing elements.

Preserve the exact building, framing, perspective, camera angle, composition, and every existing object's identity."""


def load_settings(conn) -> dict:
    """Read twilight.* settings from AppSetting; falls back to baked defaults."""
    defaults = {
        "twilight.prompt": DEFAULT_PROMPT,
        "twilight.model": "gemini-3-pro-image-preview",
        "twilight.fallbackModel": "gemini-2.5-flash-image",
        "twilight.aspectRatio": "Auto",
        "twilight.imageSize": "2K",
    }
    cur = conn.cursor()
    cur.execute('SELECT key, value FROM "AppSetting" WHERE key LIKE %s', ("twilight.%",))
    rows = cur.fetchall()
    cur.close()
    out = dict(defaults)
    for k, v in rows:
        # v is a JSONB value — could be a string or other
        out[k] = v if not isinstance(v, str) else v
    return out


def call_google(api_key: str, model: str, prompt: str, image_bytes: bytes,
                aspect_ratio: str, image_size: str, timeout: int = 300) -> bytes:
    image_config = {"imageSize": image_size}
    if aspect_ratio and aspect_ratio != "Auto":
        image_config["aspectRatio"] = aspect_ratio
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(image_bytes).decode()}},
        ]}],
        "generationConfig": {"imageConfig": image_config},
    }
    req = urllib.request.Request(
        f"{GOOGLE_API_BASE}/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    for c in data.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if p.get("inlineData", {}).get("data"):
                return base64.b64decode(p["inlineData"]["data"])
    raise RuntimeError("no inline image in Google response")


def call_google_with_fallback(api_key: str, primary: str, fallback: str | None,
                              prompt: str, image_bytes: bytes,
                              aspect_ratio: str, image_size: str) -> tuple[bytes, str]:
    last_err = None
    for model in [primary, fallback]:
        if not model:
            continue
        try:
            out = call_google(api_key, model, prompt, image_bytes, aspect_ratio, image_size)
            return out, model
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            last_err = f"{model} HTTP {e.code}: {msg}"
            if e.code != 503 and "UNAVAILABLE" not in msg:
                raise RuntimeError(last_err)
        except Exception as e:
            last_err = f"{model}: {e}"
            raise RuntimeError(last_err)
    raise RuntimeError(f"all models failed: {last_err}")


def upload_to_cf_images(bytes_: bytes, image_id: str, metadata: dict) -> str:
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    # multipart/form-data; we hand-roll it to avoid external deps
    boundary = "----arem-twilight-" + base64.urlsafe_b64encode(os.urandom(9)).decode()
    body_parts = []
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="id"\r\n\r\n')
    body_parts.append(image_id.encode() + b"\r\n")
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(b'Content-Disposition: form-data; name="metadata"\r\n\r\n')
    body_parts.append(json.dumps(metadata).encode() + b"\r\n")
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{image_id}.png"\r\n'.encode()
    )
    body_parts.append(b"Content-Type: image/png\r\n\r\n")
    body_parts.append(bytes_ + b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/images/v1",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    if not data.get("success"):
        raise RuntimeError(f"CF Images upload failed: {json.dumps(data)[:200]}")
    return data["result"]["id"]


def fetch_source(cf_id: str) -> bytes:
    url = f"https://imagedelivery.net/{CF_HASH}/{cf_id}/w2400"
    req = urllib.request.Request(url, headers={"User-Agent": "arem-twilight"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def process_one(row: dict, settings: dict, api_key: str) -> tuple[str, str | None, str | None, str | None]:
    """Returns (id, status, outputCfImageId, errorMessage)."""
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        cur = conn.cursor()
        # Mark processing if still pending
        cur.execute(
            'UPDATE "TwilightRequest" SET status=%s, "dispatchedAt"=%s '
            'WHERE id=%s AND status=%s',
            ("processing", datetime.now(timezone.utc), row["id"], "pending"),
        )
        if cur.rowcount == 0:
            conn.commit()
            return row["id"], "skipped (not pending)", None, None
        conn.commit()

        # Fetch source bytes
        cur.execute('SELECT "cfImageId" FROM "ImageReview" WHERE id=%s', (row["imageReviewId"],))
        ir = cur.fetchone()
        if not ir or not ir[0]:
            return mark_fail(conn, row["id"], "no cfImageId on ImageReview")
        src_bytes = fetch_source(ir[0])

        # Generate
        out_bytes, model_used = call_google_with_fallback(
            api_key, settings["twilight.model"], settings["twilight.fallbackModel"],
            settings["twilight.prompt"], src_bytes,
            settings["twilight.aspectRatio"], settings["twilight.imageSize"],
        )

        # Upload
        cf_image_id = upload_to_cf_images(
            out_bytes,
            f"twilight-{row['id']}",
            {"source": "dispatch_twilight.py", "model": model_used},
        )

        # Mark done
        cur.execute(
            'UPDATE "TwilightRequest" SET status=%s, "outputCfImageId"=%s, '
            '"completedAt"=%s, "modelRun"=%s WHERE id=%s',
            ("done", cf_image_id, datetime.now(timezone.utc), f"google-{model_used}", row["id"]),
        )
        conn.commit()
        return row["id"], "done", cf_image_id, None
    except Exception as e:
        msg = str(e)[:500]
        return mark_fail(conn, row["id"], msg)
    finally:
        conn.close()


def mark_fail(conn, row_id: str, msg: str) -> tuple[str, str, None, str]:
    cur = conn.cursor()
    cur.execute(
        'UPDATE "TwilightRequest" SET status=%s, "errorMessage"=%s, "completedAt"=%s '
        'WHERE id=%s',
        ("error", msg, datetime.now(timezone.utc), row_id),
    )
    conn.commit()
    return row_id, "error", None, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    for var in ("DATABASE_URL", "GOOGLE_AI_API_KEY",
                "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        if not os.environ.get(var):
            sys.exit(f"missing env: {var}")

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    settings = load_settings(conn)
    log(f"settings: model={settings['twilight.model']} fallback={settings['twilight.fallbackModel']} "
        f"aspect={settings['twilight.aspectRatio']} size={settings['twilight.imageSize']} "
        f"prompt_len={len(settings['twilight.prompt'])}")

    # Snapshot pending rows
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    q = 'SELECT id, "imageReviewId" FROM "TwilightRequest" WHERE status=%s ORDER BY "createdAt"'
    if args.limit > 0:
        q += f" LIMIT {args.limit}"
    cur.execute(q, ("pending",))
    rows = list(cur.fetchall())
    cur.close()
    conn.close()
    log(f"pending: {len(rows)}")
    if not rows:
        return

    api_key = os.environ["GOOGLE_AI_API_KEY"]
    done = err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, dict(r), settings, api_key): r["id"] for r in rows}
        for f in as_completed(futs):
            rid, status, cf_id, msg = f.result()
            elapsed = time.time() - t0
            if status == "done":
                done += 1
                log(f"  [{done+err}/{len(rows)}] DONE  {rid}  cf={cf_id}  ({elapsed:.0f}s)")
            elif status == "error":
                err += 1
                log(f"  [{done+err}/{len(rows)}] ERR   {rid}  {msg}  ({elapsed:.0f}s)")
            else:
                log(f"  [{done+err}/{len(rows)}] SKIP  {rid}  {status}  ({elapsed:.0f}s)")

    log(f"done={done} error={err} total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
