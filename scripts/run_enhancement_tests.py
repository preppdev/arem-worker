"""Run Nano Banana enhancement-condition evaluations on random samples.

For each condition: pick N images where conditions->>'<key>' = 'true',
run them through Google Gemini Image with the saved per-condition prompt,
upload the result to CF Images, insert EnhancementTest rows.

Usage:
    python -m scripts.run_enhancement_tests \\
        --conditions dead-fixture,color-cast,reflection \\
        --per-condition 5

Env:
    DATABASE_URL, GOOGLE_AI_API_KEY,
    CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import psycopg2  # type: ignore
import psycopg2.extras  # type: ignore

CF_HASH = "hfTNewiEIKKL4K9rrhwauQ"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
PRIMARY_MODEL = "gemini-3-pro-image-preview"
FALLBACK_MODEL = "gemini-2.5-flash-image"
ASPECT = "Auto"
SIZE = "2K"

# Prompts are pulled from AppSetting when present; these are the bake-in
# defaults that match lib/settings-defaults.ts.
DEFAULT_PROMPTS = {
    "dead-fixture":
        "Analyze this photo for light fixtures that appear to not be working and edit them so they are glowing as if operational. Do not modify any other element of the image; preserve the building, scene, framing, and time-of-day exactly.",
    "color-cast":
        "Analyze this photo and correct any out-of-place color casts present that don't match the general tone of the scene. Preserve the building, scene, framing, exposure, and intentional lighting exactly — only neutralize unnatural color casts.",
    "reflection":
        "Analyze this photo for the reflection of a photographer, camera, tripod, humans, or photography equipment in the reflection of mirrors or reflective surfaces and remove all traces of the reflection. Preserve everything else (building, scene, framing, lighting) exactly.",
}


def log(s: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)


def load_prompt(conn, condition: str) -> str:
    cur = conn.cursor()
    cur.execute('SELECT value FROM "AppSetting" WHERE key=%s', (f"enhancement.{condition}.prompt",))
    row = cur.fetchone()
    cur.close()
    if row and isinstance(row[0], str) and row[0].strip():
        return row[0]
    return DEFAULT_PROMPTS.get(condition, "")


def pick_samples(conn, condition: str, n: int) -> list[dict]:
    """Pick N random images flagged with the condition. Skips ones already in EnhancementTest for this condition."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT ir.id, ir."cfImageId", ir."midStem"
        FROM "ImageReview" ir
        WHERE ir."cfImageId" IS NOT NULL
          AND ir.conditions IS NOT NULL
          AND ir.conditions->>%s = 'true'
          AND NOT EXISTS (
            SELECT 1 FROM "EnhancementTest" et
            WHERE et."imageReviewId" = ir.id AND et.condition = %s
          )
        ORDER BY random()
        LIMIT %s
        """,
        (condition, condition, n),
    )
    rows = list(cur.fetchall())
    cur.close()
    return rows


def fetch_source(cf_id: str) -> bytes:
    url = f"https://imagedelivery.net/{CF_HASH}/{cf_id}/w2400"
    req = urllib.request.Request(url, headers={"User-Agent": "arem-enhancement-test"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def call_google(api_key: str, model: str, prompt: str, image_bytes: bytes) -> bytes:
    image_config = {"imageSize": SIZE}
    if ASPECT and ASPECT != "Auto":
        image_config["aspectRatio"] = ASPECT
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
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    for c in data.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if p.get("inlineData", {}).get("data"):
                return base64.b64decode(p["inlineData"]["data"])
    raise RuntimeError("no inline image in Google response")


def call_google_with_fallback(api_key: str, prompt: str, image_bytes: bytes) -> tuple[bytes, str]:
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            return call_google(api_key, model, prompt, image_bytes), model
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code != 503 and "UNAVAILABLE" not in msg:
                raise RuntimeError(f"{model} HTTP {e.code}: {msg}")
    raise RuntimeError("all models failed (503)")


def upload_cf_images(bytes_: bytes, image_id: str, metadata: dict) -> str:
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    boundary = "----arem-enh-" + base64.urlsafe_b64encode(os.urandom(9)).decode()
    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="id"\r\n\r\n')
    parts.append(image_id.encode() + b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="metadata"\r\n\r\n')
    parts.append(json.dumps(metadata).encode() + b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{image_id}.png"\r\n'.encode()
    )
    parts.append(b"Content-Type: image/png\r\n\r\n")
    parts.append(bytes_ + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
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
        raise RuntimeError(f"CF upload failed: {json.dumps(data)[:200]}")
    return data["result"]["id"]


def process_one(sample: dict, condition: str, prompt: str, api_key: str) -> dict:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        cur = conn.cursor()
        # Create the row in processing state
        test_id = "et_" + secrets.token_hex(12)
        cur.execute(
            """
            INSERT INTO "EnhancementTest"
              (id, condition, "imageReviewId", "sourceCfImageId", prompt, status, "createdAt")
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """,
            (test_id, condition, sample["id"], sample["cfImageId"], prompt, "processing"),
        )
        conn.commit()

        src = fetch_source(sample["cfImageId"])
        out, model_used = call_google_with_fallback(api_key, prompt, src)
        cf_id = upload_cf_images(
            out, f"enh-{condition}-{test_id}",
            {"source": "run_enhancement_tests.py", "condition": condition, "model": model_used},
        )

        cur.execute(
            'UPDATE "EnhancementTest" SET status=%s, "outputCfImageId"=%s, "modelUsed"=%s, "completedAt"=NOW() WHERE id=%s',
            ("done", cf_id, model_used, test_id),
        )
        conn.commit()
        return {"id": test_id, "status": "done", "midStem": sample["midStem"], "model": model_used}
    except Exception as e:
        msg = str(e)[:400]
        try:
            cur.execute(
                'UPDATE "EnhancementTest" SET status=%s, "errorMessage"=%s, "completedAt"=NOW() WHERE id=%s',
                ("error", msg, test_id),
            )
            conn.commit()
        except Exception:
            pass
        return {"id": test_id if "test_id" in dir() else None, "status": "error",
                "midStem": sample["midStem"], "error": msg}
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", required=True,
                    help="comma-separated condition keys, e.g. dead-fixture,color-cast,reflection")
    ap.add_argument("--per-condition", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    for var in ("DATABASE_URL", "GOOGLE_AI_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        if not os.environ.get(var):
            sys.exit(f"missing env: {var}")

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    api_key = os.environ["GOOGLE_AI_API_KEY"]

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    plan = []
    for c in conds:
        prompt = load_prompt(conn, c)
        if not prompt:
            log(f"WARN: no prompt for {c}, skipping")
            continue
        samples = pick_samples(conn, c, args.per_condition)
        log(f"{c}: picked {len(samples)} samples (prompt {len(prompt)} chars)")
        for s in samples:
            plan.append((s, c, prompt))
    conn.close()

    log(f"total tasks: {len(plan)}")
    if not plan:
        return

    t0 = time.time()
    done = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, s, c, p, api_key): (s["midStem"], c) for s, c, p in plan}
        for f in as_completed(futs):
            mid, cond = futs[f]
            r = f.result()
            if r["status"] == "done":
                done += 1
                log(f"  DONE  {cond}/{mid}  model={r.get('model')}  [{done+err}/{len(plan)}]")
            else:
                err += 1
                log(f"  ERR   {cond}/{mid}  {r.get('error')}  [{done+err}/{len(plan)}]")

    log(f"finished: done={done} err={err} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
