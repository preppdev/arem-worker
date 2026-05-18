"""Off-the-shelf classification of untouched ImageReview rows.

Three-stage on the local 3090:
  1. Interior vs exterior using pipeline/classifier_v2.pth (99.2% acc, already-trained).
  2. For interior images, room classifier (CLASSIFIER_ROOM env) — 9 room types
     (bedroom / bathroom / kitchen / living_room / dining_room / hallway /
     utility_laundry_stairs / garage / office_den).
  3. For exterior images, CLIP zero-shot for front / back / ambiguous.

Source images are pulled from Cloudflare Images at w384.

Usage:
    # Dry-run on the next 100 untouched rows
    python -m scripts.classify_exterior_subtype --limit 100 --dry-run

    # Process 5000 rows, write JSON file only (no DB write)
    python -m scripts.classify_exterior_subtype --limit 5000 --out /tmp/classify.json

    # Process everything untouched and write to DB
    python -m scripts.classify_exterior_subtype --all --apply

Env vars:
    DATABASE_URL              — postgres connection string (required)
    CLASSIFIER_PATH           — defaults to ../pipeline/classifier_v2.pth
    CLASSIFIER_ROOM           — path to room TwoHead checkpoint (optional;
                                without it, interior images get isInteriorWorker
                                set but no roomTypeWorker)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

import torch  # type: ignore
from PIL import Image  # type: ignore
import psycopg2  # type: ignore

# Reuse the trained interior/exterior classifier from the worker.
from pipeline.run_pipeline import (  # type: ignore  # noqa: E402
    classify,
    classify_room,
    load_classifier,
    load_room_classifier,
)

CF_HASH = "hfTNewiEIKKL4K9rrhwauQ"
CLASSIFIER_PATH = Path(
    os.environ.get(
        "CLASSIFIER_PATH",
        str(Path(__file__).resolve().parent.parent / "pipeline" / "classifier_v2.pth"),
    )
)
INTERIOR_THRESHOLD = 0.6  # interior verdict requires this much confidence
FRONT_BACK_THRESHOLD = 0.55  # else → ambiguous


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_image(cf_id: str, timeout: int = 10) -> Image.Image:
    url = f"https://imagedelivery.net/{CF_HASH}/{cf_id}/w384"
    req = urllib.request.Request(url, headers={"User-Agent": "arem-worker/classify"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return Image.open(BytesIO(data)).convert("RGB")


def load_clip(device):
    """Lazy-import open_clip so this module can be inspected without it."""
    try:
        import open_clip  # type: ignore
    except ImportError:
        sys.exit("open-clip-torch not installed: pip install open-clip-torch")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    # Prompts per class — averaged. Empirically more reliable than single prompts.
    prompts = {
        "front": [
            "the front exterior of a residential house",
            "a front facade of a home with the main entrance",
            "a house viewed from the street with a driveway",
            "the front of a residential home, curb view",
        ],
        "back": [
            "the back of a residential house with a backyard",
            "the rear of a home with a deck or patio",
            "a backyard view of a residential property",
            "the back exterior of a home",
        ],
        "ambiguous": [
            "an aerial drone view of a property",
            "a close-up architectural detail",
            "a neighborhood street view",
            "a blurry or unclear photograph",
        ],
    }
    flat_prompts, flat_labels = [], []
    for label, ps in prompts.items():
        for p in ps:
            flat_prompts.append(p)
            flat_labels.append(label)
    text_tokens = tokenizer(flat_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features, flat_labels


def clip_front_back(model, preprocess, text_features, flat_labels, image: Image.Image, device) -> tuple[str, float]:
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(x)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ text_features.T).cpu().numpy()[0]
    # Aggregate per-label (mean over prompts)
    label_scores = {}
    label_counts = {}
    for sim, lbl in zip(sims, flat_labels):
        label_scores[lbl] = label_scores.get(lbl, 0.0) + float(sim)
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    avg = {lbl: label_scores[lbl] / label_counts[lbl] for lbl in label_scores}
    # Softmax-ish over the 3 classes to express confidence
    vals = list(avg.values())
    keys = list(avg.keys())
    mx = max(vals)
    exps = [pow(2.718281828, (v - mx) * 30.0) for v in vals]  # temperature 30
    total = sum(exps)
    probs = [e / total for e in exps]
    top_idx = probs.index(max(probs))
    return keys[top_idx], probs[top_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="/tmp/classify-out.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="process N rows but skip writing the output file")
    ap.add_argument("--no-front-back", action="store_true",
                    help="skip CLIP front/back step (interior/exterior only)")
    ap.add_argument("--apply", action="store_true",
                    help="after processing, write verdicts directly to ImageReview "
                         "(sets exteriorSubType + isInteriorCorrected + classifiedAt). "
                         "WITHOUT this flag, only the JSON file is produced.")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not os.environ.get("DATABASE_URL"):
        sys.exit("DATABASE_URL must be set")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    # 1. Load the trained interior/exterior classifier.
    log(f"loading classifier from {CLASSIFIER_PATH}")
    if not CLASSIFIER_PATH.exists():
        sys.exit(f"missing {CLASSIFIER_PATH}")
    clf, label_names, tfm = load_classifier(CLASSIFIER_PATH, device)

    # 2. Optional: room classifier for interior subtype.
    room_model = None
    room_ckpt_env = os.environ.get("CLASSIFIER_ROOM")
    if room_ckpt_env:
        log(f"loading room classifier from {room_ckpt_env}")
        room_model = load_room_classifier(Path(room_ckpt_env), device)
    else:
        log("CLASSIFIER_ROOM env not set — interiors will get isInteriorWorker=true "
            "but no roomTypeWorker")

    # 3. Load CLIP for front/back (unless disabled).
    clip_bundle = None
    if not args.no_front_back:
        log("loading open_clip ViT-B-32 (laion2b)")
        clip_bundle = load_clip(device)

    # 3. Pull rows from DB.
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    # Process all rows with a CF image, EXCEPT user-labeled ones (verifiedBy set).
    # This deliberately overwrites any prior auto-label (Haiku etc) with the
    # local classifier output, which is more accurate at 99.2%.
    if args.all:
        cur.execute("""
            SELECT id, "cfImageId" FROM "ImageReview"
            WHERE "cfImageId" IS NOT NULL
              AND "verifiedBy" IS NULL
            ORDER BY "createdAt" DESC
        """)
    else:
        cur.execute("""
            SELECT id, "cfImageId" FROM "ImageReview"
            WHERE "cfImageId" IS NOT NULL
              AND "verifiedBy" IS NULL
            ORDER BY "createdAt" DESC
            LIMIT %s
        """, (args.limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    log(f"processing {len(rows)} rows")

    results = []
    t0 = time.time()
    for i, (rid, cf_id) in enumerate(rows):
        if i % 50 == 0 and i:
            rate = i / (time.time() - t0)
            eta_min = (len(rows) - i) / rate / 60
            log(f"  [{i}/{len(rows)}] rate={rate:.1f}/s eta={eta_min:.1f}min")

        try:
            img = fetch_image(cf_id)
        except Exception as e:
            results.append({"id": rid, "verdict": None, "reason": f"fetch_failed: {e}"})
            continue

        # Save the image briefly for the classifier (it expects a Path).
        tmp_path = Path(f"/tmp/_classify_tmp_{os.getpid()}.jpg")
        img.save(tmp_path, "JPEG", quality=85)
        try:
            is_interior, conf, _ = classify(clf, label_names, tfm, tmp_path, device)
        finally:
            tmp_path.unlink(missing_ok=True)

        if is_interior and conf >= INTERIOR_THRESHOLD:
            entry = {
                "id": rid,
                "verdict": "interior",
                "interior_conf": conf,
                "subtype_conf": None,
                "room": None,
                "room_conf": None,
            }
            if room_model is not None:
                tmp_path = Path(f"/tmp/_classify_room_tmp_{os.getpid()}.jpg")
                img.save(tmp_path, "JPEG", quality=85)
                try:
                    room_info = classify_room(room_model, tmp_path, device)
                finally:
                    tmp_path.unlink(missing_ok=True)
                if room_info:
                    entry["room"] = room_info["roomType"]
                    entry["room_conf"] = room_info["roomConfidence"]
            results.append(entry)
            continue

        if args.no_front_back:
            # Just mark exterior — caller decides subtype.
            results.append({
                "id": rid,
                "verdict": "exterior",
                "interior_conf": conf,
                "subtype_conf": None,
            })
            continue

        # Exterior — run CLIP for front/back/ambiguous
        model, preprocess, text_features, flat_labels = clip_bundle
        sub, sub_conf = clip_front_back(
            model, preprocess, text_features, flat_labels, img, device
        )
        # If softmax confidence is below threshold, fall back to ambiguous.
        if sub_conf < FRONT_BACK_THRESHOLD:
            sub = "ambiguous"
        results.append({
            "id": rid,
            "verdict": sub,            # front | back | ambiguous
            "interior_conf": conf,     # interior confidence (so we know how confident "not interior" was)
            "subtype_conf": sub_conf,
        })

    log(f"done in {(time.time()-t0)/60:.1f}min")
    # Distribution summary
    from collections import Counter
    dist = Counter(r["verdict"] for r in results)
    log(f"distribution: {dict(dist)}")

    if args.dry_run:
        log("--dry-run set, skipping write")
        return

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {args.out}")

    if not args.apply:
        log("--apply not set, leaving DB untouched. Review the JSON, then re-run with --apply to write.")
        return

    # Apply to DB. JSONB-null safe merge for classifierModelVersions.
    log("applying to DB...")
    SAFE_MERGE = (
        "CASE WHEN \"classifierModelVersions\" IS NULL "
        "OR jsonb_typeof(\"classifierModelVersions\") != 'object' "
        "THEN '{}'::jsonb ELSE \"classifierModelVersions\" END || "
        "'{\"exteriorSubtype\":\"local-clip-v1\"}'::jsonb"
    )
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    applied = 0
    hero_seen = set()  # not used here — front/back/ambiguous have no uniqueness rule
    for r in results:
        if r["verdict"] is None:
            continue
        if r["verdict"] == "interior":
            cur.execute(
                f"UPDATE \"ImageReview\" SET "
                f"\"isInteriorWorker\"=true, \"roomTypeWorker\"=%s, "
                f"\"roomConfidenceWorker\"=%s, "
                f"\"isInteriorCorrected\"=NULL, \"exteriorSubType\"=NULL, "
                f"\"classifiedAt\"=NOW(), \"classifierModelVersions\"=({SAFE_MERGE}) "
                f"WHERE id=%s AND \"exteriorSubType\" IS NULL AND \"isInteriorCorrected\" IS NULL",
                (r.get("room"), r.get("room_conf"), r["id"]),
            )
        elif r["verdict"] == "exterior":
            cur.execute(
                f"UPDATE \"ImageReview\" SET "
                f"\"isInteriorWorker\"=false, \"exteriorSubType\"=NULL, "
                f"\"classifiedAt\"=NOW(), \"classifierModelVersions\"=({SAFE_MERGE}) "
                f"WHERE id=%s AND \"exteriorSubType\" IS NULL AND \"isInteriorCorrected\" IS NULL",
                (r["id"],),
            )
        else:  # front, back, ambiguous
            cur.execute(
                f"UPDATE \"ImageReview\" SET "
                f"\"isInteriorWorker\"=false, \"exteriorSubType\"=%s, "
                f"\"classifiedAt\"=NOW(), \"classifierModelVersions\"=({SAFE_MERGE}) "
                f"WHERE id=%s AND \"exteriorSubType\" IS NULL AND \"isInteriorCorrected\" IS NULL",
                (r["verdict"], r["id"]),
            )
        applied += cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    log(f"applied {applied} updates")


if __name__ == "__main__":
    main()
