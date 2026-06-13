# 03 — Room Classifier

**What:** Classifies an interior image into one of 14 room types
(bathroom, bedroom, closet, dining, exterior, foyer, garage, hallway,
kitchen, living, office, stairs, sunroom, utility). Bakes `roomType` +
confidence into EXIF and `ImageReview.roomTypeWorker`.

**Status:** Production **v4**, deployed 2026-06-13 (tag
`worker-fleet-2026-06-13`). **Informational only** — not yet used to reorder
galleries (that's the next step). 88.80% holdout accuracy. **Note:**
`CLASSIFIER_ROOM` was unset before this deploy, so room classification is
genuinely *new* in production as of 06-13.

**Code:** `pipeline/run_pipeline.py` — `load_room_classifier()` (~L648),
`classify_room()` (~L695). Trainer: `cloud_train/train_room_native.py`.

## What it does

- Input: the **Stage-2 finished** JPG (not Stage-1). v4 letterboxes the FULL
  frame to **1365×2048** (`_LetterboxTfm`, ImageNet normalize) — Jordan's
  hard mandate: **no crops, the model needs full room context**.
- Model: `convnext_base(weights=None)`, `classifier[2] → Linear(., 14)`.
  **hflip TTA** (logits summed over the image and its mirror).
- Output dict: `{roomType, roomConfidence, occupancy:"unknown",
  occupancyConfidence:None, roomDistribution:{class:prob}, version}`. (v4 is
  room-only; the legacy TwoHead also predicted occupancy — the loader keeps
  back-compat for that format.)
- Reaches the dashboard as `ImageReview.roomTypeWorker` /
  `roomConfidenceWorker` via `worker.py` → `/api/internal/image-classification`.
  EXIF: `embed_classification_exif` writes `XMP-arem:roomType`/`roomConfidence`
  + IPTC keywords + UserComment JSON.
- Checkpoint dict: `{model, classes, input_hw:(1365,2048), acc, arch:
  "convnext_base", tta:"hflip", letterbox:True, version}`. Env
  `CLASSIFIER_ROOM`. R2 `models/room_classifier_v4_native.pth` (350MB) + both
  nodes `~/models/`. ~1.7GB VRAM at inference.

## How it was produced

- Lineage: v2 ResNet/ConvNeXt-S @224/448 → **v4 ConvNeXt-Base full-frame**
  (B200 `cloud_train`, 2026-06-13).
- Trained on `corrected_room.jsonl` (33.7k labels), shoot-split seed 17.
- Recipe (`train_room_native.py`): ConvNeXt-Base, **1365×2048 letterbox**
  (= "option C": half-native res, batch 12 / accum 2 ≈ eff 24, 16 epochs,
  ~7.5h/$45 on one B200), EMA 0.999, mixup α0.2 p0.5, sqrt-inverse class
  weights, tail-class dup (3× <350, 2× <800 samples), label smoothing 0.1,
  hflip TTA. Best checkpoint **0.8880 at epoch 9** (drifted down after, so
  ep9 is the keeper). Beats the honest full-frame v2 baseline (86.86%) by
  ~2 pts. (A native 2800×4200 run was ~2h/epoch → too slow; 1365×2048 was the
  pace/quality compromise.)

## What it doesn't do / isn't good enough yet

- **Weak classes:** sunroom 0.56 recall (smallest class, few examples),
  office 0.73, foyer/dining/living ~0.81–0.83 (open-plan rooms bleed into
  each other). Strong: bathroom 0.97, kitchen 0.95, closet 0.94, bedroom 0.91.
- **Not wired to auto-reorder** yet — purely informational. `effectiveScene`
  / `sceneRank` in the editing repo's `lib/delivery.ts` already exist; turning
  on gallery auto-ordering is a small follow-up.
- No occupancy (vacant/staged) in v4.
- No confidence calibration.
- `exterior` is a class in the room head (catch-all when the scene router was
  wrong), distinct from the router's interior/exterior decision — minor
  redundancy.

## Intended fine-tune method

The collection loop is already built:
1. **`/room-collector`** dashboard page (nav "Room Label Collector (v4)")
   surfaces v4 predictions **lowest-confidence first**, lets a reviewer
   confirm-as-predicted or correct → writes `ImageReview.roomTypeCorrected`
   (+ verify stamp). Sibling tools: `/classify` (3-axis), `/room-corrections`
   (enrichment JSON). Corrections feed `effectiveScene` immediately.
2. When enough fresh corrected labels accrue (target the weak classes —
   sunroom/office especially; tail-class dup helps but real examples help
   more), export `roomTypeCorrected` rows → append to `corrected_room.jsonl`.
3. Re-run `train_room_native.py` on a B200, same recipe; bump version.
4. Accept if holdout beats 0.888 (watch per-class recall on the weak set);
   ship to `models/room_classifier_v4_native.pth` + `~/models/` on both
   nodes, new fleet tag.

> The RunPod cloud-backup image needs a CI rebuild to pick up the
> `entrypoint.sh` `CLASSIFIER_ROOM` fetch; the local fleet is already live.
