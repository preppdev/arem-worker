# 09 — Condition Pretagger (reflection etc.)

**What:** A standalone inference **service** (separate repo `arem-pretagger`)
that pre-tags finished images with per-condition confidences (reflection,
and—planned—dead-fixture, etc.). The worker calls it after Stage 2; results
populate `ImageReview.preTagConfidence` so the reviewer UI can surface
"suggested" condition flags.

**Status:** Reflection classifier live; other conditions stubbed. Runs as an
HTTP service on the fleet node (port 8090).

**Code:** repo `arem-pretagger` — `serving/app.py` (FastAPI `/pretag`),
`serving/model_registry.py` (load + infer), `training/train_reflection.py`.
Worker caller: `pipeline/pretagger.py`; dashboard sink:
`/api/internal/pretag-result`.

## What it does

- Service `POST /pretag` (token `X-Pretag-Token`): takes image bytes, returns
  `{ model_versions, latency_ms, conditions: { <cond>: { confidence,
  mask_r2_path?, bboxes? } } }`.
- Model bundles live on disk per condition:
  `<CHECKPOINT_DIR>/<condition>/{manifest.json, weights.pt}` where manifest =
  `{model_type, version, weights_path, threshold, metrics}`.
- **Reflection** (`model_type: classifier`): ResNet-18 backbone (ImageNet,
  frozen) + head `Linear(512,128)→ReLU→Dropout0.3→Linear(128,1)` → sigmoid.
  Crops/normalizes to 224, returns `confidence ∈ [0,1]`.
- Stubbed: `yolo-detect` (dead-fixture, planned YOLOv8n + classifier),
  `mobile-sam` (reflection mask, planned).
- The worker (`pipeline/pretagger.py`) POSTs the image, maps the response, and
  forwards to the dashboard `/api/internal/pretag-result` →
  `ImageReview.preTagConfidence` + `preTagModelVersion`. Reviewer accept/reject
  writes `ConditionLabel` rows (source=model-prediction) as fresh training
  data. Best-effort: if the service is down, the field stays null.

## How it was produced

- `training/train_reflection.py`: ResNet-18 + 2-layer head on verified
  positive/negative `ConditionLabel` rows, 80/20 split, random-crop / h-flip /
  color-jitter aug, early-stop on F1. Output `checkpoints/<run>/weights.pth` +
  `manifest.json`.
- The reflection truth set is the human-curated reflection labels (the
  2026-05-22 curation reduced 318 reviewer-flagged → 241 confirmed positives;
  `ImageReview.reflectionConfirmed` is the ground truth). Note this is a
  **whole-image classifier** ("is there a reflection?"), distinct from the
  camera/tripod **detector** [07] which localizes boxes.

## What it doesn't do / isn't good enough yet

- Only **reflection** is real; dead-fixture and mask models are stubs.
- Classifier-only (no localization) — it says "reflection present," not where.
  The FRCNN detector [07] is the localizer for camera/tripod specifically.
- Small training set (hundreds of labels); precision/recall bounded by data.
- Separate service to operate (must be up on the node; the worker degrades
  gracefully if not).
- Overlaps conceptually with the enhancement Haiku VLM gate [10] — the
  pretagger is a cheap local first-pass "suggestion," the VLM gate is the
  higher-quality enhancement-time gate.

## Intended fine-tune method

1. Reviewer accept/reject of pretag suggestions writes `ConditionLabel`
   (source=model-prediction) — the labeled-data flywheel. The dashboard
   `/verify-labels` and auto-mask pages add more.
2. Re-run `training/train_reflection.py` on the grown `ConditionLabel` set;
   bump version in `manifest.json`; drop the new bundle in `<CHECKPOINT_DIR>/
   reflection/` and reload the service.
3. To add a new condition: create a bundle dir + a `train_<cond>.py`
   (dead-fixture scaffold exists: YOLOv8n box stage + fixture classifier,
   bbox labels bootstrapped from Gemini then human-verified via a
   `/verify-bboxes` UI).
4. The pretagger is its own training stack (not `cloud_train` B200) — small
   models, train wherever a GPU is free.
