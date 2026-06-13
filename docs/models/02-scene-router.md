# 02 — Scene Router (Interior / Exterior)

**What:** Binary classifier that decides interior vs exterior on each
Stage-1 output. Its only production job is to **route** each image to the
correct Stage-2 lane (interior vs exterior NAFNet). Also feeds gallery
ordering and the dashboard's classification surfaces.

**Status:** Production **v4** (native-resolution). 99.27% val / 99.30% on a
1000-image holdout.

**Code:** `pipeline/run_pipeline.py` — `load_classifier()` (~L594),
`classify()` (~L618), `_LetterboxTfm` (~L573). Trainer:
`cloud_train/train_router_native.py`.

## What it does

- Input: Stage-1 output JPG. **v4** letterboxes to the native canvas
  (`input_hw` from the checkpoint, e.g. 2800×4200 aspect-preserving + pad);
  legacy v2/v3 used `Resize(256)→CenterCrop(224)`. ImageNet normalize.
- Model: plain `resnet18(weights=None)` with `fc → Linear(.,2)`. Labels
  `["exterior","interior"]`.
- Output: `(is_interior: bool, confidence: float, label: str)` via softmax.
  No TTA.
- `finish_image()` uses it to pick `s2_int` vs `s2_ext` and names the output
  `{stem}_stage2-{int|ext}.jpg`.
- Checkpoint dict: `{label_names, num_classes, model_state, val_acc,
  input_hw?}`. Env `CLASSIFIER_PATH`. Shipped **in-repo** as
  `pipeline/classifier_v4.pth` (small ~45MB). The loader is `input_hw`-aware
  (v4 letterbox vs legacy center-crop), so v2/v3 checkpoints still load.

## How it was produced

- Lineage v2 (prod, 98.91% on the v4 val) → v3 (99.32%, 2026-06-11) → **v4
  native** (trained on the B200 `cloud_train` harness, 2026-06-12).
- Trained on the COMPLETE human scene labels (`corrected_scene.jsonl`, both
  labeling-app datasets), shoot-split seed 17.
- v4's gain over prod is concentrated where it matters: exterior recall
  97.64% → 98.98%, i.e. it cuts exteriors mis-routed into the interior
  Stage-2 lane roughly in half.
- See `cloud_train/train_router_native.py` + [FINE-TUNING.md](FINE-TUNING.md).

## What it doesn't do / isn't good enough yet

- Binary only — no exterior sub-type (front/rear/aerial) and no "ambiguous"
  class. Exterior sub-scene labeling is a separate, mostly-unfinished staff
  task; the dashboard tracks `exteriorSubType` but the router doesn't predict
  it.
- The ~0.7% errors are the genuinely ambiguous frames (sunrooms, enclosed
  porches, garage interiors, through-window shots). A wrong route sends the
  image through the wrong Stage-2 model — usually tolerable but not ideal.
- No confidence calibration / temperature scaling — the softmax prob is raw
  (a flagged "save for later" item).

## Intended fine-tune method

1. Harvest fresh labels from the dashboard `/classify` interior/exterior axis
   (corrections write `isInteriorCorrected`). Low-confidence + corrected rows
   are the highest-value additions.
2. Append to `corrected_scene.jsonl`, re-run `train_router_native.py` on a
   B200 (it's cheap — ResNet-18, ~26 min/epoch GPU-decode, ~8 epochs).
3. Accept if val + 1000-img holdout beats the deployed number; commit the new
   `classifier_vN.pth` in-repo, repoint `CLASSIFIER_PATH`, new fleet tag.
4. If exterior sub-type is wanted later, that's a *new* head/model, not a
   fine-tune of this one.
