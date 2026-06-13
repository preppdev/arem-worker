# AREM Inference Models — Authoritative Reference

> Written 2026-06-13 so a fresh context (or a new engineer) can pick up the
> entire ML inference surface from a cold repo scan. Each linked doc covers,
> for one model: **how it was produced**, **what it does**, **what it
> doesn't do / isn't good enough yet**, and **the intended fine-tune
> method**. Keep these updated when a model is retrained or replaced.

AREM is a real-estate photo pipeline (Norfolk VA). RAW bracket sets come in
from photographers; the pipeline produces finished, enhanced, upright,
classified, optionally sky-swapped deliverables, then a separate enhancement
pipeline detects and repairs defects. Most inference lives in **arem-worker**
(this repo); a few models live in **arem-pretagger** and the **AREM Editing**
dashboard repo.

## The two pipelines

1. **Editing pipeline** (`worker.py` → `pipeline/run_pipeline.py`) — the
   per-shoot RAW→finished-stills path. Runs on the local fleet (AREM-Compute-1
   `10.2.0.15`, Compute-2 `10.2.0.16`, conda env `arem-photo-ai`) and a RunPod
   cloud backup. Per-image flow inside `finish_image()`:

   ```
   RAW bracket (under/mid/over) ─ lens_correct_16 (rawpy+lensfun, →uint16)
     └─ Stage 1 HDR fusion (NAFNet w16, 9ch→3ch)            [01]
        └─ Scene router (ResNet-18, interior vs exterior)    [02]  → picks Stage-2 lane
           └─ Stage 2 enhancement (NAFNet w32, int|ext)      [04]
              └─ Stage 3 polish (NAFNet w32, optional)       [04]  → overwrites Stage 2
                 └─ Room classifier (ConvNeXt-Base)          [03]  → roomType + conf
                    └─ EXIF embed + 512px thumbnail
   (then a separate pass: auto_upright [06], optional sky-swap [08])
   Single non-bracketed frames take the Stage-1S path [01].
   ```

2. **Enhancement pipeline** (`scripts/enhancement_worker.py`, standalone) —
   detect + repair conditions (reflections, vignette, etc.) on finished
   media. Claims jobs from the dashboard. Uses the camera/tripod FRCNN [07],
   a Claude Haiku VLM gate, and Nano Banana (Gemini) repair [10].

## Model index

| # | Model | Arch | Status | Doc |
|---|-------|------|--------|-----|
| 01 | Stage 1 HDR fusion (+ Stage 1S single-frame) | NAFNet w16, 9→3ch | production | [01-stage1-hdr-fusion.md](01-stage1-hdr-fusion.md) |
| 02 | Scene router (interior/exterior) | ResNet-18 | production v4 (99.3%) | [02-scene-router.md](02-scene-router.md) |
| 03 | Room classifier | ConvNeXt-Base | production v4 (88.8%), informational | [03-room-classifier.md](03-room-classifier.md) |
| 04 | Stage 2 enhancement + Stage 3 polish | NAFNet w32 ×2 | production | [04-stage2-stage3-enhancement.md](04-stage2-stage3-enhancement.md) |
| 05 | Lens correction (not ML) | rawpy + lensfun | production | [05-lens-correction.md](05-lens-correction.md) |
| 06 | Upright / perspective | GeoCalib + Hough fallback | production | [06-upright-perspective.md](06-upright-perspective.md) |
| 07 | Camera/tripod reflection detector | Faster R-CNN R50-FPN v2 | production v3 (mAP50 0.914), detect-only | [07-camera-tripod-detector.md](07-camera-tripod-detector.md) |
| 08 | Sky replacement + window segmentation | U2-Net + matting; OWLv2+SAM2 | production | [08-sky-replacement.md](08-sky-replacement.md) |
| 09 | Condition pretagger (reflection etc.) | ResNet-18 classifier (svc) | partial (reflection live, others stubbed) | [09-condition-pretagger.md](09-condition-pretagger.md) |
| 10 | Enhancement VLM gate + Nano Banana repair + auto-mask | Claude Haiku, Gemini, CLIPSeg | production (reflection/vignette) | [10-enhancement-vlm-repair.md](10-enhancement-vlm-repair.md) |

## Shared conventions (read once)

**Checkpoint storage.** Production checkpoints live in R2 bucket
`arem-training-data`:
- Stage 1: `checkpoints/stage1_jxl_full_v1/best_lpips.pth`
- Stage 2: `models/stage2/may26_{interior,exterior}_w32_b4_4gpu_ep{35,29}_inference.pth`
- Scene router: committed in-repo as `pipeline/classifier_v4.pth` (small, ~45MB)
- Room v4: `models/room_classifier_v4_native.pth` (350MB) + both nodes `~/models/`
- Detector v3: `models/camtrip_detector_v3_native.pth` (173MB) + both nodes `~/models/`

Fleet nodes pull via `scripts/sync_checkpoints.sh`; RunPod containers via
`entrypoint.sh`. Env vars point each model at its checkpoint —
`CHECKPOINT_STAGE1`, `CHECKPOINT_INTERIOR`, `CHECKPOINT_EXTERIOR`,
`CLASSIFIER_PATH`, `CLASSIFIER_ROOM`, `CHECKPOINT_POLISH_{INTERIOR,EXTERIOR}`,
`CAMTRIP_DETECTOR_PATH`. `run_local.sh` sets fleet defaults.

**Fleet deploy.** Tag `worker-fleet-YYYY-MM-DD[x]`, pin it in the editing
repo `deploy/fleet/fleet.manifest.json`, `git checkout` the tag on both
nodes, restart the service (`arem-worker-local`). The dashboard's
`/deploy-worker` page and CI rebuild the RunPod image.

**The cloud training harness** (`cloud_train/`) is the modern path for the
classifier/router/detector. It runs on RunPod B200 ($5.89/hr on-demand) and
is shared by router/room/detector. See **[FINE-TUNING.md](FINE-TUNING.md)**
for the full recipe (data staging, `common.py` GPU-decode + letterbox,
shoot-split seed 17, `run_all.sh` orchestration, self-stop). NAFNet
stages (1/2/3) use an older 4×B200 multi-GPU recipe documented in their
own pages.

**Labels.** Human labels live as append-only journals
(`corrected_scene.jsonl`, `corrected_room.jsonl`) under the dataset roots,
and as first-class columns / `ConditionLabel` rows in the dashboard DB.
Corrections collected via dashboard pages (`/classify`, `/room-collector`,
`/corrections`, `/verify-labels`) are the fine-tune fuel.
