# 07 — Camera / Tripod Reflection Detector

**What:** Object detector that finds the photographer's camera body and
tripod legs reflected in mirrors / glass / glossy surfaces. Drives the
enhancement pipeline's reflection-correction flags.

**Status:** Production **v3** (native-res), mAP50 **0.9144** (beats the prior
0.906). **Detect-only trial** — positives become review flags; a human
dispatches the Nano Banana repair (automation is one config flag away).
Fully in-house, commercially unconstrained.

**Code:** runs in `scripts/enhancement_worker.py` as the `frcnn-local`
detector — `_load_frcnn()` (~L171), `call_frcnn_detector()` (~L195). Trainer:
`cloud_train/train_detector_native.py`; standalone runner
`cloud_train/run_detector.sh`.

## What it does

- Model: `torchvision.fasterrcnn_resnet50_fpn_v2(num_classes=3)` — classes
  `1=camera, 2=tripod` (+ background). `min_size=2800, max_size=4200` (native
  res for small reflected objects).
- Input: a finished production JPG (decoded to a CUDA tensor). Inference mode,
  score threshold **0.5** at the worker; the dashboard applies a higher flag
  floor.
- Output: fractional boxes `[{x,y,w,h,label,score}]` (0–1), POSTed to the
  dashboard `/api/internal/camtrip-detections` → `CorrectionFlag` rows (red
  outline on `/delivery`, pooled on `/corrections`). Flag threshold is
  `maxScore ≥ 0.6` (step `confidenceFloor` in `lib/enhancement-steps.ts`).
- Lazy-loaded and **released after each enhancement job** (GPU shared with the
  editing worker). ~5GB peak VRAM (C2 0.84s/img, C1 0.51s/img).
- Checkpoint dict: `{model_state, num_classes:3, min_size:2800, max_size:4200,
  metrics:{mAP50}, version:"v3_native_b200"}`. Env `CAMTRIP_DETECTOR_PATH`
  (default `/home/jordan/models/camtrip_detector_v3_native.pth`). R2
  `models/camtrip_detector_v3_native.pth` (173MB) + both nodes `~/models/`.

## How it was produced

- Pivot history (memory `detector-pivot-verdict`): FRCNN real-only baseline
  0.736 → synthetic-aug 0.723 (didn't help) → Rex-Omni / LocateAnything
  zero-shot (high recall, bad precision, **non-commercial licenses**) →
  Qwen2.5-VL zero-shot (best zero-shot but resolution-bound recall) → **train
  FRCNN at native res** = v3 0.9144, the winner, and license-clean.
- Trained on the `cloud_train` B200 harness (2026-06-12), 40 epochs, batch 4,
  on `detector/ann_{train,val}_native.json` (1,508 train / 331 val,
  native-rescaled boxes). Self-contained AP50 eval. torchvision BSD-3 arch +
  COCO-pretrained init + 100% AREM photos/labels → **fully in-house, no usage
  restriction.**
- Label hygiene caveat (memory): the train set's largest-area boxes had ~25%
  phantom "tripod on empty shower glass" boxes; a hygiene pass + shadow policy
  were recommended before further fine-tunes.

## What it doesn't do / isn't good enough yet

- **Camera class is the weaker of the two** (camera ~0.84, tripod ~0.97–0.99).
  Reflected camera bodies are small/ambiguous.
- **Detect-only** — it does not repair; a human reviews each flag and clicks
  "Send to Nano Banana." We're in a multi-week trial measuring false-positive
  rate before automating.
- Only camera + tripod — not photographer body/hands, light stands, or other
  gear (the Haiku VLM gate [10] covers the broader "reflection" condition).
- Runs only in the enhancement pipeline, not inline in the editing pipeline.
- Native-res inference is heavy (~5GB, ~0.5–0.8s/img).

## Intended fine-tune method

1. The **trial itself is the label engine**: every "false positive" click on
   `/corrections` is a hard negative; every confirmed+repaired flag is a
   positive. These accumulate as `CorrectionFlag` rows.
2. Do the **label-hygiene pass first** (re-verdict the largest train boxes,
   decide the shadow/empty-glass policy) before retraining — see
   `detector-pivot-verdict` memory and `/home/jordan/qwen_eval/` audit grids.
3. Export confirmed boxes + hard negatives → rebuild `ann_train_native.json`,
   re-run `train_detector_native.py` on a B200 (40 ep). Focus the new data on
   camera-class examples and the false-positive surfaces.
4. Ship to `models/camtrip_detector_v3_native.pth` (bump version) + `~/models/`
   on both nodes; the enhancement worker picks it up via env.
5. **To automate repair after the trial:** add a `repair` spec
   (`nano-banana-pro`) to the `camera-tripod-reflection` step in
   `lib/enhancement-steps.ts` — no worker change needed.
