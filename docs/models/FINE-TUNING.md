# Cloud fine-tuning harness (router / room / detector)

The shared B200 training path for the torchvision classifiers and the
detector. NAFNet stages (1/2/3) use a different, older multi-GPU recipe —
see their own docs. This page is the reusable recipe; the per-model docs
say which knobs to change.

## Where it lives

`cloud_train/` in this repo:
- `common.py` — dataset loading (journals + manifests), `shoot_split`
  (seed 17 — splits by SHOOT so the val set has no train-shoot leakage),
  `Letterbox` (CPU) + `GpuPrep` (GPU-side nvJPEG decode → letterbox canvas →
  hflip + scalar brightness/contrast jitter; `NATIVE_HW = (2800, 4200)`).
- `train_router_native.py` — ResNet-18 interior/exterior.
- `train_room_native.py` — ConvNeXt-Base room classifier.
- `train_detector_native.py` — Faster R-CNN R50-FPN v2 camera/tripod.
- `run_all.sh` — sequential router → room → detector with `.done`-marker
  skip logic + a live-log shipper to R2 every 4 min + pod self-stop.
- `run_detector.sh` — standalone detector runner.

## Data staging (R2)

`r2:arem-training-data/cloud-train/`:
```
originals/{aligned,may}/<scene>/ours/<slug>.jpg   # full-res ~2800x4200, 47,946 imgs / 91GB
meta/{aligned,may}/...                             # manifests + journals
meta/.../corrected_scene.jsonl                     # interior/exterior labels (router)
meta/.../corrected_room.jsonl                      # room labels (room classifier)
detector/ann_{train,val}_native.json              # FRCNN annotations (native-scaled boxes)
code/*.py *.sh                                     # the staged trainer bundle
runs/<UTC-stamp>/{router,room,detector}/           # outputs + live-logs/
```

Re-stage code after editing a trainer: from the **editing** repo,
`npx tsx --env-file=.env.local <script>` using `putR2BufferTo(...,
"arem-training-data")` to write `cloud-train/code/<file>`. (Script must run
from inside the editing repo dir for module resolution.)

## Launch / iterate

1. Provision a B200 on RunPod (GraphQL `podFindAndDeployOnDemand`; secure
   cloud; `RUNPOD_API_KEY` via `npx vercel env pull` in the editing repo).
   The pod's `dockerArgs` bootstrap pulls code fresh from R2 on each start,
   so a crash-loop self-heals after you re-stage fixed code.
2. The container runs `run_all.sh`: configures rclone, pulls the dataset
   (~170GB), runs each stage, ships results + logs, then self-stops.
3. Monitor by tailing `runs/<stamp>/live-logs/<model>.log` from a node
   (`rclone cat`).
4. To restart one stage at a new config: edit the trainer, re-stage to
   `cloud-train/code/`, `podReset` the pod. Stages with an accepted
   checkpoint / `.done` marker are skipped.

## Hard-won gotchas (don't re-learn these)

- **GPU-decode is required.** CPU JPEG decode was input-bound (31% GPU
  util, 61-min router epochs). `GpuPrep` uses
  `torchvision.io.decode_jpeg(device="cuda")` → 26-min epochs.
- **`TrivialAugmentWide` crashes on CUDA tensors** (libtorch hard abort).
  Use scalar brightness/contrast jitter + hflip instead.
- **DataLoader shm exhaustion** at native res: set sharing strategy
  `file_system`, `pin_memory=False`.
- **`.done` markers, not "best.pt exists".** best.pt is written from epoch 1,
  so resume logic must key on an explicit done marker (router is exempt —
  operator early-accept).
- **Write trainers locally + scp/stage; never heredoc-over-ssh** (single-quote
  stripping corrupts `'val_acc'` → NameError).
- **`run_detector.sh` lacks the self-stop** that `run_all.sh` has — a
  finished standalone detector pod will restart and redundantly retrain.
  Add `runpodctl stop` or stop the pod manually.

## Checkpoint dict formats (so the worker loader matches)

- Router: `{label_names, num_classes, model_state, val_acc, input_hw?}`
- Room v4: `{model, classes, input_hw, acc, arch:"convnext_base", tta:"hflip", letterbox:True, version}`
- Detector: `{model_state, num_classes:3, min_size, max_size, metrics:{mAP50}, version}`

The worker's `load_*` functions in `pipeline/run_pipeline.py` (router/room)
and `scripts/enhancement_worker.py` (detector) are `input_hw`/format-aware;
keep these dict shapes stable or update both ends together.
