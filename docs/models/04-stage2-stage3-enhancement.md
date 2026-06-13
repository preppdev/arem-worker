# 04 — Stage 2 Enhancement + Stage 3 Polish

**What:** Stage 2 is the main image-enhancement model — takes the Stage-1
fused image and produces the finished, graded look. Two checkpoints
(interior, exterior); the scene router [02] picks which. Stage 3 is an
optional second NAFNet pass that polishes Stage-2 output toward a
vendor/Lightroom style.

**Status:** Production. NAFNet width 32, bf16. Stage 3 optional (per-route).

**Code:** `pipeline/run_pipeline.py` — `load_stage2()` (~L525),
`stage2_infer()` (~L901), `load_stage3()`/`stage3_infer()` (~L877, thin
wrappers over Stage 2). Arch `pipeline/models/nafnet.py`.

## What it does

- Input: Stage-1 output JPG (Stage 3 takes Stage-2 output). Read via cv2,
  dims snapped to ×16, single pass (no tiling), `MAX_MP` cap (~12 MP).
- Model: `NAFNet` with arch hyperparams read from the checkpoint, defaulting
  to the may26 **w=32** config (`middle_blk_num=12, enc=[2,2,4,8],
  dec=[2,2,2,2], use_residual`). Weights + activations cast to **bf16** — fp32
  at 12 MP with Stage1 + router + both Stage-2 models resident exceeds the
  ~48GB A40/A6000 budget; NAFNet has no softmax attention so bf16 is safe.
- Output: 8-bit RGB JPEG (quality 92). No TTA.
- Routing (`finish_image`): `is_int` from the scene router → `s2_int` else
  `s2_ext`; output `{stem}_stage2-{int|ext}.jpg`.
- **Stage 3**: if `CHECKPOINT_POLISH_{INTERIOR,EXTERIOR}` is set and exists,
  run it on the Stage-2 JPG and `replace()` Stage-2 in place; record the
  variant dir name in `ImageReview.stage3Variant`. A Stage-3 failure is
  caught and the Stage-2 output is kept.
- Checkpoints: prefer `ema_state_dict` (`.shadow`) else `model_state_dict`.
  Env `CHECKPOINT_INTERIOR` / `CHECKPOINT_EXTERIOR` /
  `CHECKPOINT_POLISH_{INTERIOR,EXTERIOR}`. R2
  `models/stage2/may26_{interior,exterior}_w32_b4_4gpu_ep{35,29}_inference.pth`.

## How it was produced

- The **may26** line (interior ep35, exterior ep29), NAFNet w32, batch 4,
  4×GPU. Trained on aligned pairs (pipeline-input → target look), matching
  the AutoHDR/vendor target first ("match before differentiate", memory
  `feedback_match_before_differentiate`).
- Interior and exterior are separate fine-tunes from a shared base so each
  lane specializes (interiors: window pull, white balance; exteriors: sky,
  greens, façades).
- Stage 3 is the same arch fine-tuned on a different objective
  (pipeline-output → vendor/Lightroom-style target) — a stylistic top-coat
  kept separate so it can be toggled/swapped without retraining Stage 2.
- Older 4×B200 recipe (not the `cloud_train/` harness).

## What it doesn't do / isn't good enough yet

- **No tiling** — same 12 MP ceiling as Stage 1; large files must be downsized.
- Single global model per lane — no per-room or per-condition specialization
  (a kitchen with mixed tungsten+daylight and a bright bedroom get the same
  interior model). The room classifier [03] is the future lever for
  conditional routing.
- Trained to parity with the vendor, not clearly past it on hard conditions;
  differentiation is the backlog.
- Stage 3 variants are experimental — they can over-stylize; that's why
  they're optional/per-route and failure-tolerant.

## Intended fine-tune method

1. Use the dashboard **vendor-match review** (`ImageReview.vendorMatch` +
   `vendorMatchReasons` on `/classify`): reviewers grade our output vs the
   vendor's and tag the gap dimension (white_balance, brightness, contrast,
   enhancement, perspective_rotation…). A pile of one tag = a targeted
   training set.
2. Build aligned input→target pairs for that condition; fine-tune the
   relevant lane (interior or exterior) from the may26 checkpoint, NAFNet w32,
   L1 + perceptual.
3. Validate with held-out shoots on the vendor-match axis before shipping.
4. New `models/stage2/<name>.pth`, repoint the lane env var, new fleet tag.
5. For a stylistic shift (not a defect), prefer a new **Stage 3** variant over
   touching Stage 2 — lower risk, toggleable.
