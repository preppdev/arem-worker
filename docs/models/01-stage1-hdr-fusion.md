# 01 — Stage 1: HDR Fusion (+ Stage 1S single-frame)

**What:** Fuses a 3-frame exposure bracket (under/mid/over) into one
well-exposed 8-bit RGB image. This is the core "merge the brackets" step —
the equivalent of Photomatix/AutoHDR's tone-merge, learned.

**Status:** Production. NAFNet, width 16, 9-channel input.

**Code:** `pipeline/run_pipeline.py` — `load_stage1()` (~L450), `stage1_infer()`
(~L498). Arch: `pipeline/models/nafnet.py`.

## What it does

- Input: the three lens-corrected, demosaiced 16-bit frames (under EV<0, mid
  |EV|≤1, over EV>0), each `uint16` 0–65535, normalized to `[0,1]`,
  concatenated channel-wise → **9 channels**. Dims snapped to a multiple of
  16; capped at `MAX_MP` (default 12 MP, env `MAX_MP_OVERRIDE`).
- Model: `NAFNet(in_channels=9, out_channels=3, width=16, middle_blk_num=12,
  enc=[2,2,4,8], dec=[2,2,2,2], use_residual=True, residual_start=3)`.
  Residual is added from the mid bracket, so the net learns the *correction*
  to a baseline exposure rather than the whole image.
- Inference: single pass, fp16 autocast, `clamp(0,1)` → `uint8`. No tiling,
  no TTA.
- Output: 8-bit RGB handed to the scene router + Stage 2.
- Checkpoint dict: `ema_state_dict` (preferred; `.shadow` sub-key) or
  `model_state_dict`; carries `epoch`, `metrics.lpips`. Env `CHECKPOINT_STAGE1`.
  R2 `checkpoints/stage1_jxl_full_v1/best_lpips.pth`.

### Stage 1S (single-frame path)
Non-bracketed orphan frames (single-frame uploads, or a bracket that lost its
mates) take a degenerate path (`run_pipeline.py` single-frame loop, ~L1201):
`lens_correct_16` already did WB + lens correction + sRGB gamma, so "Stage 1"
is just `m16 >> 8` (16→8 bit). Recorded as `"stage1": "single-develop"` in
modelVersions. A distilled single-input Stage-1S model
(`checkpoints/stage1_single_distill_w32/`) exists for higher quality on
singles; falls back to degenerate-develop if absent.

## How it was produced

- Trained on **aligned bracket→target pairs** (`r2:arem-training-data/training_pairs`
  only — the alignment matters; skipping it caused global blur, see memory
  `feedback_use_aligned_training_pairs`). Target = the AutoHDR/vendor merge we
  were matching.
- LPIPS-led objective (best checkpoint by lpips). The `jxl_full_v1` lineage is
  the production line.
- Older multi-GPU (4×B200) recipe, not the `cloud_train/` harness.

## What it doesn't do / isn't good enough yet

- **No tiling** — a >12 MP image raises rather than degrades. Sony 61MP shots
  must be downsized first; raising `MAX_MP_OVERRIDE` risks VRAM with all the
  other models resident.
- Trained to *match* AutoHDR, not beat it per-condition — per-condition
  fine-tunes (harsh window light, mixed color temp, night) are the
  differentiation backlog (memory `feedback_match_before_differentiate`).
- Single-frame quality is below true-bracket quality (no exposure information
  to fuse); the Stage-1S distill narrows but doesn't close the gap.
- Bracket detection upstream is EXIF-gap based; mis-grouped brackets feed
  garbage 9-ch input.

## Intended fine-tune method

1. Collect more **aligned** bracket→target pairs for the weak condition
   (export from shoots where reviewers graded our output below "perfect" with
   a reason tag — the `vendorMatchReasons` queue on `ImageReview` is built for
   exactly this: a pile of frames tagged `white_balance` → a WB sub-model).
2. Resume from `best_lpips.pth`, same NAFNet w16 9-ch arch, LPIPS + L1.
3. Keep the alignment step — never train on unaligned pairs.
4. Validate against held-out shoots with the side-by-side vendor-match review
   (`/classify` vendorMatch axis) before shipping.
5. Ship as a new `stage1_*` checkpoint, repoint `CHECKPOINT_STAGE1`, new fleet
   tag.

> NAFNet is production for all NAFNet stages — Restormer is retired (memory
> `feedback_nafnet_only`).
