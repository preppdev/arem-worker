# 06 — Upright / Perspective Correction

**What:** Levels and de-keystones finished images (rotate to horizon,
correct vertical convergence). Runs as a post-pipeline pass over the
finished JPGs.

**Status:** Production. **GeoCalib** (deep camera calibration) primary, with a
classical Hough fallback. GeoCalib is REQUIRE'd by default (won't silently
fall back).

**Code:** `pipeline/auto_upright.py` — `_get_geocalib_model()` (~L93),
`_gc_predict_and_warp()` (~L357), `upright_one()` (~L438), Hough
`estimate_rotation()` (~L177). Eval: `scripts/upright_eval.py`.

## What it does

- **GeoCalib** (ETH CVG, ECCV 2024) predicts gravity (roll, pitch), vertical
  FOV, and an up-confidence from a 640px-downscaled RGB. From roll/pitch it
  builds a homography `H = K · Rz(roll) · Rx(pitch) · K⁻¹` and warps
  (Lanczos4, replicate border), then crops to the largest inscribed
  rectangle.
- **Gating + damping** (env-tunable): `UPRIGHT_GC_CONF_GATE` (default 0.03 —
  below this up-confidence, skip); `UPRIGHT_GC_VK_SCALE` (default 0.7 — damp
  pitch→vertical-keystone to avoid overshoot); `UPRIGHT_GC_ROT_CLIP` (default
  5° — clip |roll|, larger outliers zeroed). `AREM_UPRIGHT_REQUIRE=1` makes
  GeoCalib-load failure abort the job instead of falling back.
- **Hough fallback** (`AREM_UPRIGHT_ENGINE=hough` or if not REQUIRE'd and
  GeoCalib unavailable): LSD/Hough line segments → length-weighted median tilt
  of near-vertical lines → rotate only.
- Returns rich metrics (estimated vs applied angle, clamp reason, crop ratio,
  engine, gc_* fields, elapsed).

## How it was produced

- GeoCalib is an off-the-shelf pretrained model (pip package in the worker
  env) — not trained by us. The AREM work is the gating/damping/homography/
  crop wrapper and the eval harness.
- Tuned against a 477-pair eval set vs Lightroom Auto Upright
  (`scripts/upright_eval.py`). The REQUIRE+clamp design came after a
  regression where a lost-geocalib build silently fell back to Hough (memory
  `upright-perspective-todo`, resolved 2026-06-11). Defaults
  (conf_gate 0.03 / vk_scale 0.7 / rot_clip 5°) are the tuned values.
- Per the eval: p95 |Δrot| vs LR ≈ 1.12° (was 1.74° pre-GeoCalib), p95 |Δvk|
  ≈ 2.44°; apply rates roughly track LR.

## What it doesn't do / isn't good enough yet

- Conservative by design (low apply rate, damping, 5° clip) — it under-corrects
  rather than risk a bad warp. Strong keystones may be left partially
  uncorrected.
- Crop-to-inscribed-rectangle loses edge pixels on large corrections.
- GeoCalib runs on a 640px downscale — fine for global geometry, but it's not
  using full detail.
- A separate Hough-window "re-upright" repair for historically-crooked shoots
  is **shelved** (92 jobs / 179 frames flagged; scripts staged on C1, run on
  Jordan's go — memory `hough-window-repair`).
- No drone-specific handling — drone aerials have different geometry (nadir/
  oblique) and shouldn't be uprighted like interiors (relevant to the drone
  pipeline build).

## Intended fine-tune method

GeoCalib isn't fine-tuned by us. To improve correction quality:
1. Re-tune `UPRIGHT_GC_*` against an expanded eval set (`upright_eval.py`
   compares to LR Auto on aligned pairs) — this is the main lever.
2. If a learned AREM-specific upright is ever wanted, that's a new model
   (train on our images + LR-Auto targets), not a GeoCalib fine-tune.
3. For drone: gate upright OFF (or use a drone-specific policy) — interiors'
   vertical-line assumptions don't hold for aerials.
