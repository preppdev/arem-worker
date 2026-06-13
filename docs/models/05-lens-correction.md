# 05 — Lens Correction (not ML, but inference-adjacent)

**What:** Demosaics the RAW and applies geometric lens correction
(distortion / vignetting / TCA) before any model sees the frame. Not a
learned model, but it defines the input distribution every downstream model
was trained against, so it's documented here.

**Status:** Production. `rawpy` (demosaic) + `lensfunpy` (correction).

**Code:** `pipeline/lens_correct.py` — `lens_correct_bracket()` (~L131);
`pipeline/run_pipeline.py` — `lens_correct_16()` (~L470).

## What it does

- Decodes the RAW with `rawpy.postprocess`: `use_auto_wb=True`,
  `no_auto_bright=True`, `output_color=sRGB`, `user_flip=-1` (honor EXIF
  orientation — earlier `user_flip=0` rotated portrait shoots 90°). 8- or
  16-bit out (the pipeline uses 16-bit → Stage 1).
- These rawpy settings deliberately match the old Photomatix prep so the
  model input distribution stays aligned with the targets we trained against.
- Looks up the lens in the lensfun DB (`_resolve_lensfun` from EXIF
  make/model/focal/aperture), builds a `lensfunpy.Modifier`, and `cv2.remap`s
  with the geometry-distortion map (Lanczos4).
- Returns `(rgb, info)` where `info` records `lensfun_maker/model`,
  `applied: bool`, `skip_reason`.
- **Graceful skip:** lenses with no lensfun profile, or lookup/apply errors,
  bypass correction and continue (rawpy demosaic still yields a usable image)
  — `info.skip_reason` is set. As of 2026-05-04 the POC-excluded lenses
  (`EXCLUDED_LENS_SUBSTRINGS`, e.g. Sony FE 12-24 GM) are bypass-and-continue
  rather than fatal.
- RAW formats: `.arw .cr2 .cr3 .nef .dng .raf .rw2`. Non-RAW (jpg/tiff/png/
  jxl/webp) get PIL decode + 8→16 bit promotion.

## How it was produced

Hand-built around rawpy + lensfun; no training. EXIF parsing + lensfun
resolution tuned against the AREM camera/lens fleet (mostly Sony bodies).

## What it doesn't do / isn't good enough yet

- **Coverage gaps:** any lens lensfun doesn't profile gets no geometric
  correction (silent skip). The Sony 12-24 GM is explicitly excluded.
- No learned correction — purely the lensfun analytic model, so it can't fix
  what lensfun doesn't model.
- Historical EXIF bugs bit non-Sony bodies (NEF/CR EXIF parsing fix
  2026-06-05, forward-only — pre-fix non-Sony shoots may need backfill;
  memory `pipeline-nef-exif-bug`).
- WB/brightness are rawpy auto, not scene-aware.

## Intended fine-tune method

N/A (not a model). To improve: add lensfun profiles for unprofiled lenses,
extend `_resolve_lensfun` matching, or (bigger) introduce a learned
distortion-correction step — but that would change the input distribution and
require retraining Stage 1/2 against the new look. Treat any change here as a
pipeline-input change that invalidates downstream training parity.
