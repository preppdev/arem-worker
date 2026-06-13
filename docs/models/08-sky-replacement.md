# 08 — Sky Replacement + Window Segmentation

**What:** Two related capabilities. (a) **Sky replacement** — segment the sky
in an exterior, matte it cleanly, and composite a chosen sky plate. (b)
**Window segmentation** — detect + mask windows (used for window-pull /
masking work). Both produce masks that downstream compositing uses.

**Status:** Production tooling, run per-shoot (sky-swap into
`<shoot>/08-Test-Edit/sky-swap/`). Not a single learned model — a pipeline of
off-the-shelf models + classical CV.

**Code:** `scripts/sky_replace_v8.py` (sky), `scripts/generate_window_masks.py`
+ `scripts/sam_box_mask.py` (windows).

## What it does

### Sky replacement (`sky_replace_v8.py`)
1. **Sky segmentation:** `skyseg_u2net.onnx` (U2-Net, binary sky/non-sky;
   onnxruntime CUDA) at 320×320 → upscaled, guided-filter refined →
   `alpha`. Path env `SKYSEG_ONNX` (default `/home/jordan/sky_models/`).
2. **Matting:** closed-form matting (`pymatting.estimate_alpha_cf`) on a
   trimap, downscaled to ≤1600px then guided-filter upsampled. Edge-aware
   alpha sharpen (gamma ramp by source gradient).
3. **Color-range expansion:** Lab ΔE<40 from confident-sky median, gated to
   the upper 65% of frame + foliage-adjacency + low gradient + low current
   alpha — recovers sky seen through tree-canopy gaps without grabbing
   ground/architecture.
4. **Decontamination + composite:** spill correction, then alpha-blend the new
   sky plate over the decontaminated source.

### Window segmentation (`generate_window_masks.py`, `sam_box_mask.py`)
- **Detect:** OWLv2 (`google/owlv2-base-patch16-ensemble`), text prompt
  `"a window"`, threshold 0.15 → boxes.
- **Segment:** SAM (`facebook/sam-vit-base`) or **SAM2**
  (`sam2_hiera_base_plus` at `/home/jordan/sky_models/`) box→mask, flood-fill
  hole-closing + dilation.

## How it was produced

- All models are **off-the-shelf** (U2-Net skyseg from RapidRaw; OWLv2/SAM/
  SAM2 from HuggingFace) — not trained by us. The AREM work is the
  matting/expansion/decontam/composite pipeline (`v8` is the current iteration
  — the version number is the tuning lineage) and the detect→segment glue.
- Sky plates / "sky library" are curated in the dashboard (`/sky-library`).
- Winners chosen by eye on real shoots; a sky-quality gate decides whether a
  shot is worth swapping.

## What it doesn't do / isn't good enough yet

- Hard edges (fine branches, railings, fences against sky) are the matting
  weak point even with closed-form matting + edge-aware sharpen.
- Color-range expansion is heuristic (Lab ΔE + gates) — tuned thresholds, not
  learned; can over/under-grab in unusual foliage/lighting.
- Sky-quality gate + plate selection are not fully automated (human in the
  loop on plate choice).
- Window masking is detect-then-segment with generic models — reflective or
  mullioned windows can split/miss.
- All these models are GPU-heavy and lazy-loaded; they compete with the
  editing worker for VRAM (released after the job).

## Intended fine-tune method

- **Sky:** the off-the-shelf U2-Net could be replaced/fine-tuned on AREM
  exteriors with hand-corrected sky mattes if edge quality becomes the
  bottleneck — but the bigger wins so far have been in the matting/expansion
  heuristics, not the segmenter. Tune `sky_replace_vN` thresholds against a
  held-out set of real shoots first; only train a segmenter if heuristics
  plateau.
- **Windows:** if OWLv2+SAM misses too often, collect window boxes/masks (the
  same OWLv2→SAM output, human-corrected) and fine-tune SAM2 or a small
  detector. There's an auto-mask testing harness ([10]) for comparing
  segmenters before committing.
- These are not on the `cloud_train` B200 harness; they'd need their own
  training scripts.
