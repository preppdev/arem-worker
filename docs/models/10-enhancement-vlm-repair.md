# 10 — Enhancement: VLM Gate + Nano Banana Repair + Auto-mask

**What:** The detect→repair machinery of the enhancement pipeline for
conditions handled by external models: a **Claude Haiku VLM** gate decides
"is this condition present (and where)?", then **Nano Banana** (Google
Gemini image model) performs the targeted repair. Plus a **CLIPSeg**
auto-mask testing harness for evaluating segmenters.

**Status:** Production for `reflection` and `vignette` (detect→repair). The
camera/tripod step [07] is detect-only. External API models (no checkpoints
we own).

**Code:** `scripts/enhancement_worker.py` — `call_detector()` (vlm-haiku,
~L234), `call_repair()` (nano-banana, ~L289), `render_annotated()`,
`region_to_bbox()`/`boxes_union_bbox()`. Dashboard realtime path:
`AREM Editing/lib/enhancement.ts` (`callGeminiImage`, `renderAnnotatedImage`,
`wrapPrompt`). Step registry: `lib/enhancement-steps.ts`. Prompts:
`lib/settings-defaults.ts` (overridable via `AppSetting`). Auto-mask:
`scripts/auto_mask_test.py`.

## What it does

### Haiku VLM detector (`vlm-haiku`)
- Model `claude-haiku-4-5-20251001`. Image (downscaled to ≤1600px for the
  5MB cap) + a prompt (text from `AppSetting[detector.promptKey]`, e.g.
  `reflection-gate-v3`, `vignette-gate-v1`).
- Returns JSON `{<condition>: bool, confidence, region (3×3 grid cell),
  reason}`. `confidenceFloor` (if set) gates promotion to repair.
- `region_to_bbox()` maps the 3×3 cell → a fractional bbox (~38% of frame,
  slight overlap). FRCNN detections instead use `boxes_union_bbox()` (+3% pad).

### Nano Banana repair (`nano-banana-pro`)
- Model `gemini-3-pro-image-preview`, fallback `gemini-2.5-flash-image` on
  503/UNAVAILABLE. `imageConfig.imageSize = "2K"` (production tier; Pro
  supports 1K/2K/4K — **set to 2K** as of 2026-06-13, was 4K).
- If a bbox exists, render a **magenta rectangle** marker (`(255,0,220)`,
  8px) on the input and wrap the prompt: "edit only inside the magenta
  rectangle, don't reproduce the rectangle, preserve everything outside." The
  marker focuses the model; it's never part of the output.
- Repair prompt text from `AppSetting[repair.promptKey]` (reviewer-editable;
  defaults in `settings-defaults.ts`).
- Output bytes uploaded to R2; for the realtime reviewer path the result
  auto-accepts → archives a before/after training pair → overwrites the
  deliverable (and refreshes Cloudflare Images — see the editing repo's
  `correction-archive`/CF-Images gotcha).

### Flow
Worker claims `EnhancementJob` from the dashboard, gets resolved step specs
(detector+repair with prompt text inlined), iterates images: detect → (if
positive) repair → POST `EnhancementRequest`. Realtime single-image path:
`/api/enhancement` → `processEnhancementRequest`.

### Auto-mask testing (`auto_mask_test.py`)
CLIPSeg (`CIDAS/clipseg-rd64-refined`) or Florence-2 — text-prompted
segmentation to evaluate mask candidates for a condition without training,
results to the `/auto-mask/[condition]` dashboard gallery. A *testing* tool,
not in the production repair path.

## How it was produced

- The detector/repair **models are external APIs** — nothing trained by us.
  The AREM work is: the step registry + spec plumbing, the prompts
  (`reflection-gate-v3` reports ~87.7% recall on 235 curated positives per
  the settings comments), the magenta-marker bbox technique, and the
  detect→repair→archive flow.
- Cost: Nano Banana ~$0.24/image at 4K (now 2K, cheaper); RunPod is not
  involved (pure API).

## What it doesn't do / isn't good enough yet

- VLM gate region is a coarse 3×3 grid — the repair bbox is approximate
  (~38% of frame) unless a real detector ([07]) supplies a tight box.
- External-API dependency: rate limits, latency (~30–60s/repair), cost, and
  occasional non-deterministic edits.
- Repair quality varies — Nano Banana can alter pixels outside intent despite
  the marker prompt; that's why the realtime path keeps a human accept step.
- Video repair via this path is unproven; conditions beyond reflection/
  vignette/camera-tripod aren't wired (glare, dead-fixture, color-cast,
  finger, photographer-shadow have prompts but are queued, not realtime).
- Prompts are hand-tuned text — no systematic prompt eval harness.

## Intended fine-tune method

- **No model training** — improvement is **prompt engineering** (edit the
  `AppSetting` prompt for a condition; `promptVersion` tracks it) and
  **threshold tuning** (`confidenceFloor`).
- The before/after pairs archived on every accepted correction
  (`enhancement-examples/<condition>/...` in R2) are a growing **supervised
  dataset** — the long-term play is to **distill** these external-API repairs
  into an in-house repair model (like we did for the detector: replace the
  non-commercial/external dependency once we have enough pairs). That would be
  a new training effort, not a fine-tune of Haiku/Gemini.
- To promote a queued condition to realtime: build/verify its repair prompt,
  add it to `REALTIME_CONDITIONS`, validate on sample images.
