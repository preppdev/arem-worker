"""Sky quality analyzer — decides whether a shoot needs sky-swap.

Per-frame metrics, computed inside the sky mask only:
  saturation (HSV S)  — clear blue / golden skies have high S;
                        overcast / hazy skies have low S
  Lab b  — negative for blue sky, positive for yellow/golden, ~0 for gray
  Lab a  — positive for red/golden, ~0 otherwise
  value  — HSV V; very-bright low-S is the overcast signature
  sky_coverage  — fraction of the frame the sky mask occupies; used as a
                  per-frame weight when aggregating to shoot-level

Per-frame scores:
  blue_score    — high for clear-blue sky (S high, Lab b negative)
  golden_score  — high for warm/golden hour (S high, Lab a+b positive)
  dull_score    — 1 - max(blue_score, golden_score) — the "needs swap" axis
  swap_score    — dull_score (current MVP; later we can fold in coverage)

Shoot decision:
  Weighted-by-coverage mean of dull_score across exteriors. If mean ≥
  SWAP_THRESHOLD (default 0.55) OR ≥50% of frames individually score
  ≥0.55, trigger sky-swap. The OR-of-two rule catches "mostly good sky
  but one or two clouds-only frames" without triggering on a single bad
  outlier, and catches "uniformly dull" without needing a single
  individual frame to be extreme.

Tunable via env:
  SKY_QUALITY_SWAP_THRESHOLD (default 0.55)
"""
from __future__ import annotations

import os
import statistics
from pathlib import Path

import cv2
import numpy as np

SWAP_THRESHOLD = float(os.environ.get("SKY_QUALITY_SWAP_THRESHOLD", "0.55"))
MIN_SKY_COVERAGE = 0.01   # below this, ignore the frame (no usable sky)


def _sky_mask_for(img_bgr: np.ndarray) -> np.ndarray:
    """Get a binary sky mask. Reuses the same U2NET-based skyseg the
    sky-swap pipeline uses, then thresholds to a hard mask."""
    # Local import — keeps cold-start cheap when sky_quality is not used.
    from scripts.sky_replace_v9_baseline import skyseg_alpha
    alpha = skyseg_alpha(img_bgr)
    return alpha >= 0.5


def analyze_frame(img_bgr: np.ndarray) -> dict:
    """Compute sky-quality metrics for one image. Returns a dict; key
    `has_sky=False` when no usable sky was detected."""
    h, w = img_bgr.shape[:2]
    try:
        sky = _sky_mask_for(img_bgr)
    except Exception as e:
        return {"has_sky": False, "error": f"skyseg: {type(e).__name__}: {e}"}
    coverage = float(sky.mean())
    if coverage < MIN_SKY_COVERAGE:
        return {"has_sky": False, "sky_coverage": coverage,
                "reason": "below_min_coverage"}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    s = float(hsv[..., 1][sky].mean())
    v = float(hsv[..., 2][sky].mean())
    a = float(lab[..., 1][sky].mean()) - 128.0
    b = float(lab[..., 2][sky].mean()) - 128.0

    # Normalized component scores in [0, 1]
    # blue_score: saturated AND Lab-b strongly negative
    blue_norm  = max(0.0, min(1.0, s / 60.0)) * max(0.0, min(1.0, (-b) / 25.0))
    # golden_score: saturated AND warm tones (a+b both positive)
    warm_chroma = max(0.0, a) + max(0.0, b)
    golden_norm = max(0.0, min(1.0, s / 60.0)) * max(0.0, min(1.0, warm_chroma / 30.0))
    # dull_score: 1 - max(blue, golden); clamped
    dull_score = 1.0 - max(blue_norm, golden_norm)
    dull_score = max(0.0, min(1.0, dull_score))

    return {
        "has_sky": True,
        "sky_coverage": round(coverage, 4),
        "saturation": round(s, 2),
        "value": round(v, 2),
        "lab_a": round(a, 2),
        "lab_b": round(b, 2),
        "blue_score": round(blue_norm, 3),
        "golden_score": round(golden_norm, 3),
        "dull_score": round(dull_score, 3),
        "swap_score": round(dull_score, 3),
    }


def should_swap_shoot(exterior_image_paths: list[Path],
                      swap_threshold: float = SWAP_THRESHOLD) -> tuple[bool, dict]:
    """Aggregate per-frame analysis across a shoot. Returns (trigger, summary).

    Trigger if EITHER:
      - coverage-weighted mean dull_score ≥ swap_threshold
      - OR ≥50% of analyzed frames individually have dull_score ≥ swap_threshold
    """
    per_frame: list[dict] = []
    for p in exterior_image_paths:
        try:
            img = cv2.imread(str(p))
        except Exception:
            img = None
        if img is None:
            per_frame.append({"path": str(p), "has_sky": False, "error": "imread"})
            continue
        m = analyze_frame(img)
        m["path"] = p.name if isinstance(p, Path) else str(p)
        per_frame.append(m)

    usable = [m for m in per_frame if m.get("has_sky")]
    summary: dict = {
        "n_frames":       len(per_frame),
        "n_with_sky":     len(usable),
        "swap_threshold": swap_threshold,
        "per_frame":      per_frame,
    }
    if not usable:
        summary["reason"] = "no_usable_sky_in_any_frame"
        return False, summary

    weights = [m["sky_coverage"] for m in usable]
    scores  = [m["dull_score"]   for m in usable]
    total_w = sum(weights)
    weighted_mean = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
    frac_high = sum(1 for s in scores if s >= swap_threshold) / len(scores)
    trigger = (weighted_mean >= swap_threshold) or (frac_high >= 0.50)

    summary.update({
        "weighted_mean_dull": round(weighted_mean, 3),
        "frac_high_dull":     round(frac_high, 3),
        "median_blue_score":  round(statistics.median(m["blue_score"]   for m in usable), 3),
        "median_golden_score":round(statistics.median(m["golden_score"] for m in usable), 3),
        "trigger":            trigger,
    })
    return trigger, summary
