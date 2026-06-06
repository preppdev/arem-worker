"""Sky replacement v9 — LOCKED BASELINE.

Canonical sky-replacement pipeline as of v9. Trees solid, architecture
sharp, conservative canopy-gap detection. Experiments to beat this
(v11 SegFormer, v12 HQ-SAM) live in separate files.

Pipeline:
  1. skyseg @ 320 → guided filter → coarse soft α
  2. Trimap (0.6% band)
  3. PyMatting closed-form matting → sub-pixel α
  4. Edge-aware sharpen: Sobel-modulated S-curve
  5. Lab ΔE 18 color-range expansion (upper 65%, gradient-blocked)
  6. Demote tiny floating sky islands
  7. Spill decontamination on α ∈ [0.05, 0.50]
  8. Fit sky plate, alpha-composite
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cv2  # type: ignore
import numpy as np  # type: ignore

SKYSEG_ONNX = "/home/jordan/sky_models/skyseg_u2net.onnx"
_SKYSEG = None


def _guided_filter(I, p, r, eps):
    k = (r * 2 + 1, r * 2 + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, k)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, k)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, k)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, cv2.CV_32F, k) * I + cv2.boxFilter(b, cv2.CV_32F, k)


def _load_skyseg():
    global _SKYSEG
    if _SKYSEG is None:
        import onnxruntime as ort  # type: ignore
        try:
            _SKYSEG = ort.InferenceSession(
                SKYSEG_ONNX, providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                        "CPUExecutionProvider"])
        except Exception:
            _SKYSEG = ort.InferenceSession(SKYSEG_ONNX, providers=["CPUExecutionProvider"])
    return _SKYSEG


def skyseg_alpha(image_bgr, guided_r=32, guided_eps=1e-4):
    s = _load_skyseg()
    h0, w0 = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, (320, 320), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / \
          np.array([0.229, 0.224, 0.225], np.float32)
    x = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    out = s.run(None, {"input.1": x})[0][0, 0]
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-6:
        return np.zeros((h0, w0), np.float32)
    out = (out - lo) / (hi - lo)
    alpha = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)
    guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = _guided_filter(guide, alpha.astype(np.float32), r=guided_r, eps=guided_eps)
    return np.clip(alpha, 0.0, 1.0)


def build_trimap(alpha, band_frac=0.006):
    h, w = alpha.shape
    band = max(3, int(band_frac * min(h, w)))
    high = (alpha > 0.85).astype(np.uint8) * 255
    low = (alpha < 0.15).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    tri = np.full_like(alpha, 128, np.uint8)
    tri[cv2.erode(high, k) > 0] = 255
    tri[cv2.erode(low, k) > 0] = 0
    return tri


def closed_form_matting(image_bgr, trimap_u8, max_dim=1600):
    from pymatting import estimate_alpha_cf  # type: ignore
    h0, w0 = image_bgr.shape[:2]
    long_side = max(h0, w0)
    if long_side > max_dim:
        scale = max_dim / long_side
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        img_s = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tri_s = cv2.resize(trimap_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_s, tri_s = image_bgr, trimap_u8
    rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    tri = tri_s.astype(np.float64) / 255.0
    alpha_s = estimate_alpha_cf(rgb, tri)
    if alpha_s.shape != (h0, w0):
        guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        alpha_up = cv2.resize(alpha_s.astype(np.float32), (w0, h0),
                              interpolation=cv2.INTER_LINEAR)
        alpha_s = _guided_filter(guide, alpha_up, r=4, eps=1e-6)
    return np.clip(alpha_s, 0.0, 1.0).astype(np.float32)


def source_gradient_map(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    grad = np.sqrt(gx * gx + gy * gy)
    p99 = float(np.percentile(grad, 99))
    g = np.clip(grad / max(p99, 1e-6), 0, 1)
    return cv2.GaussianBlur(g, (15, 15), 0)


def edge_aware_sharpen(image_bgr, alpha, gamma_soft=1.5, gamma_hard=4.0,
                       band=(0.05, 0.95)):
    grad = source_gradient_map(image_bgr)
    gamma_map = gamma_soft + (gamma_hard - gamma_soft) * grad
    a = alpha.copy()
    soft = (a > band[0]) & (a < band[1])
    if soft.any():
        a_s = a[soft]; g_s = gamma_map[soft]
        above = a_s > 0.5
        a_new = np.where(above,
                         1.0 - (1.0 - a_s) ** g_s,
                         a_s ** g_s)
        a[soft] = a_new
    return np.clip(a, 0, 1).astype(np.float32), grad


def expand_by_sky_color(image_bgr, alpha, grad, color_tol_lab=18.0,
                        grad_block=0.35, top_frac=0.65):
    h, w = alpha.shape
    sel = alpha > 0.9
    if int(sel.sum()) < 1000:
        return alpha
    sky_med_bgr = np.median(image_bgr[sel].astype(np.float32), axis=0)
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    med_lab = cv2.cvtColor(
        sky_med_bgr.astype(np.uint8)[None, None], cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)
    delta = np.linalg.norm(img_lab - med_lab[None, None, :], axis=2)
    upper = np.zeros_like(alpha, dtype=bool); upper[:int(h * top_frac), :] = True
    candidates = (
        (delta < color_tol_lab) & upper
        & (grad < grad_block) & (alpha < 0.4)
    )
    if not candidates.any():
        return alpha
    a = alpha.copy()
    t = np.clip(1.0 - (delta[candidates] / color_tol_lab), 0, 1).astype(np.float32)
    a[candidates] = np.maximum(a[candidates], 0.65 + 0.30 * t)
    return np.clip(a, 0, 1).astype(np.float32)


def demote_islands(alpha, min_frac=0.0001):
    h, w = alpha.shape
    hard_sky = (alpha > 0.6).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard_sky, connectivity=8)
    if n <= 1:
        return alpha
    min_px = int(min_frac * h * w)
    keep = np.zeros(n, bool); keep[0] = True
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[i] = True
    drop_mask = ~keep[labels]
    a = alpha.copy()
    a[drop_mask] = np.minimum(a[drop_mask], 0.3)
    return a


def estimate_sky_color(image_bgr, alpha):
    sel = alpha > 0.9
    if int(sel.sum()) < 1000:
        return np.array([235, 230, 215], np.float32)
    return np.median(image_bgr[sel].astype(np.float32), axis=0)


def decontaminate_spill(image_bgr, alpha, sky_color_bgr, strength=1.0,
                        alpha_lo=0.05, alpha_hi=0.50):
    out = image_bgr.astype(np.float32).copy()
    band = (alpha > alpha_lo) & (alpha < alpha_hi)
    if not band.any():
        return out.astype(np.uint8)
    a = alpha[band][:, None]
    I = out[band]
    F = (I - a * sky_color_bgr[None]) / np.clip(1.0 - a, 1e-3, 1.0)
    out[band] = np.clip(strength * F + (1 - strength) * I, 0, 255)
    return out.astype(np.uint8)


def fit_plate(plate_bgr, target_h, target_w, seed=None, flip=False,
              zoom_range=(1.10, 1.30), rotation_range_deg=(0.0, 0.0),
              vertical_extent=0.5):
    """Fit plate to target dimensions.

    `vertical_extent` (default 0.5): fraction of `target_h` that the plate
    actually occupies, anchored at the TOP of the canvas. The remaining
    bottom band is filled by edge-replicating the plate's last row —
    invisible at composite time because the alpha matte is zero there
    (the sky region of any real exterior shot doesn't extend to the
    bottom of the frame). Setting this to 0.5 makes the plate's natural
    horizon-gradient land at the source's actual horizon line (~middle
    of the frame) instead of being stretched all the way to the bottom.
    Pass 1.0 to disable.

    `flip`: when True, horizontally mirrors the plate before fitting.
    Driven by exterior-subtype labels (front vs rear of house), not by
    random per-frame coin flips — same perspective shots must get the
    same flip. The caller decides.

    `seed`: when None, center-crop with no zoom (deterministic). When an
    int, applies conservative random variation:
      1. Extra zoom factor in `zoom_range` (default 10%–30%)
      2. XY offset paired with the extra zoom so we never read out of
         plate bounds.
    Flip is NOT seed-driven — pass `flip=True` explicitly.
    """
    fit_h = max(1, int(round(target_h * vertical_extent)))
    fit_w = target_w

    ph, pw = plate_bgr.shape[:2]
    fit_aspect = fit_w / fit_h
    plate_aspect = pw / ph

    if seed is None:
        rnd = None
        extra_zoom = 1.0
        ox_frac = 0.0
        oy_frac = 0.0
        rot_deg = 0.0
    else:
        import os as _os
        import random as _r
        rnd = _r.Random(int(seed) & 0xFFFFFFFF)
        extra_zoom = rnd.uniform(zoom_range[0], zoom_range[1])
        ox_frac = rnd.uniform(-0.5, 0.5)
        oy_frac = rnd.uniform(-0.5, 0.5)
        _rot_lo = float(_os.environ.get("SKY_V9_PLATE_ROT_LO_DEG", str(rotation_range_deg[0])))
        _rot_hi = float(_os.environ.get("SKY_V9_PLATE_ROT_HI_DEG", str(rotation_range_deg[1])))
        rot_deg = rnd.uniform(_rot_lo, _rot_hi)

    if flip:
        plate_bgr = plate_bgr[:, ::-1].copy()

    # Step 0: rotate the plate around its center by `rot_deg`. DISABLED BY
    # DEFAULT (rotation_range_deg=(0,0)) — the ±25° rotation was the root
    # cause of the "abstract art" smears: on any plate that carries a
    # horizon/landscape band, rotating it tilts that band into a steep
    # diagonal and BORDER_REPLICATE drags it as parallel streaks across
    # the source's sky region. Per-frame variety already comes from the
    # zoom + XY pan below. Only re-enable via SKY_V9_PLATE_ROT_*_DEG if
    # you have proven the plate set is 100% sky-only, and keep it small.
    # Border mode is REPLICATE (not REFLECT) so a re-enabled rotation
    # extends edge pixels instead of mirroring plate content inward.
    if abs(rot_deg) > 0.05:
        ph0, pw0 = plate_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((pw0 / 2.0, ph0 / 2.0), rot_deg, 1.0)
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        new_w = int(round(ph0 * sin + pw0 * cos))
        new_h = int(round(ph0 * cos + pw0 * sin))
        M[0, 2] += new_w / 2.0 - pw0 / 2.0
        M[1, 2] += new_h / 2.0 - ph0 / 2.0
        plate_bgr = cv2.warpAffine(
            plate_bgr, M, (new_w, new_h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        ph, pw = plate_bgr.shape[:2]
        plate_aspect = pw / ph
        fit_aspect = fit_w / fit_h

    # Step 1: cover-fit to the plate's TARGET ROW BAND (fit_w × fit_h)
    if plate_aspect > fit_aspect:
        new_w = int(round(ph * fit_aspect))
        x0_cover = (pw - new_w) // 2
        cover = plate_bgr[:, x0_cover:x0_cover + new_w]
    else:
        new_h = int(round(pw / fit_aspect))
        y0_cover = (ph - new_h) // 2
        cover = plate_bgr[y0_cover:y0_cover + new_h, :]

    # Step 2: upscale to (fit_w × fit_h) × extra_zoom for pan headroom
    bigger_w = int(round(fit_w * extra_zoom))
    bigger_h = int(round(fit_h * extra_zoom))
    bigger = cv2.resize(cover, (bigger_w, bigger_h),
                        interpolation=cv2.INTER_LANCZOS4)

    # Step 3: pan within wiggle room
    wiggle_x = bigger_w - fit_w
    wiggle_y = bigger_h - fit_h
    cx = wiggle_x // 2 + int(round(ox_frac * wiggle_x))
    cy = wiggle_y // 2 + int(round(oy_frac * wiggle_y))
    cx = max(0, min(wiggle_x, cx))
    cy = max(0, min(wiggle_y, cy))
    plate_band = bigger[cy:cy + fit_h, cx:cx + fit_w]

    # Step 4: paste into the top of the target canvas. Bottom band gets
    # the plate's last row replicated — invisible at composite time
    # because alpha=0 there, but avoids leaving uninitialized memory.
    if fit_h >= target_h:
        return plate_band[:target_h]
    canvas = np.empty((target_h, target_w, plate_band.shape[2]), dtype=plate_band.dtype)
    canvas[:fit_h] = plate_band
    canvas[fit_h:] = plate_band[-1:].repeat(target_h - fit_h, axis=0)
    return canvas


def match_plate_luminance(plate_bgr, src_bgr, alpha, *,
                          min_alpha=0.85, min_shift=0.0, strength=1.0,
                          std_strength=0.5, ab_strength=0.0):
    """Match plate to source sky in Lab space.

    L channel: shift mean (full strength) and scale variance toward the
    source distribution at `std_strength` (default 0.5 — partial so a
    clean plate doesn't get stretched into noise). This is the dominant
    brightness fix.

    a/b channels: shift mean at `ab_strength` (default 0.0 — disabled to
    preserve the plate's blue character). Set via env if a particular
    shoot's sky has a tint that doesn't match the plate pack.

    Sky samples: alpha >= `min_alpha` (default 0.85 — looser than v1's
    0.95 to get a usable sample on tight-canopy frames where the
    high-confidence sky region is small).
    """
    import os as _os
    if _os.environ.get("SKY_V9_PLATE_LUMA_MATCH", "1").lower() not in ("1", "true", "yes"):
        return plate_bgr
    _str       = float(_os.environ.get("SKY_V9_PLATE_LUMA_STRENGTH",     str(strength)))
    _min_shift = float(_os.environ.get("SKY_V9_PLATE_LUMA_MIN_SHIFT",    str(min_shift)))
    _min_alpha = float(_os.environ.get("SKY_V9_PLATE_LUMA_MIN_ALPHA",    str(min_alpha)))
    _std_str   = float(_os.environ.get("SKY_V9_PLATE_LUMA_STD_STRENGTH", str(std_strength)))
    _ab_str    = float(_os.environ.get("SKY_V9_PLATE_LUMA_AB_STRENGTH",  str(ab_strength)))

    sky_mask = alpha >= _min_alpha
    if int(sky_mask.sum()) < 100:
        return plate_bgr
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    plate_lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_l = src_lab[..., 0][sky_mask]
    src_l_mean = float(src_l.mean()); src_l_std = float(src_l.std())
    plate_l = plate_lab[..., 0]
    plate_l_mean = float(plate_l.mean()); plate_l_std = float(plate_l.std() + 1e-6)

    delta = (src_l_mean - plate_l_mean) * _str
    if abs(delta) >= _min_shift:
        # Scale variance toward source, then shift mean. Scale factor is
        # blended toward 1.0 at (1 - std_strength) so std_strength=0
        # leaves variance unchanged.
        scale = 1.0 + _std_str * (src_l_std / plate_l_std - 1.0)
        new_l = (plate_l - plate_l_mean) * scale + plate_l_mean + delta
        plate_lab[..., 0] = np.clip(new_l, 0, 255)

    if _ab_str > 0.0:
        for ch in (1, 2):
            src_c_mean = float(src_lab[..., ch][sky_mask].mean())
            plate_c_mean = float(plate_lab[..., ch].mean())
            shift = (src_c_mean - plate_c_mean) * _ab_str
            plate_lab[..., ch] = np.clip(plate_lab[..., ch] + shift, 0, 255)

    return cv2.cvtColor(plate_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite(plate, src_decontam, alpha):
    a = alpha[..., None].astype(np.float32)
    out = plate.astype(np.float32) * a + src_decontam.astype(np.float32) * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def alpha_to_rgb(a):
    g = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def enforce_top_edge_sky(coarse, *, top_band_frac=0.02, dilate_px=16,
                          erode_before_cc_px=12):
    # Real outdoor sky must be reachable from the top edge through
    # connected sky pixels. Window reflections, glass tabletops, and
    # other indoor sky-impostors are always enclosed by structure.
    # We erode the binary mask before connectivity analysis so a thin
    # bridge of stray sky-tagged pixels through e.g. a porch ceiling
    # doesn't keep an enclosed window glued to the top-sky region.
    import os as _os
    if _os.environ.get("SKY_V9_TOP_EDGE", "1").lower() not in ("1","true","yes"):
        return coarse
    _band  = float(_os.environ.get("SKY_V9_TOP_EDGE_BAND_FRAC", str(top_band_frac)))
    _dil   = int(_os.environ.get("SKY_V9_TOP_EDGE_DILATE", str(dilate_px)))
    _erode = int(_os.environ.get("SKY_V9_TOP_EDGE_ERODE", str(erode_before_cc_px)))
    h, w = coarse.shape
    binary = (coarse > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return coarse
    # Erode so thin/weak bridges (porch ceiling stragglers, single-pixel
    # connectors) don't merge structurally separate regions.
    if _erode > 0:
        kern_e = np.ones((_erode, _erode), np.uint8)
        eroded = cv2.erode(binary, kern_e, iterations=1)
    else:
        eroded = binary
    if eroded.sum() == 0:
        return np.zeros_like(coarse)
    n_lab, labels = cv2.connectedComponents(eroded, connectivity=8)
    top_band = max(1, int(h * _band))
    top_labels = np.unique(labels[:top_band, :])
    keep_set = {int(l) for l in top_labels if int(l) != 0}
    if not keep_set:
        return np.zeros_like(coarse)
    keep = np.isin(labels, list(keep_set)).astype(np.uint8)
    # Re-grow: undo the erosion plus the soft-fringe margin
    regrow = _erode + max(0, _dil)
    if regrow > 0:
        kern_d = np.ones((regrow, regrow), np.uint8)
        keep = cv2.dilate(keep, kern_d, iterations=1)
    out = coarse.copy()
    out[keep == 0] = 0.0
    return out


def dark_suppress_alpha(alpha, src_bgr, *, v_low=60, v_high=140, strength=1.0):
    # Cap alpha by HSV V — dark pixels (V<v_low) get alpha ~0, bright
    # pixels (V>v_high) untouched. Physical: real sky is never near-black,
    # so deep shadows tagged as sky are matting / segmenter mistakes.
    import os as _os
    if _os.environ.get("SKY_V9_DARK_SUPPRESS", "1").lower() not in ("1","true","yes"):
        return alpha
    _vl  = float(_os.environ.get("SKY_V9_DARK_SUPPRESS_V_LOW",  str(v_low)))
    _vh  = float(_os.environ.get("SKY_V9_DARK_SUPPRESS_V_HIGH", str(v_high)))
    _str = float(_os.environ.get("SKY_V9_DARK_SUPPRESS_STRENGTH", str(strength)))
    hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2].astype(np.float32)
    cap = np.clip((V - _vl) / max(1.0, _vh - _vl), 0.0, 1.0)
    cap_eff = 1.0 - _str * (1.0 - cap)
    return np.clip(alpha * cap_eff, 0, 1).astype(np.float32)


def run_v9(src, plate, plate_seed=None, flip_plate=False,
           vertical_extent=0.5):
    """Single-call entry — returns (alpha, result).

    `plate_seed`: when provided, fit_plate randomizes extra zoom + XY
    offset deterministically per-frame so each shot in a shoot gets a
    unique sky viewpoint. When None, fit_plate is deterministic (center
    crop, no zoom).
    `flip_plate`: explicit horizontal-mirror flag. Caller drives this from
    exterior-subtype (front vs rear of house), NOT random per-frame.
    `vertical_extent`: fraction of source height the plate occupies,
    anchored at the top. 0.5 (default) aligns the plate's horizon
    gradient with a typical source horizon. Pass 1.0 to disable.
    """
    coarse = skyseg_alpha(src)
    # Guards: refuse to composite when the detected "sky" is too small
    # OR is in the wrong part of the frame. Both protect against
    # interior false positives (skylight slit, hallway light, bright
    # wall, kitchen window glare).
    import os as _os
    import numpy as _np
    _bin = coarse >= 0.5
    _n_sky = int(_bin.sum())
    _cov = _n_sky / max(1, _bin.size)
    _min_cov = float(_os.environ.get("SKY_V9_MIN_COVERAGE", "0.005"))
    if _cov < _min_cov:
        raise RuntimeError(
            f"insufficient sky coverage ({_cov:.4f} < {_min_cov:.4f}) — "
            f"likely a false positive on a small bright region"
        )
    # Spatial: real outdoor sky in an exterior shot sits in the upper
    # portion of the frame. If most of the "sky" mass is in the lower
    # half, we're almost certainly looking at a window/light/ceiling
    # fixture inside, not actual sky.
    _h_src = _bin.shape[0]
    _top_frac = float(_bin[:_h_src // 2, :].sum()) / max(1, _n_sky)
    _min_top  = float(_os.environ.get("SKY_V9_MIN_TOP_FRAC", "0.60"))
    if _top_frac < _min_top:
        raise RuntimeError(
            f"sky mass not in top half ({_top_frac:.2f} < {_min_top:.2f}) — "
            f"likely a window/skylight/fixture, not exterior sky"
        )
    # Drop coarse-sky components not connected to the top edge of the
    # image. Catches window-reflection false positives where the segmenter
    # tags blue-ish glass as sky despite it being enclosed by structure.
    coarse = enforce_top_edge_sky(coarse)
    if not (coarse > 0.5).any():
        raise RuntimeError(
            "no sky region reaches the top edge — likely window "
            "reflections / enclosed glass, refusing to composite"
        )
    trimap = build_trimap(coarse)
    try:
        alpha_raw = closed_form_matting(src, trimap)
    except Exception as _matte_err:
        # pymatting rejects the trimap ("did not contain foreground
        # values") or fails its incomplete-Cholesky preconditioner on
        # thin / low-contrast sky. We've already cleared the coverage +
        # top-edge guards above, so there IS real sky here -- rather than
        # drop the frame, fall back to a guided-filter matte off the
        # coarse (top-edge-enforced) segmentation. Softer edges, but a
        # usable QC composite instead of a hard skip.
        import sys as _sys
        print("[v9] closed-form matting failed (%s); falling back to "
              "guided coarse alpha" % _matte_err, file=_sys.stderr)
        _guide = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        alpha_raw = _guided_filter(_guide,
                                   np.clip(coarse, 0.0, 1.0).astype(np.float32),
                                   r=8, eps=1e-6)
        alpha_raw = np.clip(alpha_raw, 0.0, 1.0).astype(np.float32)
    alpha_sharp, grad = edge_aware_sharpen(src, alpha_raw)
    alpha_exp = expand_by_sky_color(src, alpha_sharp, grad)
    alpha = demote_islands(alpha_exp)
    # Re-enforce top-edge connectivity on the post-expansion alpha. The
    # expand step can paint window glass / glass-tabletop sky-blue because
    # it color-matches the sampled sky — drop any α>0.5 component that
    # doesn't reach the top band.
    alpha = enforce_top_edge_sky(alpha, dilate_px=0)
    alpha = dark_suppress_alpha(alpha, src)
    sky_bgr = estimate_sky_color(src, alpha)
    src_clean = decontaminate_spill(src, alpha, sky_bgr)
    # Match the plate's brightness to the source's existing sky so the
    # composite doesn't show a hard step where partial-α leaks the
    # original through. L-only — keeps the plate's blue character.
    plate_matched = match_plate_luminance(plate, src, alpha)
    plate_fit = fit_plate(plate_matched, src.shape[0], src.shape[1],
                          seed=plate_seed, flip=flip_plate,
                          vertical_extent=vertical_extent)
    result = composite(plate_fit, src_clean, alpha)
    return alpha, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sky-plate", required=True)
    ap.add_argument("--out", default="/tmp/skyv9b")
    ap.add_argument("--dropbox-dir",
                    default="dropbox:!American Real Estate Media/sky-v9-baseline/")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    src = cv2.imread(args.image); plate = cv2.imread(args.sky_plate)
    if src is None or plate is None:
        print("ERROR"); return 1
    print(f"[sky] src {src.shape[1]}x{src.shape[0]}", flush=True)
    cv2.imwrite(os.path.join(out, "src.jpg"), src, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    t = time.time()
    alpha, result = run_v9(src, plate)
    print(f"[sky] v9 pipeline in {time.time()-t:.1f}s, "
          f"α coverage {(alpha > 0.5).mean()*100:.1f}%", flush=True)
    cv2.imwrite(os.path.join(out, "alpha_refined.png"), alpha_to_rgb(alpha))
    cv2.imwrite(os.path.join(out, "result.jpg"), result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if args.dropbox_dir:
        tag = args.tag or os.path.splitext(os.path.basename(args.image))[0]
        dest = args.dropbox_dir.rstrip("/") + "/" + tag + "/"
        for fn in ("src.jpg", "alpha_refined.png", "result.jpg"):
            p = os.path.join(out, fn)
            if not os.path.exists(p):
                continue
            r = subprocess.run(["rclone", "copyto", p, dest + fn],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                print(f"[sky] rclone fail {fn}: {r.stderr.strip()[:200]}")
        print(f"[sky] uploaded to {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
