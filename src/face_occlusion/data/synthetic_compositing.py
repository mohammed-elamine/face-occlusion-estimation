"""Realistic occluder compositing — the shared "no-seam" engine.

Pasting an RGBA occluder onto a face looks fake mostly because of the *seam*, not
the shape. This module blends a placed occluder into a host image so it belongs in
the photo, via a few cheap, toggleable steps:

* **feather** the alpha edge so there is no hard cut,
* **harmonize** the occluder luminance toward the local face lighting,
* add a soft **contact shadow** just beneath the occluder,
* add light **grain** over the occluder so it isn't suspiciously clean,
* optionally **seamless-clone** (Poisson) for hard cases.

It is deliberately occluder-agnostic: the mask fitter renders a mask into an RGBA
layer and hands it here; hands/glasses (later) will reuse the exact same engine.
The occluder colour is preserved by default (a white surgical mask stays white) —
seamless cloning, which would wash that out, is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

try:  # cv2 is a project dependency; guard so imports never hard-fail.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class CompositingConfig:
    """Knobs for :func:`composite_occluder` (sensible defaults for masks)."""

    feather_px: float = 2.0
    harmonize: bool = True
    harmonize_strength: float = 0.5  # 0 keeps occluder colour, 1 fully matches lighting
    # Skin-tone color transfer toward a reference region (for hands). Needs a
    # ``reference_mask``; off by default so masks keep their own colour.
    color_match: bool = False
    color_match_strength: float = 0.7
    shadow: bool = True
    shadow_strength: float = 0.30
    shadow_offset_px: int = 4
    shadow_blur_px: float = 6.0
    grain: bool = True
    grain_sigma: float = 3.0
    seamless: bool = False  # Poisson clone; off so mask colour is preserved


def _alpha_from(occluder_rgba: np.ndarray) -> np.ndarray:
    return occluder_rgba[..., 3].astype(np.float32) / 255.0


def composite_occluder(
    host_rgb: Image.Image,
    occluder_rgba: Image.Image,
    rng: np.random.Generator,
    cfg: CompositingConfig | None = None,
    reference_mask: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Blend an RGBA occluder onto ``host_rgb`` so it looks natural.

    Parameters
    ----------
    host_rgb:
        The face image (PIL, any mode; converted to RGB).
    occluder_rgba:
        The occluder rendered at the **same size** as the host, with alpha.
    rng:
        Seeded generator (only the grain step is stochastic).

    Returns
    -------
    ``(blended_rgb, coverage_mask)`` where ``coverage_mask`` is the boolean
    occluder footprint (alpha above a small threshold), suitable for the existing
    severity proxy.
    """
    cfg = cfg or CompositingConfig()
    host = np.asarray(host_rgb.convert("RGB"), dtype=np.float32)
    h, w = host.shape[:2]
    occ = occluder_rgba.convert("RGBA")
    if occ.size != (w, h):
        occ = occ.resize((w, h), Image.BILINEAR)
    occ = np.asarray(occ, dtype=np.float32)

    alpha = _alpha_from(occ)  # (h, w) in [0, 1]
    rgb = occ[..., :3].copy()
    coverage = alpha > 0.05  # footprint BEFORE feathering (for severity)

    # 0) Skin-tone color transfer: recolour the occluder toward a reference region
    #    of the host (e.g. the face skin) so a hand matches the face it covers.
    if cfg.color_match and reference_mask is not None and coverage.any():
        rgb = _color_transfer(rgb, coverage, host, reference_mask, cfg.color_match_strength)

    # 1) Feather the alpha edge so there is no hard cut.
    if cfg.feather_px > 0:
        a_img = Image.fromarray((alpha * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(cfg.feather_px)
        )
        alpha = np.asarray(a_img, dtype=np.float32) / 255.0

    # 2) Harmonize occluder luminance toward the local face lighting it covers.
    if cfg.harmonize and coverage.any():
        host_lum = host.mean(axis=2)
        occ_lum = rgb.mean(axis=2)
        target = float(host_lum[coverage].mean())
        src = float(occ_lum[coverage].mean()) + 1e-6
        gain = 1.0 + cfg.harmonize_strength * (target / src - 1.0)
        rgb = np.clip(rgb * gain, 0.0, 255.0)

    out = host.copy()

    # 3) Soft contact shadow: darken a crescent just beneath the occluder.
    if cfg.shadow and coverage.any():
        shifted = np.roll(alpha, cfg.shadow_offset_px, axis=0)
        sh_img = Image.fromarray((shifted * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(cfg.shadow_blur_px)
        )
        shadow = np.asarray(sh_img, dtype=np.float32) / 255.0
        shadow = np.clip(shadow - alpha, 0.0, 1.0)  # only where the occluder isn't
        out *= (1.0 - cfg.shadow_strength * shadow)[..., None]

    # 4) Composite the (harmonized, feathered) occluder over the shadowed host.
    a3 = alpha[..., None]
    if cfg.seamless and cv2 is not None and coverage.any():
        out = _seamless_clone(out, rgb, coverage)
        out = a3 * rgb + (1.0 - a3) * out  # keep a soft alpha edge on top
    else:
        out = a3 * rgb + (1.0 - a3) * out

    # 5) Light grain over the occluder so it isn't suspiciously clean.
    if cfg.grain and cfg.grain_sigma > 0 and coverage.any():
        noise = rng.normal(0.0, cfg.grain_sigma, size=out.shape).astype(np.float32)
        out = np.where(coverage[..., None], np.clip(out + noise, 0.0, 255.0), out)

    return Image.fromarray(out.astype(np.uint8), mode="RGB"), coverage


def _color_transfer(
    rgb: np.ndarray,
    coverage: np.ndarray,
    host: np.ndarray,
    reference_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Reinhard mean/std color transfer of the occluder toward a host region.

    Matches the per-channel mean and standard deviation of the occluder pixels
    (under ``coverage``) to those of the host pixels under ``reference_mask``,
    blended by ``strength``. Used to give a hand the face's skin tone.
    """
    ref = np.asarray(reference_mask, dtype=bool)
    if ref.shape != coverage.shape:
        ref_img = Image.fromarray(ref.astype(np.uint8) * 255).resize(
            (coverage.shape[1], coverage.shape[0]), Image.NEAREST
        )
        ref = np.asarray(ref_img) > 127
    if not ref.any():
        return rgb
    src = rgb[coverage]
    target = host[ref]
    s_mean, s_std = src.mean(axis=0), src.std(axis=0) + 1e-6
    t_mean, t_std = target.mean(axis=0), target.std(axis=0) + 1e-6
    transferred = (src - s_mean) / s_std * t_std + t_mean
    blended = (1.0 - strength) * src + strength * transferred
    out = rgb.copy()
    out[coverage] = np.clip(blended, 0.0, 255.0)
    return out


def _seamless_clone(host: np.ndarray, occ_rgb: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Poisson blend the occluder into the host (optional, off by default)."""
    mask = (coverage.astype(np.uint8)) * 255
    ys, xs = np.where(coverage)
    if ys.size == 0:
        return host
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    blended = cv2.seamlessClone(
        occ_rgb.astype(np.uint8), host.astype(np.uint8), mask, center, cv2.NORMAL_CLONE
    )
    return blended.astype(np.float32)
