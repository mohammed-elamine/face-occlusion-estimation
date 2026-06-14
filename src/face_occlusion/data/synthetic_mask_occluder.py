"""Landmark-driven realistic face-mask occluder (MaskTheFace-style).

Warps a real mask template onto the face using MediaPipe landmarks, then hands it
to the shared compositor so it blends in. This is the first *realistic* occluder
(replacing flat geometric patches); hands and glasses will follow the same
template→fit→composite pattern.

Fitting uses the six MaskTheFace key points (``a``–``f``). The four corners
(``a``, ``c``, ``d``, ``f``) define a quadrilateral mapped onto face anchor points
via a perspective warp; the template's own alpha shape supplies the nose peak and
chin dip. The ``coverage_level`` knob raises the top edge from the mouth (mild)
toward the nose bridge (strong), which is how mask severity is controlled — the
generator's severity proxy + accept/reject then validate the band.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .synthetic_compositing import CompositingConfig, composite_occluder

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# assets/occluders/masks relative to the repo root (…/src/face_occlusion/data/).
_DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "occluders" / "masks"

# MediaPipe Face Mesh landmark indices (all within the standard face-oval set).
_CHIN = 152
_NOSE_BRIDGE = 168
_SUBNASAL = 2
_SIDE_LANDMARKS = (234, 454, 132, 361, 172, 397)  # cheek/jaw silhouette points
_MIN_LANDMARKS = max(_SIDE_LANDMARKS + (_CHIN, _NOSE_BRIDGE, _SUBNASAL)) + 1


@lru_cache(maxsize=4)
def load_mask_templates(asset_dir: str | None = None) -> dict[str, tuple[np.ndarray, dict]]:
    """Load ``{name: (rgba_array, anchor_points)}`` from a mask asset directory."""
    d = Path(asset_dir) if asset_dir else _DEFAULT_ASSET_DIR
    anchors = json.loads((d / "anchors.json").read_text())
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for name, info in anchors.items():
        rgba = np.asarray(Image.open(d / info["template"]).convert("RGBA"))
        out[name] = (rgba, info["points"])
    if not out:
        raise ValueError(f"No mask templates found in {d}")
    return out


def _jitter_quad(quad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small rotation/scale/translation of the destination quad for variety."""
    center = quad.mean(axis=0)
    angle = np.deg2rad(rng.uniform(-5.0, 5.0))
    scale = rng.uniform(0.95, 1.05)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32) * scale
    span = quad[:, 0].max() - quad[:, 0].min()
    shift = rng.uniform(-0.02, 0.02, size=2).astype(np.float32) * span
    return ((quad - center) @ rot.T + center + shift).astype(np.float32)


def fit_mask(
    image_size: tuple[int, int],
    landmarks: np.ndarray,
    coverage_level: float,
    *,
    template: tuple[np.ndarray, dict],
    rng: np.random.Generator,
    jitter: bool = True,
) -> Image.Image:
    """Warp a mask ``template`` onto the face; return an RGBA layer at image size.

    ``coverage_level`` in ``[0, 1]`` raises the mask's top edge from the base of
    the nose (mild) to the nose bridge (strong).
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv (cv2) is required for mask fitting")
    width, height = image_size
    rgba, anchors = template
    th, tw = rgba.shape[:2]

    def tpt(key: str) -> list[float]:
        x, y = anchors[key]
        return [float(min(max(x, 0), tw - 1)), float(min(max(y, 0), th - 1))]

    src = np.float32([tpt("a"), tpt("c"), tpt("d"), tpt("f")])

    lm = np.asarray(landmarks, dtype=np.float32)
    chin_y = float(lm[_CHIN, 1])
    top_low = float(lm[_SUBNASAL, 1])  # base of nose (mild)
    top_high = float(lm[_NOSE_BRIDGE, 1])  # nose bridge, higher up (strong)
    top_y = top_low + float(np.clip(coverage_level, 0.0, 1.0)) * (top_high - top_low)
    face_h = float(lm[:, 1].max() - lm[:, 1].min())
    bottom_y = chin_y + 0.05 * face_h

    side_x = lm[list(_SIDE_LANDMARKS), 0]
    left_x, right_x = float(side_x.min()), float(side_x.max())
    margin_x = 0.05 * (right_x - left_x)
    left_x -= margin_x
    right_x += margin_x

    dst = np.float32([[left_x, top_y], [right_x, top_y], [left_x, bottom_y], [right_x, bottom_y]])
    if jitter:
        dst = _jitter_quad(dst, rng)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rgba,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA")


def build_realistic_mask_sampler(
    *,
    asset_dir: str | None = None,
    template_names: tuple[str, ...] | None = None,
    compositing: CompositingConfig | None = None,
):
    """Build an occluder sampler ``(image, rng, scale, ctx) -> (image, mask, info)``.

    ``ctx`` must expose ``.landmarks`` (the dense MediaPipe pixel coordinates). The
    sampler maps the generator's ``scale`` to a ``coverage_level`` so the mild band
    yields a low mask (mouth+chin) and the strong band a high mask (covering the
    nose). Returns an empty occluder when landmarks are unavailable, so the
    generator's accept/reject simply retries or fails the view.
    """
    templates = load_mask_templates(asset_dir)
    if template_names:
        templates = {k: v for k, v in templates.items() if k in template_names}
        if not templates:
            raise ValueError(f"None of {template_names} are available mask templates")
    names = tuple(templates)
    comp_cfg = compositing or CompositingConfig()

    def sampler(image, rng, scale, ctx):
        landmarks = getattr(ctx, "landmarks", None)
        empty = np.zeros((image.height, image.width), dtype=bool)
        if landmarks is None or len(landmarks) < _MIN_LANDMARKS:
            return image, empty, {"mask_failed": "no_landmarks"}
        name = names[int(rng.integers(0, len(names)))]
        # Map scale (0.45 mild … 2.2 strong) → coverage_level in [0, 1], + jitter.
        coverage = float(np.clip((float(scale) - 0.45) / (2.2 - 0.45), 0.0, 1.0))
        coverage = float(np.clip(coverage + rng.uniform(-0.1, 0.1), 0.0, 1.0))
        occ_rgba = fit_mask(image.size, landmarks, coverage, template=templates[name], rng=rng)
        out, mask = composite_occluder(image, occ_rgba, rng, comp_cfg)
        return out, mask, {"mask_template": name, "coverage_level": coverage}

    return sampler
