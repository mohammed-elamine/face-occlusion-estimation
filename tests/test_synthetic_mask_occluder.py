"""Tests for the landmark mask-fitter and the realistic-mask sampler."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from face_occlusion.data.synthetic_mask_occluder import (
    build_realistic_mask_sampler,
    fit_mask,
    load_mask_templates,
)

SIZE = 128


def _fake_landmarks(size=SIZE):
    """A plausible frontal landmark cloud filling the central face area.

    Only the indices the fitter reads need sensible positions; the rest just need
    to exist and lie inside the face box.
    """
    n = 468
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.3 * size, 0.7 * size, size=(n, 2)).astype(np.float32)
    # Place the specific reference landmarks the fitter uses.
    lm[152] = [0.50 * size, 0.85 * size]  # chin (low)
    lm[168] = [0.50 * size, 0.40 * size]  # nose bridge (high)
    lm[2] = [0.50 * size, 0.60 * size]  # subnasal (mid)
    lm[234] = [0.20 * size, 0.55 * size]  # left cheek
    lm[454] = [0.80 * size, 0.55 * size]  # right cheek
    lm[132] = [0.22 * size, 0.62 * size]
    lm[361] = [0.78 * size, 0.62 * size]
    lm[172] = [0.28 * size, 0.78 * size]
    lm[397] = [0.72 * size, 0.78 * size]
    return lm


def _one_template():
    templates = load_mask_templates()
    name = next(iter(templates))
    return templates[name]


def test_templates_load_with_alpha():
    templates = load_mask_templates()
    assert len(templates) >= 4
    rgba, points = next(iter(templates.values()))
    assert rgba.shape[2] == 4  # RGBA
    assert {"a", "b", "c", "d", "e", "f"} <= set(points)


def test_fit_mask_stays_in_bounds_and_covers_lower_face():
    occ = fit_mask(
        (SIZE, SIZE),
        _fake_landmarks(),
        0.5,
        template=_one_template(),
        rng=np.random.default_rng(0),
        jitter=False,
    )
    assert occ.size == (SIZE, SIZE)
    alpha = np.asarray(occ)[..., 3]
    ys, xs = np.where(alpha > 10)
    assert ys.size > 0
    # Mask sits in the lower-central face, not the forehead.
    assert ys.mean() > 0.45 * SIZE
    assert xs.min() >= 0 and xs.max() < SIZE


def test_higher_coverage_raises_top_edge_and_covers_more():
    low = np.asarray(
        fit_mask(
            (SIZE, SIZE),
            _fake_landmarks(),
            0.0,
            template=_one_template(),
            rng=np.random.default_rng(0),
            jitter=False,
        )
    )[..., 3]
    high = np.asarray(
        fit_mask(
            (SIZE, SIZE),
            _fake_landmarks(),
            1.0,
            template=_one_template(),
            rng=np.random.default_rng(0),
            jitter=False,
        )
    )[..., 3]
    # Higher coverage reaches a smaller (higher) top row and covers more area.
    top_low = np.where(low > 10)[0].min()
    top_high = np.where(high > 10)[0].min()
    assert top_high < top_low
    assert int((high > 10).sum()) > int((low > 10).sum())


def test_sampler_composites_and_reports_template():
    sampler = build_realistic_mask_sampler()
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8), "RGB"
    )
    ctx = SimpleNamespace(landmarks=_fake_landmarks(), image_size=(SIZE, SIZE))
    out, mask, info = sampler(image, np.random.default_rng(1), scale=1.5, ctx=ctx)
    assert out.size == (SIZE, SIZE)
    assert mask.dtype == bool and mask.any()
    assert "mask_template" in info and 0.0 <= info["coverage_level"] <= 1.0


def test_sampler_no_landmarks_is_safe_noop():
    sampler = build_realistic_mask_sampler()
    image = Image.new("RGB", (SIZE, SIZE))
    ctx = SimpleNamespace(landmarks=None)
    out, mask, info = sampler(image, np.random.default_rng(0), scale=1.0, ctx=ctx)
    assert out is image and not mask.any()
    assert info.get("mask_failed") == "no_landmarks"
