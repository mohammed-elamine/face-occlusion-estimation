"""Tests for the realistic occluder compositor."""

from __future__ import annotations

import numpy as np
from PIL import Image

from face_occlusion.data.synthetic_compositing import (
    CompositingConfig,
    composite_occluder,
)

SIZE = 64


def _host(color=(120, 110, 100)):
    return Image.new("RGB", (SIZE, SIZE), color=color)


def _occluder_rgba(box=(24, 24, 40, 40), color=(240, 240, 245)):
    """An opaque rectangular occluder centred in the image."""
    arr = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1, :3] = color
    arr[y0:y1, x0:x1, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def test_far_pixels_are_unchanged():
    host = _host()
    occ = _occluder_rgba()
    out, coverage = composite_occluder(host, occ, np.random.default_rng(0))
    out_arr = np.asarray(out)
    host_arr = np.asarray(host)
    # A corner far from the occluder (and its feather/shadow zone) is untouched.
    assert np.array_equal(out_arr[:8, :8], host_arr[:8, :8])


def test_coverage_mask_matches_occluder_footprint():
    occ = _occluder_rgba(box=(24, 24, 40, 40))
    _, coverage = composite_occluder(_host(), occ, np.random.default_rng(0))
    assert coverage[30, 30]  # inside the occluder
    assert not coverage[2, 2]  # far outside
    # Footprint area ~ the 16x16 box.
    assert 200 < int(coverage.sum()) < 320


def test_occluder_region_changes():
    host = _host(color=(30, 30, 30))  # dark face
    occ = _occluder_rgba(color=(245, 245, 245))  # white mask
    out, _ = composite_occluder(host, occ, np.random.default_rng(1))
    out_arr = np.asarray(out).astype(int)
    # The covered centre is clearly lighter than the dark host.
    assert out_arr[32, 32].mean() > 120


def test_harmonize_darkens_bright_occluder_on_dark_face():
    host = _host(color=(20, 20, 20))
    occ = _occluder_rgba(color=(255, 255, 255))
    plain = composite_occluder(
        host,
        occ,
        np.random.default_rng(0),
        CompositingConfig(harmonize=False, shadow=False, grain=False, feather_px=0),
    )[0]
    harmonized = composite_occluder(
        host,
        occ,
        np.random.default_rng(0),
        CompositingConfig(
            harmonize=True, harmonize_strength=0.8, shadow=False, grain=False, feather_px=0
        ),
    )[0]
    c_plain = np.asarray(plain)[32, 32].mean()
    c_harm = np.asarray(harmonized)[32, 32].mean()
    assert c_harm < c_plain  # harmonization pulls the white mask toward dark lighting


def test_feathering_softens_the_edge():
    occ = _occluder_rgba()
    hard = composite_occluder(
        _host(),
        occ,
        np.random.default_rng(0),
        CompositingConfig(feather_px=0, shadow=False, grain=False, harmonize=False),
    )[0]
    soft = composite_occluder(
        _host(),
        occ,
        np.random.default_rng(0),
        CompositingConfig(feather_px=2.0, shadow=False, grain=False, harmonize=False),
    )[0]
    # The feathered version has more distinct intensity levels along an edge row
    # (a hard cut has ~2 levels; a feather introduces intermediate ones).
    hard_levels = len(np.unique(np.asarray(hard)[24].mean(axis=1).round()))
    soft_levels = len(np.unique(np.asarray(soft)[24].mean(axis=1).round()))
    assert soft_levels > hard_levels


def test_deterministic_given_seed():
    host, occ = _host(), _occluder_rgba()
    a = composite_occluder(host, occ, np.random.default_rng(7))[0]
    b = composite_occluder(host, occ, np.random.default_rng(7))[0]
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_color_match_shifts_occluder_toward_reference():
    # Host: a dark-skin-toned face; occluder: a pale "hand". With color_match the
    # occluder should move toward the host's reference (skin) colour.
    host = _host(color=(80, 50, 40))  # warm dark skin
    occ = _occluder_rgba(color=(230, 220, 210))  # pale hand
    ref = np.zeros((SIZE, SIZE), dtype=bool)
    ref[:16, :16] = True  # a patch of "skin" to match toward
    cfg = CompositingConfig(
        color_match=True,
        color_match_strength=1.0,
        harmonize=False,
        shadow=False,
        grain=False,
        feather_px=0,
    )
    out = np.asarray(composite_occluder(host, occ, np.random.default_rng(0), cfg, ref)[0])
    center = out[32, 32].astype(int)
    # The covered pixel is pulled toward the dark/warm skin (much darker than 230).
    assert center.mean() < 160
    assert center[0] >= center[2]  # warmer (R >= B), like the skin reference


def test_color_match_noop_without_reference():
    host = _host(color=(80, 50, 40))
    occ = _occluder_rgba(color=(230, 220, 210))
    cfg = CompositingConfig(
        color_match=True, harmonize=False, shadow=False, grain=False, feather_px=0
    )
    with_ref = np.asarray(
        composite_occluder(host, occ, np.random.default_rng(0), cfg, np.ones((SIZE, SIZE), bool))[0]
    )
    no_ref = np.asarray(composite_occluder(host, occ, np.random.default_rng(0), cfg, None)[0])
    # No reference => color_match is a no-op, so the occluder keeps its pale colour.
    assert no_ref[32, 32].mean() > 200
    assert with_ref[32, 32].mean() < no_ref[32, 32].mean()
