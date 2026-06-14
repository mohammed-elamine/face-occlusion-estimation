"""Tests for the hand fitter, placements, and the realistic-hand sampler."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from face_occlusion.data.synthetic_hand_occluder import (
    _placement_targets,
    _similarity_transform,
    build_realistic_hand_sampler,
    fit_hand,
)

SIZE = 128


def _fake_landmarks(size=SIZE):
    n = 468
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.3 * size, 0.7 * size, size=(n, 2)).astype(np.float32)
    lm[152] = [0.50 * size, 0.85 * size]  # chin
    lm[168] = [0.50 * size, 0.40 * size]  # nose bridge
    # widen the face extent so placements spread out
    lm[234] = [0.18 * size, 0.55 * size]
    lm[454] = [0.82 * size, 0.55 * size]
    return lm


def _fake_region_masks(size=SIZE):
    cheeks = np.zeros((size, size), dtype=bool)
    cheeks[int(0.5 * size) : int(0.7 * size), int(0.25 * size) : int(0.75 * size)] = True
    mouth = np.zeros((size, size), dtype=bool)
    mouth[int(0.62 * size) : int(0.70 * size), int(0.4 * size) : int(0.6 * size)] = True
    return {"cheeks": cheeks, "mouth": mouth, "face": cheeks | mouth}


def _hand_asset():
    """A synthetic opaque 'hand' (60x100) with palm high, wrist at the bottom."""
    rgba = np.zeros((100, 60, 4), dtype=np.uint8)
    rgba[10:70, 10:50, :3] = (170, 130, 110)  # palm
    rgba[10:70, 10:50, 3] = 255
    rgba[70:100, 25:35, :3] = (170, 130, 110)  # wrist to bottom edge
    rgba[70:100, 25:35, 3] = 255
    anchors = {"palm": [30.0, 40.0], "wrist": [30.0, 95.0]}
    return rgba, anchors


def _write_asset_dir(tmp_path):
    rgba, anchors = _hand_asset()
    Image.fromarray(rgba, "RGBA").save(tmp_path / "hand_000.png")
    (tmp_path / "anchors.json").write_text(
        json.dumps({"hand_000": {"template": "hand_000.png", "points": anchors}})
    )
    return tmp_path


# ─── transform ────────────────────────────────────────────────────────────────


def test_similarity_transform_maps_both_points():
    src = np.array([[0.0, 0.0], [0.0, 10.0]])
    dst = np.array([[5.0, 5.0], [5.0, 25.0]])  # scale x2, no rotation
    m = _similarity_transform(src, dst)
    p0 = m @ [0.0, 0.0, 1.0]
    p1 = m @ [0.0, 10.0, 1.0]
    assert np.allclose(p0, [5.0, 5.0], atol=1e-4)
    assert np.allclose(p1, [5.0, 25.0], atol=1e-4)


# ─── placements ───────────────────────────────────────────────────────────────


def test_placement_targets_are_anatomically_ordered():
    t = _placement_targets(_fake_landmarks(), _fake_region_masks(), (SIZE, SIZE))
    # forehead is above mouth is above chin (smaller y = higher).
    assert t["forehead"][0][1] < t["mouth"][0][1] < t["chin"][0][1]
    # cheek wrist-exits point sideways (off the face).
    assert t["left_cheek"][1][0] < 0 and t["right_cheek"][1][0] > 0
    # chin wrist exits downward.
    assert t["chin"][1][1] > 0


# ─── fit_hand ─────────────────────────────────────────────────────────────────


def test_fit_hand_places_wrist_toward_exit():
    rgba, anchors = _hand_asset()
    target = np.array([64.0, 80.0])
    exit_dir = np.array([0.0, 1.0])  # downward
    occ = fit_hand(
        (SIZE, SIZE),
        target=target,
        exit_dir=exit_dir,
        hand_length=30.0,
        asset=(rgba, anchors),
        rng=np.random.default_rng(0),
        flip=False,
    )
    alpha = np.asarray(occ)[..., 3]
    ys, xs = np.where(alpha > 10)
    assert ys.size > 0 and xs.min() >= 0 and xs.max() < SIZE
    # The hand's lowest covered pixels (wrist) sit below the palm target.
    assert ys.max() > target[1]


def test_fit_hand_larger_length_covers_more():
    rgba, anchors = _hand_asset()
    kw = dict(
        target=np.array([64.0, 64.0]),
        exit_dir=np.array([0.0, 1.0]),
        asset=(rgba, anchors),
        rng=np.random.default_rng(0),
        flip=False,
    )
    small = np.asarray(fit_hand((SIZE, SIZE), hand_length=20.0, **kw))[..., 3] > 10
    big = np.asarray(fit_hand((SIZE, SIZE), hand_length=50.0, **kw))[..., 3] > 10
    assert int(big.sum()) > int(small.sum())


# ─── sampler ──────────────────────────────────────────────────────────────────


def test_hand_sampler_composites(tmp_path):
    sampler = build_realistic_hand_sampler(asset_dir=str(_write_asset_dir(tmp_path)))
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8), "RGB"
    )
    ctx = SimpleNamespace(landmarks=_fake_landmarks(), region_masks=_fake_region_masks())
    out, mask, info = sampler(image, np.random.default_rng(1), scale=1.5, ctx=ctx)
    assert out.size == (SIZE, SIZE)
    assert mask.dtype == bool and mask.any()
    assert info["hand_asset"] == "hand_000" and info["placement"] in (
        "chin",
        "mouth",
        "left_cheek",
        "right_cheek",
        "forehead",
    )


def test_hand_sampler_no_landmarks_is_safe_noop(tmp_path):
    sampler = build_realistic_hand_sampler(asset_dir=str(_write_asset_dir(tmp_path)))
    image = Image.new("RGB", (SIZE, SIZE))
    out, mask, info = sampler(image, np.random.default_rng(0), 1.0, SimpleNamespace(landmarks=None))
    assert out is image and not mask.any()
    assert info.get("hand_failed") == "no_landmarks"
