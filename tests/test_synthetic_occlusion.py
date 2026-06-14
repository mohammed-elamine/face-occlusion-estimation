"""Tests for the synthetic occlusion generator (Stage 3)."""

from __future__ import annotations

import builtins
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from face_occlusion.data.synthetic_occlusion import (
    DEFAULT_OCCLUDER_TYPES,
    DEFAULT_REGION_WEIGHTS,
    OVERLAP_FLAG_KEYS,
    OVERLAP_METRIC_KEYS,
    REQUIRED_REGION_MASKS,
    FaceRegionResult,
    MediaPipeFaceRegionProvider,
    SyntheticOcclusionGenerator,
    build_aligned_face_region_masks,
    build_generator_from_config,
    compute_occluder_overlap_metrics,
    compute_severity,
)

SIZE = 224


@pytest.fixture
def fake_face_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = np.full((SIZE, SIZE, 3), 160, dtype=np.uint8)
    arr += rng.integers(-20, 21, size=arr.shape, dtype=np.int8).astype(np.uint8)
    return Image.fromarray(arr)


class _StaticRegionProvider:
    def __init__(self, result: FaceRegionResult) -> None:
        self.result = result
        self.calls = 0

    def extract(self, _image: Image.Image) -> FaceRegionResult:
        self.calls += 1
        return self.result


def _valid_region_result(size: int = SIZE) -> FaceRegionResult:
    return FaceRegionResult(
        valid=True,
        masks=build_aligned_face_region_masks(size),
        landmarks=np.zeros((468, 2), dtype=np.float32),
        metadata={"provider": "mock"},
    )


def _invalid_region_result(reason: str = "no_face_detected") -> FaceRegionResult:
    return FaceRegionResult(
        valid=False,
        masks={},
        landmarks=None,
        failure_reason=reason,
        metadata={"provider": "mock"},
    )


def _generator(
    *,
    seed: int = 0,
    result: FaceRegionResult | None = None,
    **kwargs,
) -> SyntheticOcclusionGenerator:
    return SyntheticOcclusionGenerator(
        seed=seed,
        face_region_provider=_StaticRegionProvider(result or _valid_region_result()),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Region provider contract
# ---------------------------------------------------------------------------


def test_valid_provider_result_has_required_boolean_masks():
    result = _valid_region_result()
    assert result.valid
    for name in REQUIRED_REGION_MASKS:
        assert name in result.masks
        assert result.masks[name].shape == (SIZE, SIZE), name
        assert result.masks[name].dtype == bool, name


def test_invalid_provider_result_carries_failure_reason():
    result = _invalid_region_result("no_face_detected")
    assert result.valid is False
    assert result.masks == {}
    assert result.failure_reason == "no_face_detected"


def test_mediapipe_missing_dependency_error_is_clear(monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "mediapipe":
            raise ModuleNotFoundError("No module named 'mediapipe'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match="MediaPipe is required"):
        MediaPipeFaceRegionProvider()


def test_legacy_region_masks_shape_and_face_area_positive():
    masks = build_aligned_face_region_masks(SIZE)
    for name in ("face", "eyes", "mouth", "nose", "cheeks", "forehead_chin", "background"):
        assert masks[name].shape == (SIZE, SIZE), name
        assert masks[name].dtype == bool, name
    assert masks["face"].sum() > 0
    # Eyes/mouth/nose are subsets of the face mask.
    for name in ("eyes", "mouth", "nose"):
        assert np.logical_and(masks[name], ~masks["face"]).sum() == 0, name
        assert np.logical_and(masks[name], masks["face"]).sum() > 0, name
    # Face and background are complementary.
    assert (masks["face"] | masks["background"]).all()
    assert not (masks["face"] & masks["background"]).any()


def test_background_severity_weight_is_zero():
    assert DEFAULT_REGION_WEIGHTS["background"] == 0.0


def test_compute_severity_zero_for_empty_mask():
    masks = build_aligned_face_region_masks(SIZE)
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    assert compute_severity(empty, masks, DEFAULT_REGION_WEIGHTS) == 0.0


def test_compute_severity_increases_with_coverage():
    masks = build_aligned_face_region_masks(SIZE)
    eyes_only = masks["eyes"]
    eyes_and_mouth = masks["eyes"] | masks["mouth"]
    s1 = compute_severity(eyes_only, masks, DEFAULT_REGION_WEIGHTS)
    s2 = compute_severity(eyes_and_mouth, masks, DEFAULT_REGION_WEIGHTS)
    assert 0.0 < s1 < s2 <= 1.0


def _tiny_region_masks() -> dict[str, np.ndarray]:
    face = np.zeros((4, 4), dtype=bool)
    face[:, :2] = True
    eyes = np.zeros((4, 4), dtype=bool)
    eyes[0, :2] = True
    mouth = np.zeros((4, 4), dtype=bool)
    mouth[1, :2] = True
    nose = np.zeros((4, 4), dtype=bool)
    nose[2, :2] = True
    lower_face = np.zeros((4, 4), dtype=bool)
    lower_face[2:, :2] = True
    return {
        "face": face,
        "left_eye": eyes[:, :],
        "right_eye": np.zeros((4, 4), dtype=bool),
        "eyes": eyes,
        "mouth": mouth,
        "nose": nose,
        "lower_face": lower_face,
        "cheeks": face & ~(eyes | mouth | nose | lower_face),
        "forehead_chin": np.zeros((4, 4), dtype=bool),
        "background": ~face,
    }


def test_compute_occluder_overlap_metrics_simple_masks():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True  # eye
    mask[1, 0] = True  # mouth
    mask[2, 3] = True  # background
    mask[3, 3] = True  # background
    metrics = compute_occluder_overlap_metrics(mask, _tiny_region_masks())
    assert metrics["face_overlap_ratio"] == pytest.approx(0.5)
    assert metrics["background_overlap_ratio"] == pytest.approx(0.5)
    assert metrics["important_region_overlap"] == pytest.approx(0.5)
    assert metrics["eye_overlap_ratio"] == pytest.approx(0.5)
    assert metrics["mouth_overlap_ratio"] == pytest.approx(0.5)
    assert metrics["nose_overlap_ratio"] == pytest.approx(0.0)
    assert metrics["lower_face_overlap_ratio"] == pytest.approx(0.0)
    assert metrics["occluder_area_ratio"] == pytest.approx(0.25)
    assert metrics["occluder_face_area_ratio"] == pytest.approx(0.25)


def test_compute_occluder_overlap_metrics_empty_mask_no_nans():
    metrics = compute_occluder_overlap_metrics(
        np.zeros((4, 4), dtype=bool),
        _tiny_region_masks(),
    )
    assert set(metrics) == set(OVERLAP_METRIC_KEYS) - {"weighted_severity"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(v == 0.0 for v in metrics.values())


def test_no_geometry_fallback_when_provider_fails(monkeypatch, fake_face_image):
    def _raise_if_called(_size: int):
        raise AssertionError("geometry fallback should not be called")

    monkeypatch.setattr(
        "face_occlusion.data.synthetic_occlusion.build_aligned_face_region_masks",
        _raise_if_called,
    )
    gen = _generator(result=_invalid_region_result("no_face_detected"))
    pair = gen.generate_pair(fake_face_image, rng=np.random.default_rng(0))
    assert pair.valid is False
    assert pair.region_masks == {}
    assert pair.metadata["failure_reason"] == "no_face_detected"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def test_generate_pair_returns_valid_pair(fake_face_image):
    gen = _generator(seed=0)
    pair = gen.generate_pair(fake_face_image, rng=np.random.default_rng(0))
    assert pair.valid
    assert pair.mild is not None and pair.strong is not None
    assert 0.05 <= pair.mild.severity <= 0.15
    assert 0.35 <= pair.strong.severity <= 0.60
    assert pair.mild.severity < pair.strong.severity
    for view in (pair.mild, pair.strong):
        for key in OVERLAP_METRIC_KEYS:
            assert key in view.metadata
            assert np.isfinite(view.metadata[key])
        for key in OVERLAP_FLAG_KEYS:
            assert key in view.metadata
            assert isinstance(view.metadata[key], bool)


def test_output_size_matches_input(fake_face_image):
    gen = _generator(seed=1)
    pair = gen.generate_pair(fake_face_image, rng=np.random.default_rng(1))
    assert pair.mild.image.size == fake_face_image.size
    assert pair.strong.image.size == fake_face_image.size
    assert pair.mild.occluder_mask.shape == (SIZE, SIZE)
    assert pair.strong.occluder_mask.shape == (SIZE, SIZE)


def test_severities_are_finite_and_in_unit_interval(fake_face_image):
    gen = _generator(seed=2)
    for s in range(10):
        pair = gen.generate_pair(fake_face_image, rng=np.random.default_rng(s))
        for view in (pair.mild, pair.strong):
            assert view is not None
            assert math.isfinite(view.severity)
            assert 0.0 <= view.severity <= 1.0


def test_same_seed_is_reproducible(fake_face_image):
    gen = _generator(seed=0)
    p1 = gen.generate_pair(fake_face_image, rng=np.random.default_rng(42))
    p2 = gen.generate_pair(fake_face_image, rng=np.random.default_rng(42))
    assert np.array_equal(np.asarray(p1.mild.image), np.asarray(p2.mild.image))
    assert np.array_equal(np.asarray(p1.strong.image), np.asarray(p2.strong.image))
    assert p1.mild.severity == p2.mild.severity
    assert p1.strong.severity == p2.strong.severity


def test_different_seeds_usually_differ(fake_face_image):
    gen = _generator(seed=0)
    p1 = gen.generate_pair(fake_face_image, rng=np.random.default_rng(0))
    p2 = gen.generate_pair(fake_face_image, rng=np.random.default_rng(1))
    assert not np.array_equal(np.asarray(p1.mild.image), np.asarray(p2.mild.image))


def test_invalid_severity_band_raises():
    with pytest.raises(ValueError):
        _generator(severity_bands={"mild": (0.2, 0.1)})
    with pytest.raises(ValueError):
        _generator(severity_bands={"mild": (-0.1, 0.1)})


def test_unknown_occluder_type_raises():
    with pytest.raises(ValueError):
        _generator(occluder_types=["nonexistent"])


def test_non_square_input_raises(fake_face_image):
    gen = _generator(seed=0)
    rect = fake_face_image.resize((128, 256))
    with pytest.raises(ValueError):
        gen.generate_pair(rect, rng=np.random.default_rng(0))


def test_unsatisfiable_band_returns_invalid_pair(fake_face_image):
    # 0.95–0.99 is unreachable for a single geometric layout under default
    # weights; generator must report ``valid=False`` rather than crash.
    gen = _generator(
        severity_bands={"mild": (0.05, 0.15), "strong": (0.95, 0.99)},
        max_attempts=8,
        seed=0,
    )
    pair = gen.generate_pair(fake_face_image, rng=np.random.default_rng(0))
    assert pair.valid is False
    assert pair.strong is None
    assert pair.metadata["strong_attempts"] == 8
    assert pair.metadata["failure_reason"] == "generation_failed"


def test_invalid_provider_returns_invalid_pair(fake_face_image):
    gen = _generator(result=_invalid_region_result("invalid_face_mask"))
    pair = gen.generate_pair(fake_face_image, rng=np.random.default_rng(0))
    assert pair.valid is False
    assert pair.mild is None and pair.strong is None
    assert pair.metadata["mediapipe_valid"] is False
    assert pair.metadata["failure_reason"] == "invalid_face_mask"


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def _cfg(**overrides):
    base = {
        "enabled": True,
        "region_provider": "mediapipe",
        "severity": {"mild": {"min": 0.05, "max": 0.15}, "strong": {"min": 0.30, "max": 0.55}},
        "region_weights": dict(DEFAULT_REGION_WEIGHTS),
        "occluder_types": list(DEFAULT_OCCLUDER_TYPES),
        "max_attempts": 50,
        "seed": 0,
    }
    base.update(overrides)
    return SimpleNamespace(
        get=lambda key, default=None: {"synthetic_occlusion": base}.get(key, default)
    )


def test_build_generator_from_config_disabled_returns_none():
    cfg = SimpleNamespace(
        get=lambda k, d=None: {"synthetic_occlusion": {"enabled": False}}.get(k, d)
    )
    assert build_generator_from_config(cfg) is None


def test_aligned_geometry_runtime_provider_raises_clear_error():
    with pytest.raises(ValueError, match="MediaPipe-only"):
        SyntheticOcclusionGenerator(
            region_provider="aligned_geometry",
            face_region_provider=_StaticRegionProvider(_valid_region_result()),
            seed=0,
        )


def test_build_generator_from_config_aligned_geometry_raises_clear_error():
    with pytest.raises(ValueError, match="MediaPipe-only"):
        build_generator_from_config(_cfg(region_provider="aligned_geometry"))
