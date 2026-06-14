"""Tests for label-preserving background augmentation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from face_occlusion.data.background_augment import (
    BackgroundAugment,
    apply_background_augmentation,
)


def _image_and_mask(size=16):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    mask = np.zeros((size, size), dtype=bool)
    mask[4:12, 4:12] = True  # central face region
    return image, mask, arr


@pytest.mark.parametrize("mode", ["replace", "brightness", "noise"])
def test_face_pixels_are_never_modified(mode):
    image, mask, arr = _image_and_mask()
    out = apply_background_augmentation(image, mask, np.random.default_rng(1), mode=mode)
    out_arr = np.asarray(out)
    # Face pixels (mask True) must be byte-identical -> label is preserved.
    assert np.array_equal(out_arr[mask], arr[mask])


@pytest.mark.parametrize("mode", ["replace", "brightness", "noise"])
def test_background_pixels_change(mode):
    image, mask, arr = _image_and_mask()
    out = np.asarray(
        apply_background_augmentation(image, mask, np.random.default_rng(2), mode=mode)
    )
    bg = ~mask
    # At least some background pixels differ (the augmentation did something).
    assert not np.array_equal(out[bg], arr[bg])


def test_mask_is_resized_to_image():
    image, _, arr = _image_and_mask(size=16)
    small_mask = np.zeros((8, 8), dtype=bool)
    small_mask[2:6, 2:6] = True  # maps to central region after nearest-resize
    out = apply_background_augmentation(image, small_mask, np.random.default_rng(3), mode="replace")
    assert np.asarray(out).shape == arr.shape


def test_invalid_mode_raises():
    image, mask, _ = _image_and_mask()
    with pytest.raises(ValueError):
        apply_background_augmentation(image, mask, np.random.default_rng(0), mode="nope")


def test_background_augment_noop_without_mask():
    image, _, arr = _image_and_mask()
    aug = BackgroundAugment(mask_lookup=lambda _id: None, p=1.0, seed=0)
    out = aug(image, "missing", 0)
    assert np.array_equal(np.asarray(out), arr)


def test_background_augment_applies_with_mask_and_is_deterministic():
    image, mask, _ = _image_and_mask()
    aug = BackgroundAugment(mask_lookup=lambda _id: mask, p=1.0, seed=7)
    a = np.asarray(aug(image, "x", 3))
    b = np.asarray(aug(image, "x", 3))
    assert np.array_equal(a, b)  # same id+idx+seed -> identical
    assert np.array_equal(a[mask], np.asarray(image)[mask])  # face preserved


def test_background_augment_p_zero_is_noop():
    image, mask, arr = _image_and_mask()
    aug = BackgroundAugment(mask_lookup=lambda _id: mask, p=0.0, seed=0)
    out = aug(image, "x", 0)
    assert np.array_equal(np.asarray(out), arr)


def test_background_augment_rejects_bad_params():
    with pytest.raises(ValueError):
        BackgroundAugment(mask_lookup=lambda _id: None, p=2.0)
    with pytest.raises(ValueError):
        BackgroundAugment(mask_lookup=lambda _id: None, modes=("bogus",))
