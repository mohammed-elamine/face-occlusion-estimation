"""Label-preserving background augmentation.

This is the *safe* use of MediaPipe segmentation (use #1 in the contrastive
plan): perturb only the pixels OUTSIDE the face region, so the true face
visibility — and therefore the occlusion label — is unchanged. This is distinct
from synthetic occluder pasting (which changes visibility and is only valid for
ranking supervision, never for the regression label).

The face mask is expensive to compute (MediaPipe), so it is precomputed once
into the synthetic cache and looked up here by training id. With no mask for an
id, augmentation is a no-op, which is always label-safe.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

_MODES = ("replace", "brightness", "noise")


def apply_background_augmentation(
    image: Image.Image,
    face_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = "replace",
    brightness_range: tuple[float, float] = (0.6, 1.4),
    noise_std: float = 25.0,
) -> Image.Image:
    """Perturb only non-face pixels of ``image``; face pixels are left exact.

    Parameters
    ----------
    image:
        RGB PIL image.
    face_mask:
        Boolean array, ``True`` on face pixels. Resized to the image if needed
        (nearest-neighbour) so a cached mask can be reused across resizes.
    mode:
        ``"replace"`` fills the background with a random solid colour,
        ``"brightness"`` scales background luminance, ``"noise"`` adds Gaussian
        noise to the background.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    mask = np.asarray(face_mask)
    if mask.shape != (h, w):
        mask = (
            np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST))
            > 127
        )
    bg = ~mask.astype(bool)
    if not bg.any():
        return image.convert("RGB")

    out = arr.copy()
    if mode == "replace":
        color = rng.integers(0, 256, size=3).astype(np.float32)
        out[bg] = color
    elif mode == "brightness":
        factor = float(rng.uniform(*brightness_range))
        out[bg] = np.clip(arr[bg] * factor, 0, 255)
    else:  # noise
        noise = rng.normal(0.0, noise_std, size=arr[bg].shape)
        out[bg] = np.clip(arr[bg] + noise, 0, 255)

    # Face pixels are copied through untouched by construction (out[bg] only).
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


class BackgroundAugment:
    """Stochastic, per-id background augmentation driven by cached face masks.

    ``mask_lookup`` maps a training id to a boolean face mask (or ``None`` when
    no mask is cached). Augmentation is applied with probability ``p`` and a
    randomly chosen mode; a missing mask or a draw above ``p`` returns the image
    unchanged. Deterministic given ``seed`` and the id.
    """

    def __init__(
        self,
        mask_lookup: Callable[[object], np.ndarray | None],
        *,
        p: float = 0.5,
        modes: tuple[str, ...] = _MODES,
        seed: int = 42,
        brightness_range: tuple[float, float] = (0.6, 1.4),
        noise_std: float = 25.0,
    ) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        bad = [m for m in modes if m not in _MODES]
        if bad:
            raise ValueError(f"unknown background modes {bad}; valid: {_MODES}")
        self.mask_lookup = mask_lookup
        self.p = float(p)
        self.modes = tuple(modes)
        self.seed = int(seed)
        self.brightness_range = brightness_range
        self.noise_std = float(noise_std)

    def __call__(self, image: Image.Image, sample_id: object, idx: int) -> Image.Image:
        mask = self.mask_lookup(sample_id)
        if mask is None:
            return image
        rng = np.random.default_rng([self.seed, int(idx)])
        if rng.random() >= self.p:
            return image
        mode = str(rng.choice(self.modes))
        return apply_background_augmentation(
            image,
            mask,
            rng,
            mode=mode,
            brightness_range=self.brightness_range,
            noise_std=self.noise_std,
        )
