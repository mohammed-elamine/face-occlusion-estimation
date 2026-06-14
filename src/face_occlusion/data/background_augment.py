"""Label-preserving background augmentation.

This is the *safe* use of a face mask: perturb only the pixels OUTSIDE the face
region, so the true face visibility -- and therefore the occlusion label -- is
unchanged. This is distinct from synthetic occluder pasting (which changes
visibility and is only valid for ranking supervision, never the regression label).

The goal is **background-invariance**: by randomizing the background strongly and
diversely, the model is pushed to read occlusion from the face itself rather than
from background shortcuts. Two design choices serve that:

* **Soft-alpha feathering.** The face mask is dilated a few pixels (to protect the
  hairline/jaw) and blurred into an alpha in ``[0, 1]``; the output is
  ``alpha * face + (1 - alpha) * variant``. The original mask pixels are forced to
  ``alpha = 1`` so the face is byte-preserved, while the boundary is feathered so
  no hard oval seam is left for the model to exploit as a localization shortcut.
* **Diverse variants.** Beyond a flat colour / brightness / noise, the background
  can be replaced by a heavy blur (bokeh), a spatially scrambled copy of its own
  background, or a smooth random texture -- much harder to "subtract" than a
  constant.

The face mask is precomputed (MediaPipe) and looked up here by training id. With
no mask for an id, augmentation is a no-op, which is always label-safe.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

_MODES = ("replace", "brightness", "noise", "blur", "shuffle", "texture")


def _resize_mask(face_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.asarray(face_mask)
    if mask.shape != (h, w):
        mask = (
            np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST))
            > 127
        )
    return mask.astype(bool)


def _soft_alpha(mask: np.ndarray, dilate_px: int, feather_px: float) -> np.ndarray:
    """Feathered alpha in [0, 1]: 1 on the (original) face, soft transition outside.

    The mask is dilated by ``dilate_px`` (protect the hairline/jaw), Gaussian-blurred
    by ``feather_px`` (kill the hard seam), then the original mask is forced back to 1
    so face pixels are preserved exactly.
    """
    m = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if dilate_px > 0:
        m = m.filter(ImageFilter.MaxFilter(2 * int(dilate_px) + 1))  # binary dilation
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(float(feather_px)))
    alpha = np.asarray(m, dtype=np.float32) / 255.0
    # Force exact preservation of the true face region (feather only outside it).
    alpha[mask] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def _background_variant(
    arr: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    brightness_range: tuple[float, float],
    noise_std: float,
) -> np.ndarray:
    """A full-image candidate; only its non-face region is blended in via the alpha."""
    h, w = arr.shape[:2]
    if mode == "replace":
        v = np.empty_like(arr)
        v[:] = rng.integers(0, 256, size=3).astype(np.float32)
        return v
    if mode == "brightness":
        return np.clip(arr * float(rng.uniform(*brightness_range)), 0, 255)
    if mode == "noise":
        return np.clip(arr + rng.normal(0.0, noise_std, size=arr.shape), 0, 255)
    if mode == "blur":
        radius = float(rng.uniform(4.0, 12.0))  # heavy bokeh
        blurred = Image.fromarray(arr.astype(np.uint8), mode="RGB").filter(
            ImageFilter.GaussianBlur(radius)
        )
        return np.asarray(blurred, dtype=np.float32)
    if mode == "shuffle":
        v = arr.copy()
        if rng.random() < 0.5:
            v = v[:, ::-1]
        if rng.random() < 0.5:
            v = v[::-1, :]
        if h == w:  # rot90 only when square (keeps shape)
            v = np.rot90(v, int(rng.integers(0, 4)))
        v = np.roll(
            v,
            shift=(int(rng.integers(-h // 3, h // 3 + 1)), int(rng.integers(-w // 3, w // 3 + 1))),
            axis=(0, 1),
        )
        return np.ascontiguousarray(v, dtype=np.float32)
    if mode == "texture":
        # Smooth low-frequency random field (bicubic-upsampled coarse noise).
        coarse = rng.uniform(0, 255, size=(8, 8, 3)).astype(np.uint8)
        field = Image.fromarray(coarse, mode="RGB").resize((w, h), Image.BICUBIC)
        return np.asarray(field, dtype=np.float32)
    raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


def apply_background_augmentation(
    image: Image.Image,
    face_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str = "replace",
    brightness_range: tuple[float, float] = (0.6, 1.4),
    noise_std: float = 25.0,
    dilate_px: int = 2,
    feather_px: float = 2.0,
) -> Image.Image:
    """Perturb only non-face pixels of ``image``; the true face region is preserved exactly.

    ``mode`` selects the background variant (``replace`` flat colour, ``brightness``
    luminance scale, ``noise`` Gaussian, ``blur`` bokeh, ``shuffle`` scrambled own
    background, ``texture`` smooth random field). The face mask is dilated +
    feathered into a soft alpha so the composite has no hard oval seam.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    mask = _resize_mask(face_mask, h, w)
    if mask.all():  # whole image is face -> nothing to augment
        return image.convert("RGB")

    alpha = _soft_alpha(mask, dilate_px, feather_px)[..., None]
    variant = _background_variant(arr, rng, mode, brightness_range, noise_std)
    out = alpha * arr + (1.0 - alpha) * variant
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


class BackgroundAugment:
    """Stochastic, per-id background augmentation driven by cached face masks.

    ``mask_lookup`` maps a training id to a boolean face mask (or ``None`` when no
    mask is cached). :meth:`__call__` applies an augmentation with probability ``p``
    (a random mode); :meth:`make_variant` always produces a fresh variant (a second,
    independently randomized background) for the background-invariance consistency
    view. A missing mask is a no-op. Deterministic given ``seed`` and the id index.
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
        dilate_px: int = 2,
        feather_px: float = 2.0,
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
        self.dilate_px = int(dilate_px)
        self.feather_px = float(feather_px)

    def _apply(self, image: Image.Image, mask: np.ndarray, rng: np.random.Generator) -> Image.Image:
        mode = str(rng.choice(self.modes))
        return apply_background_augmentation(
            image,
            mask,
            rng,
            mode=mode,
            brightness_range=self.brightness_range,
            noise_std=self.noise_std,
            dilate_px=self.dilate_px,
            feather_px=self.feather_px,
        )

    def __call__(self, image: Image.Image, sample_id: object, idx: int) -> Image.Image:
        mask = self.mask_lookup(sample_id)
        if mask is None:
            return image
        rng = np.random.default_rng([self.seed, int(idx)])
        if rng.random() >= self.p:
            return image
        return self._apply(image, mask, rng)

    def make_variant(self, image: Image.Image, sample_id: object, idx: int) -> Image.Image:
        """Always-on second background variant for the consistency view (or no-op if no mask).

        Uses an independent RNG stream so it differs from :meth:`__call__`.
        """
        mask = self.mask_lookup(sample_id)
        if mask is None:
            return image
        rng = np.random.default_rng([self.seed, int(idx), 1])
        return self._apply(image, mask, rng)
