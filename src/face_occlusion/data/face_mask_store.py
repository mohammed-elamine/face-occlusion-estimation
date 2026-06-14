"""Dedicated face-mask store — precomputed MediaPipe face masks, one PNG per image.

Decoupled from the synthetic-occlusion cache so background augmentation (and any future
label-preserving augmentation) can fetch a face mask for **any** training image, not just
the synthetic ranking anchors. Masks mirror the image's relative path under ``root_dir``,
so a mask's path is derived deterministically from the image id — there is no manifest.

The ``load_mask`` signature matches what :class:`~face_occlusion.data.background_augment.
BackgroundAugment` expects (``Callable[[object], np.ndarray | None]``), so the store drops
straight into ``BackgroundAugment(mask_lookup=store.load_mask)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class FaceMaskStore:
    """Lookup of precomputed boolean face masks keyed by image id (relative path).

    Parameters
    ----------
    root_dir:
        Directory holding the masks. A mask for image id ``"database1/img.webp"`` lives at
        ``root_dir / "database1/img.png"`` (the relative path mirrored, extension swapped to
        ``.png``).
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def mask_path(self, sample_id: object) -> Path:
        """Deterministic mask path for an image id (relative path mirrored, ext -> .png)."""
        return self.root_dir / Path(str(sample_id)).with_suffix(".png")

    def has_mask(self, sample_id: object) -> bool:
        return self.mask_path(sample_id).exists()

    def load_mask(self, sample_id: object) -> np.ndarray | None:
        """Boolean face mask (``True`` on face) for ``sample_id``, or ``None`` if absent.

        Returns ``None`` for any id without a stored mask (e.g. a MediaPipe detection miss),
        so callers no-op safely — identical contract to ``SyntheticCache.load_mask``.
        """
        path = self.mask_path(sample_id)
        if not path.exists():
            return None
        with Image.open(path) as m:
            return np.asarray(m.convert("L")) > 127

    def meta(self) -> dict[str, Any]:
        """Provenance/coverage metadata written by the builder, or ``{}`` if absent."""
        meta_path = self.root_dir / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def __repr__(self) -> str:
        return f"FaceMaskStore(root_dir={str(self.root_dir)!r})"
