"""Generator wiring for realistic occluders (mask + hand; no MediaPipe needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from face_occlusion.data.synthetic_occlusion import (
    SyntheticOcclusionGenerator,
    build_aligned_face_region_masks,
    build_generator_from_config,
)
from face_occlusion.utils.config import Config


class _StubProvider:
    """Avoids constructing the real MediaPipe provider in unit tests."""

    def extract(self, image):  # pragma: no cover - not called here
        raise NotImplementedError


def _hand_asset_dir(tmp_path):
    """A minimal hand-asset dir so the hand sampler can be built in tests."""
    rgba = np.zeros((80, 50, 4), dtype=np.uint8)
    rgba[10:60, 10:40] = (170, 130, 110, 255)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "hand_000.png")
    manifest = {
        "hand_000": {"template": "hand_000.png", "points": {"palm": [25, 30], "wrist": [25, 75]}}
    }
    (tmp_path / "anchors.json").write_text(json.dumps(manifest))
    return str(tmp_path)


def test_realistic_mask_generator_builds_sampler_and_is_single():
    gen = SyntheticOcclusionGenerator(
        occluder_types=["realistic_mask"], face_region_provider=_StubProvider(), seed=0
    )
    assert "realistic_mask" in gen._realistic_samplers
    assert gen._single_occluder is True


def test_geometric_generator_has_no_mask_sampler():
    gen = SyntheticOcclusionGenerator(
        occluder_types=["random_face_rectangle"], face_region_provider=_StubProvider(), seed=0
    )
    assert "realistic_mask" not in gen._realistic_samplers
    assert gen._single_occluder is False


def test_unknown_occluder_type_still_raises():
    with pytest.raises(ValueError, match="Unknown occluder type"):
        SyntheticOcclusionGenerator(
            occluder_types=["not_a_real_type"], face_region_provider=_StubProvider()
        )


def test_build_generator_from_config_parses_mask_block():
    cfg = Config(
        {
            "synthetic_occlusion": {
                "enabled": True,
                "occluder_types": ["realistic_mask"],
                "severity": {
                    "mild": {"min": 0.08, "max": 0.20},
                    "strong": {"min": 0.28, "max": 0.55},
                },
                "mask": {
                    "templates": ["surgical", "cloth"],
                    "compositing": {"feather_px": 1.0, "shadow_strength": 0.2},
                },
                "seed": 7,
            }
        }
    )
    gen = build_generator_from_config(cfg)
    assert gen.occluder_types == ("realistic_mask",)
    assert "realistic_mask" in gen._realistic_samplers
    assert gen.severity_bands == {"mild": (0.08, 0.20), "strong": (0.28, 0.55)}


def test_build_generator_disabled_returns_none():
    cfg = Config({"synthetic_occlusion": {"enabled": False}})
    assert build_generator_from_config(cfg) is None


# ─── hands ────────────────────────────────────────────────────────────────────


def test_realistic_hand_generator_builds_sampler(tmp_path):
    gen = SyntheticOcclusionGenerator(
        occluder_types=["realistic_hand"],
        hand_params={"asset_dir": _hand_asset_dir(tmp_path)},
        face_region_provider=_StubProvider(),
        seed=0,
    )
    assert "realistic_hand" in gen._realistic_samplers
    assert gen._single_occluder is True


def test_mixed_mask_hand_generator_is_single_with_both(tmp_path):
    gen = SyntheticOcclusionGenerator(
        occluder_types=["realistic_mask", "realistic_hand"],
        hand_params={"asset_dir": _hand_asset_dir(tmp_path)},
        face_region_provider=_StubProvider(),
        seed=0,
    )
    assert set(gen._realistic_samplers) == {"realistic_mask", "realistic_hand"}
    assert gen._single_occluder is True  # one occluder per view, never stacked


def test_forced_type_pins_the_occluder_for_a_whole_view():
    # With a wide-open band any occluder is accepted, so we can check that
    # forced_type pins every component to that single type (a triple is one type).
    gen = SyntheticOcclusionGenerator(
        occluder_types=["random_face_rectangle", "blurred_patch"],
        severity_bands={"mild": (0.01, 0.99), "strong": (0.01, 0.99)},
        face_region_provider=_StubProvider(),
        seed=0,
    )
    masks = build_aligned_face_region_masks(96)
    img = Image.new("RGB", (96, 96), (120, 120, 120))
    view, _ = gen._sample_view(
        img, masks, None, "strong", np.random.default_rng(0), forced_type="random_face_rectangle"
    )
    assert view is not None
    assert "random_face_rectangle" in view.metadata["occluder_type"]
    assert "blurred_patch" not in view.metadata["occluder_type"]


def test_build_generator_from_config_parses_hand_block(tmp_path):
    cfg = Config(
        {
            "synthetic_occlusion": {
                "enabled": True,
                "occluder_types": ["realistic_hand"],
                "hand": {
                    "asset_dir": _hand_asset_dir(tmp_path),
                    "compositing": {"color_match": True, "color_match_strength": 0.8},
                },
                "seed": 3,
            }
        }
    )
    gen = build_generator_from_config(cfg)
    assert gen.occluder_types == ("realistic_hand",)
    assert "realistic_hand" in gen._realistic_samplers
