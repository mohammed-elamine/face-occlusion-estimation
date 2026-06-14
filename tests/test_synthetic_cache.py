"""Tests for the synthetic-cache schema, anchor selection, and manifest IO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_occlusion.data.synthetic_cache import (
    MANIFEST_FILENAME,
    coverage_table,
    load_cache_manifest,
    select_balanced_anchors,
    view_filenames,
)


def test_view_filenames_are_deterministic_and_distinct():
    f = view_filenames(7)
    assert f["clean"] == "views/000007_clean.webp"
    assert f["mild"] == "views/000007_mild.webp"
    assert f["strong"] == "views/000007_strong.webp"
    assert len(set(f.values())) == 3


def test_select_balanced_anchors_caps_large_cells_keeps_small():
    # bin A: 100 rows (should be capped to 5); bin B: 3 rows (kept fully).
    df = pd.DataFrame(
        {
            "occlusion_bin": ["A"] * 100 + ["B"] * 3,
            "gender": [0.0] * 100 + [1.0] * 3,
            "x": range(103),
        }
    )
    rng = np.random.default_rng(0)
    out = select_balanced_anchors(
        df, bin_col="occlusion_bin", gender_col="gender", max_per_bin_gender=5, rng=rng
    )
    counts = out.groupby(["occlusion_bin", "gender"]).size()
    assert counts[("A", 0.0)] == 5
    assert counts[("B", 1.0)] == 3


def test_select_balanced_anchors_rejects_nonpositive_cap():
    df = pd.DataFrame({"occlusion_bin": ["A"], "gender": [0.0]})
    with pytest.raises(ValueError):
        select_balanced_anchors(
            df,
            bin_col="occlusion_bin",
            gender_col="gender",
            max_per_bin_gender=0,
            rng=np.random.default_rng(0),
        )


def test_coverage_table_orders_by_bin():
    manifest = pd.DataFrame(
        {
            "id": list("abcd"),
            "occlusion_bin": ["0.60_1.00", "0.00_0.05", "0.00_0.05", "0.40_0.60"],
            "gender": [1.0, 0.0, 1.0, 0.0],
        }
    )
    table = coverage_table(manifest, bin_order=["0.00_0.05", "0.40_0.60", "0.60_1.00"])
    assert table.iloc[0]["occlusion_bin"] == "0.00_0.05"
    assert table.iloc[-1]["occlusion_bin"] == "0.60_1.00"
    assert int(table[table["occlusion_bin"] == "0.00_0.05"]["count"].sum()) == 2


def test_load_cache_manifest_roundtrip(tmp_path):
    manifest = pd.DataFrame(
        {
            "id": ["x.webp"],
            "occlusion_bin": ["0.40_0.60"],
            "gender": [1.0],
            "clean_path": ["views/000000_clean.webp"],
            "mild_path": ["views/000000_mild.webp"],
            "strong_path": ["views/000000_strong.webp"],
        }
    )
    manifest.to_csv(tmp_path / MANIFEST_FILENAME, index=False)
    loaded = load_cache_manifest(tmp_path)
    assert loaded.loc[0, "id"] == "x.webp"


def test_load_cache_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cache_manifest(tmp_path)


# ─── SyntheticCache + cache-backed dataset ────────────────────────────────────


def _build_tiny_cache(tmp_path, ids):
    from PIL import Image

    from face_occlusion.data.synthetic_cache import (
        MANIFEST_COLUMNS,
        SyntheticCache,
        view_filenames,
    )

    (tmp_path / "views").mkdir(parents=True, exist_ok=True)
    records = []
    for i, sample_id in enumerate(ids):
        names = view_filenames(i)
        colors = {"clean": (10, 10, 10), "mild": (120, 120, 120), "strong": (240, 240, 240)}
        for key, color in colors.items():
            Image.new("RGB", (16, 16), color=color).save(tmp_path / names[key])
        records.append(
            {
                "id": sample_id,
                "occlusion_bin": "0.40_0.60",
                "gender": 1.0,
                "clean_path": names["clean"],
                "mild_path": names["mild"],
                "strong_path": names["strong"],
                "mild_severity": 0.1,
                "strong_severity": 0.4,
                "mild_occluder_type": "a",
                "strong_occluder_type": "b",
            }
        )
    pd.DataFrame(records, columns=list(MANIFEST_COLUMNS)).to_csv(
        tmp_path / "manifest.csv", index=False
    )
    return SyntheticCache(tmp_path)


def test_synthetic_cache_lookup(tmp_path):
    cache = _build_tiny_cache(tmp_path, ["imgs/a.png"])
    assert "imgs/a.png" in cache
    assert "imgs/missing.png" not in cache
    entry = cache.lookup("imgs/a.png")
    assert entry["clean_path"].exists()
    assert entry["mild_severity"] == 0.1
    assert cache.lookup("imgs/missing.png") is None


def test_cache_backed_dataset_attaches_views(tmp_path):
    from PIL import Image

    from face_occlusion.data.dataset import FaceOcclusionDataset
    from face_occlusion.data.transforms import (
        build_eval_transform,
        build_synthetic_view_transform,
    )

    class _Cfg:
        class augmentation:
            resize = 16

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (16, 16), color=(50, 90, 130)).save(img_dir / name)
    df = pd.DataFrame(
        [
            {"filename": "imgs/a.png", "FaceOcclusion": 0.5, "gender": 1.0},
            {"filename": "imgs/b.png", "FaceOcclusion": 0.5, "gender": 1.0},
        ]
    )
    # Cache only contains "a.png"; "b.png" must fall back to valid=False.
    cache = _build_tiny_cache(tmp_path / "cache", ["imgs/a.png"])
    ds = FaceOcclusionDataset(
        df,
        image_root=tmp_path,
        transform=build_eval_transform(_Cfg()),
        mode="train",
        image_col="filename",
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
        synthetic_cache=cache,
        synthetic_view_transform=build_synthetic_view_transform(_Cfg()),
        synthetic_target_size=16,
    )
    cached = ds[0]
    assert bool(cached["synthetic_valid"]) is True
    assert cached["synthetic_clean_image"].shape == cached["image"].shape
    assert cached["synthetic_mild_severity"].item() == pytest.approx(0.1)

    missing = ds[1]
    assert bool(missing["synthetic_valid"]) is False
    assert missing["synthetic_failure_reason"] == "not_in_cache"
