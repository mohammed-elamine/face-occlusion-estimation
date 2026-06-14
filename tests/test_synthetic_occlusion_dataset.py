"""Dataset integration tests for synthetic occlusion (Stage 3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from face_occlusion.data.dataset import FaceOcclusionDataset
from face_occlusion.data.synthetic_occlusion import (
    FaceRegionResult,
    SyntheticOcclusionGenerator,
    build_aligned_face_region_masks,
)
from face_occlusion.data.transforms import (
    build_eval_transform,
    build_synthetic_view_transform,
)

SIZE = 64


class _Cfg:
    class augmentation:
        resize = SIZE


class _StaticRegionProvider:
    def __init__(self, result: FaceRegionResult) -> None:
        self.result = result

    def extract(self, _image) -> FaceRegionResult:
        return self.result


def _valid_region_result(size: int = SIZE) -> FaceRegionResult:
    return FaceRegionResult(
        valid=True,
        masks=build_aligned_face_region_masks(size),
        metadata={"provider": "mock"},
    )


def _invalid_region_result(reason: str = "no_face_detected") -> FaceRegionResult:
    return FaceRegionResult(
        valid=False,
        masks={},
        failure_reason=reason,
        metadata={"provider": "mock"},
    )


def _make_dataset(
    tmp_path: Path,
    *,
    synthetic: bool,
    region_result: FaceRegionResult | None = None,
) -> FaceOcclusionDataset:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(4):
        p = img_dir / f"img_{i}.png"
        Image.new("RGB", (SIZE, SIZE), color=(40 + i * 30, 100, 120)).save(p)
        rows.append({"filename": f"imgs/img_{i}.png", "FaceOcclusion": 0.1, "gender": 1.0})
    df = pd.DataFrame(rows)
    tf = build_eval_transform(_Cfg())
    kwargs = dict(
        image_root=tmp_path,
        transform=tf,
        mode="train",
        image_col="filename",
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
    )
    if synthetic:
        kwargs["synthetic_generator"] = SyntheticOcclusionGenerator(
            seed=0,
            face_region_provider=_StaticRegionProvider(region_result or _valid_region_result()),
        )
        kwargs["synthetic_view_transform"] = build_synthetic_view_transform(_Cfg())
        kwargs["synthetic_target_size"] = SIZE
    return FaceOcclusionDataset(df, **kwargs)


def test_disabled_dataset_has_no_synthetic_keys(tmp_path: Path):
    ds = _make_dataset(tmp_path, synthetic=False)
    item = ds[0]
    assert not any(k.startswith("synthetic_") for k in item.keys())


def test_enabled_dataset_attaches_synthetic_tensors(tmp_path: Path):
    ds = _make_dataset(tmp_path, synthetic=True)
    item = ds[0]
    for key in (
        "synthetic_mild_image",
        "synthetic_strong_image",
        "synthetic_mild_severity",
        "synthetic_strong_severity",
        "synthetic_valid",
        "synthetic_failure_reason",
    ):
        assert key in item
    # Tensors share shape and dtype with the real image tensor.
    assert item["synthetic_mild_image"].shape == item["image"].shape
    assert item["synthetic_strong_image"].shape == item["image"].shape
    assert item["synthetic_mild_image"].dtype == item["image"].dtype
    assert item["synthetic_valid"].dtype == torch.bool
    assert item["synthetic_failure_reason"] == ""


def test_dataloader_can_collate_synthetic_batch(tmp_path: Path):
    ds = _make_dataset(tmp_path, synthetic=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 2
    assert batch["synthetic_mild_image"].shape == batch["image"].shape
    assert batch["synthetic_strong_image"].shape == batch["image"].shape
    assert batch["synthetic_mild_severity"].shape == (2,)
    assert batch["synthetic_valid"].shape == (2,)


def test_invalid_synthetic_generation_collates_with_valid_false(tmp_path: Path):
    ds = _make_dataset(
        tmp_path,
        synthetic=True,
        region_result=_invalid_region_result("no_face_detected"),
    )
    item = ds[0]
    assert item["synthetic_valid"].item() is False
    assert item["synthetic_failure_reason"] == "no_face_detected"

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["synthetic_mild_image"].shape == batch["image"].shape
    assert batch["synthetic_strong_image"].shape == batch["image"].shape
    assert batch["synthetic_valid"].shape == (2,)
    assert not batch["synthetic_valid"].any()


def test_val_mode_ignores_synthetic_generator(tmp_path: Path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(exist_ok=True)
    Image.new("RGB", (SIZE, SIZE), color=(50, 80, 100)).save(img_dir / "img_0.png")
    df = pd.DataFrame([{"filename": "imgs/img_0.png", "FaceOcclusion": 0.2, "gender": 0.0}])
    ds = FaceOcclusionDataset(
        df,
        image_root=tmp_path,
        transform=build_eval_transform(_Cfg()),
        mode="val",
        image_col="filename",
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
        synthetic_generator=SyntheticOcclusionGenerator(
            seed=0,
            face_region_provider=_StaticRegionProvider(_valid_region_result()),
        ),
        synthetic_view_transform=build_synthetic_view_transform(_Cfg()),
        synthetic_target_size=SIZE,
    )
    item = ds[0]
    assert not any(k.startswith("synthetic_") for k in item.keys())
