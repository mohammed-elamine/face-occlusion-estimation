from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from face_occlusion.data.dataset import FaceOcclusionDataset
from face_occlusion.data.transforms import build_eval_transform


class _MiniCfg:
    class augmentation:
        resize = 32


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(3):
        p = img_dir / f"img_{i}.png"
        Image.new("RGB", (40, 40), color=(i * 50, 0, 0)).save(p)
        rows.append({"filename": f"imgs/img_{i}.png", "FaceOcclusion": 0.1 * i, "gender": 1.0})
    df = pd.DataFrame(rows)
    tf = build_eval_transform(_MiniCfg())
    ds = FaceOcclusionDataset(
        df,
        image_root=tmp_path,
        transform=tf,
        mode="train",
        image_col="filename",
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
    )
    return ds


def test_dataset_len(tiny_dataset):
    assert len(tiny_dataset) == 3


def test_dataset_keys_and_shape(tiny_dataset):
    item = tiny_dataset[0]
    assert set(item.keys()) >= {"image", "target", "gender", "image_id", "path"}
    assert item["image"].shape == (3, 32, 32)


def test_dataset_target_in_unit_range(tiny_dataset):
    for i in range(len(tiny_dataset)):
        t = float(tiny_dataset[i]["target"])
        assert 0.0 <= t <= 1.0


def test_dataset_normalizes_percent(tmp_path: Path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (40, 40)).save(img_dir / "a.png")
    df = pd.DataFrame([{"filename": "imgs/a.png", "FaceOcclusion": 50.0, "gender": 0.0}])
    ds = FaceOcclusionDataset(
        df,
        image_root=tmp_path,
        transform=build_eval_transform(_MiniCfg()),
        mode="train",
        image_col="filename",
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
        target_scale="auto",
    )
    assert abs(float(ds[0]["target"]) - 0.5) < 1e-6
