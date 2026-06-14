"""Load-bearing tests for the datamodule split/train.csv merge.

A stale split (ids that no longer match train.csv) used to silently drop rows,
quietly shrinking the dataset and making val/score incomparable across runs
(review R9). The merge must now be loud: raise by default, downgrade to a
warning only when ``split.allow_missing_rows`` is set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.utils.config import Config


def _write_csvs(tmp_path, *, train_ids, split_ids):
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "filename": [f"database3/database3/m.{i}/0-FaceId-0_align.webp" for i in train_ids],
            "FaceOcclusion": [0.1 for _ in train_ids],
            "gender": [i % 2 for i in train_ids],
        }
    ).to_csv(train_csv, index=False)

    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "filename": [f"database3/database3/m.{i}/0-FaceId-0_align.webp" for i in split_ids],
            "split": ["train" if k % 5 else "val" for k in range(len(split_ids))],
        }
    ).to_csv(split_path, index=False)
    return train_csv, split_path


def _cfg(tmp_path, train_csv, split_path, *, allow_missing=False) -> Config:
    return Config(
        {
            "project": {"seed": 42},
            "data": {
                "train_csv": str(train_csv),
                "test_csv": str(train_csv),
                "image_root": str(tmp_path),
                "image_col": "filename",
                "target_col": "FaceOcclusion",
                "gender_col": "gender",
                "id_col": "filename",
                "target_scale": "auto",
                "num_workers": 0,
            },
            "split": {
                "split_path": str(split_path),
                "occlusion_bins": [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0],
                "allow_missing_rows": allow_missing,
            },
            "augmentation": {
                "resize": 224,
                "horizontal_flip_p": 0.5,
                "color_jitter_p": 0.0,
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.05,
                "rotation_degrees": 3,
            },
        }
    )


def test_setup_raises_on_stale_split(tmp_path):
    # Split is missing id 9 that exists in train.csv -> mismatch -> raise.
    train_csv, split_path = _write_csvs(
        tmp_path, train_ids=list(range(10)), split_ids=list(range(9))
    )
    dm = FaceOcclusionDataModule(_cfg(tmp_path, train_csv, split_path))
    with pytest.raises(ValueError, match="Split/train.csv mismatch"):
        dm.setup("fit")


def test_setup_allows_missing_rows_when_opted_in(tmp_path, capsys):
    train_csv, split_path = _write_csvs(
        tmp_path, train_ids=list(range(10)), split_ids=list(range(9))
    )
    dm = FaceOcclusionDataModule(_cfg(tmp_path, train_csv, split_path, allow_missing=True))
    dm.setup("fit")  # must not raise
    assert "mismatch" in capsys.readouterr().out.lower()
    # The one unmatched row is dropped; the rest are kept.
    assert len(dm.train_df) > 0


def test_setup_no_warning_when_split_matches(tmp_path):
    train_csv, split_path = _write_csvs(
        tmp_path, train_ids=list(range(10)), split_ids=list(range(10))
    )
    dm = FaceOcclusionDataModule(_cfg(tmp_path, train_csv, split_path))
    dm.setup("fit")  # exact match -> no raise
    assert dm.train_ds is not None and dm.val_ds is not None
