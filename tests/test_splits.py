from __future__ import annotations

import pandas as pd

from face_occlusion.data.splits import make_stratified_split


def _split_frame() -> pd.DataFrame:
    rows = []
    for group_idx in range(12):
        gender = group_idx % 2
        target = 0.03 if group_idx < 6 else 0.25
        for image_idx in range(2):
            rows.append(
                {
                    "filename": (
                        f"database3/database3/m.{group_idx:04d}/{image_idx}-FaceId-0_align.webp"
                    ),
                    "FaceOcclusion": target,
                    "gender": gender,
                }
            )
    return pd.DataFrame(rows)


def test_row_split_keeps_metadata_columns() -> None:
    split = make_stratified_split(
        _split_frame(),
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
        bins=[0.0, 0.1, 1.0],
        val_size=0.25,
        seed=42,
        strategy="row_stratified",
        stratify_by=["gender", "occlusion_bin"],
    )

    assert set(split["split"]) == {"train", "val"}
    assert {"database", "source_subfolder", "group_id", "face_id", "occlusion_bin"}.issubset(
        split.columns
    )


def test_group_split_keeps_groups_out_of_both_splits() -> None:
    split = make_stratified_split(
        _split_frame(),
        target_col="FaceOcclusion",
        gender_col="gender",
        id_col="filename",
        bins=[0.0, 0.1, 1.0],
        val_size=0.25,
        seed=42,
        strategy="group_stratified",
        stratify_by=["gender", "occlusion_bin"],
        group_col="group_id",
    )

    train_groups = set(split.loc[split["split"] == "train", "group_id"])
    val_groups = set(split.loc[split["split"] == "val", "group_id"])

    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)
