"""Tests for path-derived metadata parsing."""

import pandas as pd

from face_occlusion.data.metadata import add_path_metadata


def test_add_path_metadata_parses_known_patterns():
    df = pd.DataFrame(
        {
            "filename": [
                "database1/img00000004.webp",
                "database2/database2/train/100.webp",
                "database2/database2/test/101.webp",
                "database3/database3/m.0109kg/0-FaceId-0_align.webp",
                "database3/database3/m.0109kg/1-FaceId-2_align.webp",
            ]
        }
    )

    out = add_path_metadata(df)

    assert out["database"].tolist() == [
        "database1",
        "database2",
        "database2",
        "database3",
        "database3",
    ]
    assert out["source_subfolder"].tolist() == [
        "database1",
        "database2/database2/train",
        "database2/database2/test",
        "database3/database3/m.0109kg",
        "database3/database3/m.0109kg",
    ]
    assert out["group_id"].tolist() == [
        "database1/img00000004.webp",
        "database2/database2/train/100.webp",
        "database2/database2/test/101.webp",
        "database3/m.0109kg",
        "database3/m.0109kg",
    ]
    assert out["face_id"].tolist() == [-1, -1, -1, 0, 2]
