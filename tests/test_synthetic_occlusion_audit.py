"""Tests for synthetic occlusion audit diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
from scripts.analysis.generate_synthetic_occlusion_audit import (
    _apply_audit_filters,
    _build_record,
    _prepare_audit_dataframe,
    _write_group_summary,
    build_coverage_summary,
)

from face_occlusion.data.synthetic_occlusion import OVERLAP_FLAG_KEYS, OVERLAP_METRIC_KEYS


def _metadata(value: float = 0.25) -> dict[str, object]:
    out: dict[str, object] = {
        "occluder_type": "random_face_rectangle",
        "num_attempts": 3,
    }
    for key in OVERLAP_METRIC_KEYS:
        out[key] = value
    for key in OVERLAP_FLAG_KEYS:
        out[key] = False
    return out


def test_audit_filters_target_gender_and_database():
    df = pd.DataFrame(
        {
            "filename": [
                "database1/a.png",
                "database3/person/b.png",
                "database3/person/c.png",
            ],
            "FaceOcclusion": [0.05, 0.45, 0.80],
            "gender": [0.0, 0.0, 1.0],
        }
    )
    prepared = _prepare_audit_dataframe(
        df,
        image_col="filename",
        target_col="FaceOcclusion",
        target_scale="unit",
        occlusion_bins=[0.0, 0.1, 0.4, 0.7, 1.0],
    )
    filtered = _apply_audit_filters(
        prepared,
        target_col="FaceOcclusion",
        gender_col="gender",
        target_min=0.4,
        target_max=0.7,
        database="database3",
        gender=0.0,
    )
    assert filtered["filename"].tolist() == ["database3/person/b.png"]
    assert filtered["occlusion_bin"].tolist() == ["0.40_0.70"]


def test_audit_record_includes_overlap_columns():
    image = Image.new("RGB", (8, 8))
    mild = SimpleNamespace(image=image, severity=0.1, metadata=_metadata(0.2))
    strong = SimpleNamespace(image=image, severity=0.4, metadata=_metadata(0.4))
    pair = SimpleNamespace(
        valid=True,
        mild=mild,
        strong=strong,
        metadata={
            "region_provider": "mediapipe",
            "mediapipe_valid": True,
            "failure_reason": None,
            "mild_occluder_type": "random_face_rectangle",
            "strong_occluder_type": "blurred_patch",
            "mild_attempts": 3,
            "strong_attempts": 4,
        },
    )
    row = pd.Series(
        {
            "FaceOcclusion": 0.4,
            "gender": 1.0,
            "database": "database1",
            "occlusion_bin": "0.40_0.60",
        }
    )
    record = _build_record(
        sample_index=0,
        row=row,
        image_path="database1/a.png",
        pair=pair,
        target_col="FaceOcclusion",
        gender_col="gender",
    )
    for prefix in ("mild", "strong"):
        for key in OVERLAP_METRIC_KEYS:
            assert f"{prefix}_{key}" in record
            assert np.isfinite(record[f"{prefix}_{key}"])
        for key in OVERLAP_FLAG_KEYS:
            assert f"{prefix}_{key}" in record
    assert record["database"] == "database1"
    assert record["occlusion_bin"] == "0.40_0.60"


def test_group_summary_writes_expected_columns(tmp_path):
    records = [
        {
            "sample_index": 0,
            "occlusion_bin": "0.40_0.60",
            "gender": 1.0,
            "database": "database1",
            "mediapipe_valid": True,
            "synthetic_valid": True,
            "mild_occluder_type": "a",
            "strong_occluder_type": "b",
            "mild_severity": 0.1,
            "strong_severity": 0.4,
            "mild_important_region_overlap": 0.2,
            "strong_important_region_overlap": 0.4,
            "mild_background_overlap_ratio": 0.1,
            "strong_background_overlap_ratio": 0.2,
            "num_attempts_mild": 2,
            "num_attempts_strong": 4,
        }
    ]
    out = tmp_path / "group.csv"
    _write_group_summary(records, out)
    df = pd.read_csv(out)
    assert df.loc[0, "count"] == 1
    assert df.loc[0, "valid_rate"] == 1.0
    assert "mean_strong_important_region_overlap" in df.columns


def _coverage_record(idx, bin_label, gender, mp_valid, ordering_ok):
    return {
        "sample_index": idx,
        "occlusion_bin": bin_label,
        "gender": gender,
        "mediapipe_valid": mp_valid,
        "ordering_ok": ordering_ok,
        # synthetic_valid is mediapipe AND ordering, mirroring the generator.
        "synthetic_valid": mp_valid and ordering_ok,
    }


def test_coverage_summary_rates_by_bin_and_gender():
    records = [
        # low-occ female: both succeed
        _coverage_record(0, "0.00_0.05", 0.0, True, True),
        _coverage_record(1, "0.00_0.05", 0.0, True, True),
        # high-occ male: MediaPipe fails half the time (the R8 concern)
        _coverage_record(2, "0.60_1.00", 1.0, True, True),
        _coverage_record(3, "0.60_1.00", 1.0, False, False),
    ]
    summary = build_coverage_summary(records)

    low = summary[(summary["occlusion_bin"] == "0.00_0.05") & (summary["gender"] == 0.0)].iloc[0]
    assert low["count"] == 2
    assert low["mediapipe_valid_rate"] == 1.0
    assert low["gender_label"] == "female"

    high = summary[(summary["occlusion_bin"] == "0.60_1.00") & (summary["gender"] == 1.0)].iloc[0]
    assert high["count"] == 2
    assert high["mediapipe_valid_rate"] == 0.5
    assert high["synthetic_valid_rate"] == 0.5
    assert high["gender_label"] == "male"


def test_coverage_summary_empty_records():
    assert build_coverage_summary([]).empty
