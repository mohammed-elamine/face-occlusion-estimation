"""Tests for challenge submission generation."""

import pandas as pd
from scripts.inference.predict_test import build_submission

from face_occlusion.utils import Config


def _cfg(**data_overrides):
    data = {
        "image_col": "filename",
        "target_col": "FaceOcclusion",
        "gender_col": "gender",
    }
    data.update(data_overrides)
    return Config({"data": data})


def _preds():
    return pd.DataFrame({"filename": ["database1/img00000004.webp"], "pred_clipped": [0.12]})


def test_build_submission_adds_dummy_gender():
    submission = build_submission(_preds(), _cfg(submission_dummy_gender=0))

    assert submission.columns.tolist() == ["filename", "FaceOcclusion", "gender"]
    assert submission["gender"].tolist() == [0]


def test_build_submission_defaults_to_teacher_dummy():
    # Matches the teacher's example: results_df['gender'] = 'x'.
    submission = build_submission(_preds(), _cfg())

    assert submission.columns.tolist() == ["filename", "FaceOcclusion", "gender"]
    assert submission["gender"].tolist() == ["x"]


def test_build_submission_dummy_gender_override():
    # Explicit override wins over the config value (used by the ensemble writer).
    submission = build_submission(_preds(), _cfg(submission_dummy_gender=0), dummy_gender="x")

    assert submission["gender"].tolist() == ["x"]
