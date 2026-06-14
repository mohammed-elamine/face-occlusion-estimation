"""Tests for challenge submission generation."""

import pandas as pd
from scripts.inference.predict_test import build_submission

from face_occlusion.utils import Config


def test_build_submission_adds_dummy_gender():
    cfg = Config(
        {
            "data": {
                "image_col": "filename",
                "target_col": "FaceOcclusion",
                "gender_col": "gender",
                "submission_dummy_gender": 0,
            }
        }
    )
    preds = pd.DataFrame(
        {
            "filename": ["database1/img00000004.webp"],
            "pred_clipped": [0.12],
        }
    )

    submission = build_submission(preds, cfg)

    assert submission.columns.tolist() == ["filename", "FaceOcclusion", "gender"]
    assert submission["gender"].tolist() == [0]
