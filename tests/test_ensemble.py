"""Tests for prediction-ensemble averaging."""

import numpy as np
import pandas as pd
import pytest

from face_occlusion.inference import ensemble_average


def _frame(ids, preds, **extra):
    data = {"image_id": ids, "pred_clipped": preds}
    data.update(extra)
    return pd.DataFrame(data)


def test_equal_weight_mean():
    a = _frame(["x", "y"], [0.0, 1.0], filename=["x.webp", "y.webp"])
    b = _frame(["x", "y"], [0.2, 0.6])
    out = ensemble_average([a, b], keep_cols=["image_id", "filename"])
    assert out["pred_clipped"].tolist() == [0.1, 0.8]
    # passthrough + per-member columns are preserved.
    assert out["filename"].tolist() == ["x.webp", "y.webp"]
    assert out["member_0_pred_clipped"].tolist() == [0.0, 1.0]
    assert out["member_1_pred_clipped"].tolist() == [0.2, 0.6]


def test_weighted_average():
    a = _frame(["x"], [0.0])
    b = _frame(["x"], [1.0])
    out = ensemble_average([a, b], weights=[3, 1])
    assert out["pred_clipped"].iloc[0] == pytest.approx(0.25)


def test_alignment_ignores_row_order():
    a = _frame(["x", "y"], [0.0, 1.0])
    b = _frame(["y", "x"], [0.6, 0.2])  # reversed order, same keys
    out = ensemble_average([a, b]).set_index("image_id")["pred_clipped"]
    assert out.loc["x"] == pytest.approx(0.1)
    assert out.loc["y"] == pytest.approx(0.8)


def test_missing_key_raises():
    a = _frame(["x", "y"], [0.0, 1.0])
    b = _frame(["x"], [0.2])
    with pytest.raises(ValueError, match="missing"):
        ensemble_average([a, b])


def test_duplicate_key_raises():
    a = _frame(["x", "x"], [0.0, 1.0])
    b = _frame(["x", "y"], [0.2, 0.6])
    with pytest.raises(ValueError, match="duplicate"):
        ensemble_average([a, b])


def test_bad_weights_raise():
    a = _frame(["x"], [0.0])
    b = _frame(["x"], [1.0])
    with pytest.raises(ValueError, match="weights length"):
        ensemble_average([a, b], weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_average([a, b], weights=[0.0, 0.0])


def test_ensemble_of_clipped_stays_in_range():
    rng = np.random.default_rng(0)
    ids = [str(i) for i in range(50)]
    frames = [_frame(ids, rng.uniform(0, 1, size=50)) for _ in range(4)]
    out = ensemble_average(frames)
    assert out["pred_clipped"].between(0.0, 1.0).all()
