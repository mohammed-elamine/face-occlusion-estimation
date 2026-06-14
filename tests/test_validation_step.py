"""End-to-end test of the validation buffer -> val/score contract.

The challenge score is computed once per epoch over the whole concatenated
validation set (it is grouped by gender). These tests drive ``validation_step``
followed by ``on_validation_epoch_end`` on a stub model and assert the logged
``val/score`` / ``val/err_female`` / ``val/err_male`` match a direct
``challenge_score`` computation. Previously only the hand-seeded buffer path was
covered, so the buffer-population contract (gender formatting, clip) was never
verified end to end.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from face_occlusion.metrics.challenge_metric import challenge_score
from face_occlusion.models.outputs import OcclusionModelOutput
from face_occlusion.training.lit_module import FaceOcclusionLitModule

BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]


class _IdentityModel(torch.nn.Module):
    """Returns the input batch as the regression prediction (no ordinal head)."""

    use_ordinal_head = False

    def forward(self, x):
        return OcclusionModelOutput(y_pred=x.reshape(-1).float())


def _make_module() -> FaceOcclusionLitModule:
    module = FaceOcclusionLitModule.__new__(FaceOcclusionLitModule)
    pl.LightningModule.__init__(module)
    module.cfg = SimpleNamespace(split=SimpleNamespace(occlusion_bins=BINS))
    module._val_buffer = []
    module._female_value = "0.0"
    module._male_value = "1.0"
    module.model = _IdentityModel()
    module._ord_loss_enabled = False
    module._cons_loss_enabled = False
    module._mono_loss_enabled = False
    return module


def _batch(preds, targets, genders) -> dict:
    n = len(preds)
    return {
        "image": torch.tensor(preds, dtype=torch.float32),
        "target": torch.tensor(targets, dtype=torch.float32),
        "gender": torch.tensor(genders, dtype=torch.float32),
        "image_id": [f"id_{i}" for i in range(n)],
        "filename": [f"f_{i}.webp" for i in range(n)],
        "path": [f"/root/f_{i}.webp" for i in range(n)],
        "database": ["database3"] * n,
        "source_subfolder": ["sub"] * n,
        "group_id": [f"g_{i}" for i in range(n)],
        "face_id": torch.zeros(n, dtype=torch.int64),
    }


def test_validation_step_to_val_score_matches_challenge_metric():
    preds = [0.10, 0.20, 0.50, 0.90, 0.05, 0.30, 0.70, 0.45]
    targets = [0.00, 0.10, 0.60, 1.00, 0.00, 0.40, 0.80, 0.40]
    genders = [0, 0, 0, 0, 1, 1, 1, 1]

    module = _make_module()
    logs: dict[str, float] = {}
    module.log = lambda name, value, *a, **k: logs.__setitem__(  # type: ignore[assignment]
        name, float(value.detach().item()) if torch.is_tensor(value) else float(value)
    )

    # Two batches to confirm the score pools across the whole epoch.
    module.validation_step(_batch(preds[:4], targets[:4], genders[:4]), 0)
    module.validation_step(_batch(preds[4:], targets[4:], genders[4:]), 1)
    module.on_validation_epoch_end()

    gender_str = np.array([f"{float(g):.1f}" for g in genders])
    expected = challenge_score(
        np.array(preds),
        np.array(targets),
        gender_str,
        female_value="0.0",
        male_value="1.0",
    )

    # abs tolerance accommodates float32 (module) vs float64 (numpy expected).
    assert logs["val/score"] == pytest.approx(expected["score"], abs=1e-5)
    assert logs["val/err_female"] == pytest.approx(expected["err_female"], abs=1e-5)
    assert logs["val/err_male"] == pytest.approx(expected["err_male"], abs=1e-5)
    assert logs["val/gender_gap"] == pytest.approx(expected["gender_gap"], abs=1e-5)
    # Buffer is cleared so the next epoch starts fresh.
    assert module._val_buffer == []


def test_validation_step_clips_predictions_for_score():
    # A raw prediction above 1.0 must be clipped before the metric (and the
    # raw range must still be surfaced).
    preds = [1.5, 0.0]
    targets = [1.0, 0.0]
    genders = [0, 1]

    module = _make_module()
    logs: dict[str, float] = {}
    module.log = lambda name, value, *a, **k: logs.__setitem__(  # type: ignore[assignment]
        name, float(value.detach().item()) if torch.is_tensor(value) else float(value)
    )
    module.validation_step(_batch(preds, targets, genders), 0)
    module.on_validation_epoch_end()

    # Clipped pred 1.0 vs target 1.0 -> err_female 0; raw max still reported.
    assert logs["val/err_female"] == pytest.approx(0.0)
    assert logs["val/pred_max_raw"] == pytest.approx(1.5)
    assert logs["val/pct_pred_above_1"] == pytest.approx(0.5)
