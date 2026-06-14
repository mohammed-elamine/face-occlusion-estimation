"""Tests for the interrupt-safe finalize step in scripts.training.train."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.training.train import _finalize

_VAL_OUTPUTS = {
    "preds": [0.1, 0.5],
    "targets": [0.2, 0.4],
    "image_ids": ["a", "b"],
    "filenames": ["a.webp", "b.webp"],
    "paths": ["a", "b"],
    "genders": [0.0, 1.0],
    "databases": ["d1", "d1"],
    "source_subfolders": ["s", "s"],
    "group_ids": ["g1", "g2"],
    "face_ids": [0, 0],
}


def test_finalize_saves_predictions_and_status_from_memory(tmp_path):
    # No best checkpoint, but last-epoch outputs in memory -> predictions are still saved.
    module = SimpleNamespace(_last_val_outputs=dict(_VAL_OUTPUTS))
    trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=""))
    _finalize(trainer, module, None, tmp_path, interrupted=True)
    assert (tmp_path / "predictions" / "val_predictions.csv").exists()
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["status"] == "interrupted"
    assert status["predictions"] is not None


def test_finalize_never_raises_when_nothing_to_save(tmp_path):
    # No checkpoint, no in-memory outputs, and validate() blows up -> must not raise.
    class _Trainer:
        checkpoint_callback = SimpleNamespace(best_model_path="")

        def validate(self, *a, **k):
            raise RuntimeError("boom")

    module = SimpleNamespace(_last_val_outputs=None)
    _finalize(_Trainer(), module, None, tmp_path, interrupted=False)  # should be a no-op, no raise
    status = json.loads((tmp_path / "training_status.json").read_text())
    assert status["status"] == "completed"
    assert status["predictions"] is None
