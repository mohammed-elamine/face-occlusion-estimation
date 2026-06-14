"""Load-bearing tests for callback wiring.

The whole pipeline selects models on ``val/score`` (min). A regression that
monitored ``val/loss`` or used ``mode=max`` would silently pick the wrong
checkpoint, so pin the contract here.
"""

from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from face_occlusion.training import build_callbacks
from face_occlusion.utils.config import Config


def _cfg(tmp_path) -> Config:
    return Config(
        {
            "checkpoint": {
                "monitor": "val/score",
                "mode": "min",
                "save_top_k": 1,
                "filename": "best",
            },
            "training": {"early_stopping_patience": 5},
        }
    )


def test_build_callbacks_monitors_val_score_min(tmp_path):
    callbacks = build_callbacks(_cfg(tmp_path), checkpoint_dir=tmp_path)

    ckpt = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    assert ckpt.monitor == "val/score"
    assert ckpt.mode == "min"
    assert ckpt.save_top_k == 1

    early = next(c for c in callbacks if isinstance(c, EarlyStopping))
    assert early.monitor == "val/score"
    assert early.mode == "min"
    assert early.patience == 5
