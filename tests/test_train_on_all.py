"""train_on_all (final-submission refit) changes the model-selection policy."""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from face_occlusion.training import build_callbacks
from face_occlusion.utils import Config


def _cfg(train_on_all: bool) -> Config:
    return Config(
        {
            "checkpoint": {
                "monitor": "val/score",
                "mode": "min",
                "save_top_k": 1,
                "filename": "best",
            },
            "training": {"early_stopping_patience": 5},
            "split": {"train_on_all": train_on_all},
        }
    )


def _checkpoint(callbacks) -> ModelCheckpoint:
    return next(c for c in callbacks if isinstance(c, ModelCheckpoint))


def test_normal_mode_selects_on_val(tmp_path):
    callbacks = build_callbacks(_cfg(False), checkpoint_dir=tmp_path)
    assert any(isinstance(c, EarlyStopping) for c in callbacks)
    ckpt = _checkpoint(callbacks)
    assert ckpt.monitor == "val/score"
    assert ckpt.save_top_k == 1
    assert ckpt.save_last


def test_train_on_all_drops_selection(tmp_path):
    callbacks = build_callbacks(_cfg(True), checkpoint_dir=tmp_path)
    # No early stopping on the leaked monitor.
    assert not any(isinstance(c, EarlyStopping) for c in callbacks)
    ckpt = _checkpoint(callbacks)
    # No best.ckpt (would be selected on a leaked val); only last.ckpt is kept.
    assert ckpt.monitor is None
    assert ckpt.save_top_k == 0
    assert ckpt.save_last


def test_train_on_all_defaults_off(tmp_path):
    # Absent split section behaves like normal training (selection enabled).
    cfg = Config(
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
    callbacks = build_callbacks(cfg, checkpoint_dir=tmp_path)
    assert any(isinstance(c, EarlyStopping) for c in callbacks)
    assert _checkpoint(callbacks).save_top_k == 1
