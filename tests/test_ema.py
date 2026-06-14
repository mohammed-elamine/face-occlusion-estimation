"""Tests for the EMA callback (shadow update, swap/restore, wiring)."""

from __future__ import annotations

import torch
import torch.nn as nn

from face_occlusion.training.callbacks import build_callbacks
from face_occlusion.training.ema import EMACallback
from face_occlusion.utils.config import Config


class _M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 1)


def test_shadow_init_equals_params():
    m = _M()
    cb = EMACallback(decay=0.9, warmup=False)
    cb.on_fit_start(None, m)
    for n, p in m.named_parameters():
        assert torch.allclose(cb._shadow[n], p)


def test_update_lags_toward_params():
    m = _M()
    cb = EMACallback(decay=0.9, warmup=False)
    cb.on_fit_start(None, m)
    old = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)  # live = old + 1
    cb.on_train_batch_end(None, m)
    # shadow = 0.9*old + 0.1*(old+1) = old + 0.1
    for n in old:
        assert torch.allclose(cb._shadow[n], old[n] + 0.1, atol=1e-6)


def test_effective_decay_warmup():
    cb = EMACallback(decay=0.999, warmup=True)
    cb._n_updates = 0
    assert abs(cb._effective_decay() - 1.0 / 10.0) < 1e-9  # small early
    cb._n_updates = 10**6
    assert abs(cb._effective_decay() - 0.999) < 1e-6  # converges to target


def test_swap_in_ema_then_restore_live():
    m = _M()
    cb = EMACallback(decay=0.9, warmup=False)
    cb.on_fit_start(None, m)
    for n in cb._shadow:  # set a distinctive EMA value
        cb._shadow[n].fill_(5.0)
    live = {n: p.detach().clone() for n, p in m.named_parameters()}
    cb.on_validation_epoch_start(None, m)  # swap EMA in (for val + ckpt)
    for _, p in m.named_parameters():
        assert torch.allclose(p, torch.full_like(p, 5.0))
    cb.on_train_epoch_start(None, m)  # restore live for next epoch
    for n, p in m.named_parameters():
        assert torch.allclose(p, live[n])


def test_decay_out_of_range_raises():
    import pytest

    with pytest.raises(ValueError):
        EMACallback(decay=1.0)
    with pytest.raises(ValueError):
        EMACallback(decay=0.0)


def _cfg(ema_enabled: bool) -> Config:
    return Config(
        {
            "checkpoint": Config(
                {"monitor": "val/score", "mode": "min", "save_top_k": 1, "filename": "best"}
            ),
            "training": Config(
                {"early_stopping_patience": 5, "ema": Config({"enabled": ema_enabled})}
            ),
        }
    )


def test_build_callbacks_gates_ema(tmp_path):
    on = build_callbacks(_cfg(True), checkpoint_dir=tmp_path)
    assert any(isinstance(c, EMACallback) for c in on)
    off = build_callbacks(_cfg(False), checkpoint_dir=tmp_path)
    assert not any(isinstance(c, EMACallback) for c in off)
