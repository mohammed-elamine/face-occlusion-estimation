"""Tiny YAML config loader exposing dotted-attribute access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also exposes its keys as attributes, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _wrap(value)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return _wrap(super().get(key, default))


def _wrap(value: Any) -> Any:
    # Wrap nested mappings lazily so cfg.data.train_csv works at any depth.
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config(value)
    return value


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return Config(raw)
