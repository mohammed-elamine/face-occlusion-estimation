"""Helpers for self-contained experiment directories."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

RUN_SUBDIRS = ("checkpoints", "logs", "predictions", "reports", "splits")


def _config_get(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    # Support both plain dicts and the project's Config dotted-access wrapper.
    current = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
            continue
        try:
            current = getattr(current, key)
        except AttributeError:
            return default
    return current


def _slugify(value: str) -> str:
    # Run ids are used as folder names, so keep only portable characters.
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    slug = re.sub(r"-{2,}", "-", slug).strip(".-_")
    return slug or "run"


def to_plain_dict(value: Any) -> Any:
    """Recursively convert Config-like mappings into YAML/JSON-friendly objects."""

    if isinstance(value, Mapping):
        return {str(k): to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain_dict(v) for v in value]
    return value


def create_run_dir(config: Mapping[str, Any] | Any) -> Path:
    """Create and return a unique experiment run directory."""

    run_name = _config_get(
        config,
        ("experiment", "name"),
        _config_get(config, ("logging", "run_name"), "experiment"),
    )
    output_root = _config_get(config, ("experiment", "output_root"), None)
    if output_root is None:
        project_output = _config_get(config, ("project", "output_dir"), "outputs")
        output_root = Path(project_output) / "experiments"

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    base_run_id = f"{timestamp}_{_slugify(str(run_name))}"
    run_dir = output_root / base_run_id
    suffix = 1
    # Avoid overwriting if two runs start within the same second.
    while run_dir.exists():
        run_dir = output_root / f"{base_run_id}-{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir()
    return run_dir


def save_config_snapshot(config: Mapping[str, Any] | Any, run_dir: Path) -> Path:
    """Save the active configuration inside the run directory."""

    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(to_plain_dict(config), sort_keys=False), encoding="utf-8")
    return path


def _git_output(args: list[str]) -> str:
    # Git metadata is useful, but training should still run outside a git checkout.
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        return f"unavailable: {message}"
    return result.stdout.strip()


def save_git_info(run_dir: Path) -> str:
    """Save the current Git commit and working-tree status."""

    commit = _git_output(["rev-parse", "HEAD"])
    status = _git_output(["status", "--short", "--branch"])

    (run_dir / "git_commit.txt").write_text(f"{commit}\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text(f"{status}\n", encoding="utf-8")
    return commit


def save_metadata(
    config: Mapping[str, Any] | Any,
    run_dir: Path,
    config_path: str | Path | None = None,
) -> Path:
    """Write lightweight run metadata for cluster and local analysis."""

    status = _git_output(["status", "--short"])
    metadata = {
        "run_id": run_dir.name,
        "run_name": _config_get(
            config,
            ("experiment", "name"),
            _config_get(config, ("logging", "run_name"), run_dir.name),
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": _config_get(config, ("project", "seed")),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(status.strip()),
        "config_path": str(config_path) if config_path is not None else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def write_latest_run_pointer(run_dir: Path, output_root: str | Path) -> Path:
    """Write a small text pointer to the most recent run directory."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pointer = output_root / "latest_run.txt"
    pointer.write_text(f"{run_dir.resolve()}\n", encoding="utf-8")
    return pointer
