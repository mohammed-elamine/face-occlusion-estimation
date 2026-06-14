#!/usr/bin/env python
"""Download the MediaPipe Face Landmarker model to models/mediapipe/.

Used by `make mediapipe-model`. The builders (build_face_masks / build_synthetic_cache)
also auto-download on demand, so this is for an explicit, offline-friendly pre-fetch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

from face_occlusion.data.synthetic_occlusion import (
    DEFAULT_MEDIAPIPE_MODEL_DIR,
    download_mediapipe_model,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_MEDIAPIPE_MODEL_DIR / "face_landmarker.task",
        help="Where to write the .task model (default: models/mediapipe/face_landmarker.task).",
    )
    p.add_argument(
        "--force", action="store_true", help="Re-download even if the model already exists."
    )
    args = p.parse_args()
    if args.dest.exists() and not args.force:
        print(f"[mediapipe] already present: {args.dest} (use --force to re-download)")
        return
    path = download_mediapipe_model(args.dest)
    print(f"[mediapipe] downloaded model to {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
