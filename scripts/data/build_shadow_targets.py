#!/usr/bin/env python
"""Precompute the within-face deep-shadow fraction (``dark_frac``) for every training image.

These are the targets for the auxiliary shadow head (``model.use_shadow_head``). Offline,
run once; the datamodule merges the resulting CSV onto the train rows by ``id_col``.

    python -m scripts.data.build_shadow_targets \
        --config configs/convnext_ablation/09_shadow_aux_head.yaml

Output CSV columns: ``<id_col>, dark_frac`` (NaN when the face mesh could not be fit). Needs the
``synthetic`` extra (mediapipe) and the face-landmarker model (``make mediapipe-model``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from PIL import Image

from face_occlusion.data.synthetic_occlusion import _resolve_mediapipe_model_asset_path
from face_occlusion.utils import load_config


def _make_landmarker(model_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return mp, vision.FaceLandmarker.create_from_options(opts)


def _dark_frac(img_rgb: np.ndarray, mp, landmarker) -> float:
    import cv2

    h, w = img_rgb.shape[:2]
    gray = img_rgb.mean(2)
    res = landmarker.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
    )
    if not res.face_landmarks:
        return float("nan")
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 1)
    px = gray[mask.astype(bool)]
    if px.size < 100:
        return float("nan")
    return float((px < 0.6 * px.mean()).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="Output CSV (default: data.shadow_targets_csv).")
    ap.add_argument(
        "--limit", type=int, default=None, help="Process only the first N rows (debug)."
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(
        args.out or cfg.data.get("shadow_targets_csv", "data/shadow_targets/train_dark_frac.csv")
    )
    id_col = cfg.data.id_col
    image_root = Path(cfg.data.image_root)

    model_path = _resolve_mediapipe_model_asset_path()
    if model_path is None:
        raise SystemExit(
            "[shadow] face_landmarker.task not found. Run `make mediapipe-model` or "
            "`python -m scripts.data.download_mediapipe_model` first."
        )
    mp, landmarker = _make_landmarker(str(model_path))

    df = pd.read_csv(cfg.data.train_csv).dropna(subset=[id_col])
    if args.limit:
        df = df.head(args.limit)
    total = len(df)
    print(f"[shadow] computing dark_frac for {total} rows -> {out}")

    vals, n_fail = [], 0
    for i, fname in enumerate(df[id_col].astype(str)):
        try:
            arr = np.asarray(Image.open(image_root / fname).convert("RGB"))
            d = _dark_frac(arr, mp, landmarker)
        except Exception:
            d = float("nan")
        if not np.isfinite(d):
            n_fail += 1
        vals.append(d)
        if (i + 1) % 5000 == 0:
            print(f"[shadow]   {i + 1}/{total} ({n_fail} mesh-fail so far)")
    landmarker.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({id_col: df[id_col].astype(str).to_numpy(), "dark_frac": vals}).to_csv(
        out, index=False
    )
    ok = total - n_fail
    print(f"[shadow] wrote {out} | {ok}/{total} with dark_frac ({n_fail} mesh-fail -> NaN)")


if __name__ == "__main__":
    main()
