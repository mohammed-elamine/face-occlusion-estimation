#!/usr/bin/env python
"""Precompute MediaPipe face masks for ALL training images into a dedicated store.

Unlike the synthetic-occlusion cache (which masks only the ~hundreds of ranking anchors),
this covers every row of ``train.csv`` so background augmentation (and any future
label-preserving augmentation) can fetch a face mask for any training image. Masks are
written as one PNG per image, mirroring the image's relative path under ``--out-dir`` — the
mask path is derived from the image id, so there is no manifest.

Requires the ``synthetic`` extra (``uv sync --extra synthetic``) for MediaPipe.

Examples
--------
    # Full build (one-time; resumable).
    python -m scripts.data.build_face_masks \
        --config configs/baseline.yaml --out-dir data/face_masks --num-workers 8

    # Quick smoke run.
    python -m scripts.data.build_face_masks \
        --config configs/baseline.yaml --out-dir data/face_masks --limit 300
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from PIL import Image

from face_occlusion.data.normalize import assign_occlusion_bin, normalize_target
from face_occlusion.utils import load_config

# Per-worker globals (set in the pool initializer; one MediaPipe provider per process).
_PROVIDER = None
_IMAGE_ROOT: Path | None = None
_OUT_DIR: Path | None = None
_OVERWRITE = False


def _bin_labels(bins: list[float]) -> list[str]:
    return [f"{lo:.2f}_{hi:.2f}" for lo, hi in zip(bins[:-1], bins[1:])]


def _mask_path(out_dir: Path, sample_id: str) -> Path:
    return out_dir / Path(sample_id).with_suffix(".png")


def _init_worker(image_root: str, out_dir: str, overwrite: bool) -> None:
    global _PROVIDER, _IMAGE_ROOT, _OUT_DIR, _OVERWRITE
    from face_occlusion.data.synthetic_occlusion import MediaPipeFaceRegionProvider

    _PROVIDER = MediaPipeFaceRegionProvider()
    _IMAGE_ROOT = Path(image_root)
    _OUT_DIR = Path(out_dir)
    _OVERWRITE = overwrite


def _process_one(task: tuple[str, str, float]) -> tuple[str, float, str]:
    """Compute + save one face mask. Returns (bin_label, gender, status)."""
    sample_id, bin_label, gender = task
    out_path = _mask_path(_OUT_DIR, sample_id)
    if out_path.exists() and not _OVERWRITE:
        return bin_label, gender, "skipped"
    img_path = _IMAGE_ROOT / sample_id
    try:
        with Image.open(img_path) as im:
            image = im.convert("RGB")
    except Exception:
        return bin_label, gender, "load_error"
    result = _PROVIDER.extract(image)
    if not result.valid or "face" not in result.masks:
        return bin_label, gender, "no_face"
    face = (np.asarray(result.masks["face"]) > 0).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(face, mode="L").save(out_path)
    return bin_label, gender, "masked"


def _prepare_rows(cfg, limit: int | None) -> pd.DataFrame:
    """All rows of train.csv with normalized target + occlusion bin (no split/bin filter)."""
    id_col = cfg.data.id_col
    target_col = cfg.data.target_col
    gender_col = cfg.data.gender_col
    df = pd.read_csv(cfg.data.train_csv)
    df[target_col] = normalize_target(df[target_col], cfg.data.target_scale)
    bins = list(cfg.split.occlusion_bins)
    labels = _bin_labels(bins)
    idx = assign_occlusion_bin(df[target_col].to_numpy(), bins)
    df["_bin"] = [labels[i] for i in idx]
    out = pd.DataFrame(
        {
            "id": df[id_col].astype(str),
            "bin": df["_bin"].astype(str),
            "gender": df[gender_col].astype(float),
        }
    )
    if limit is not None:
        out = out.head(int(limit)).reset_index(drop=True)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("data/face_masks"))
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel MediaPipe workers (one provider each). 1 = serial (safest on macOS).",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap rows (smoke runs).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute masks even if the PNG exists (default: skip existing = resumable).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(str(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the MediaPipe model once (downloads to models/mediapipe/ if needed) BEFORE
    # spawning workers, so the worker providers find it and don't race to download.
    from face_occlusion.data.synthetic_occlusion import ensure_mediapipe_model

    model_path = ensure_mediapipe_model()
    if model_path is not None:
        print(f"[masks] MediaPipe model: {model_path}")

    rows = _prepare_rows(cfg, args.limit)
    tasks = list(zip(rows["id"], rows["bin"], rows["gender"]))
    n_total = len(tasks)
    image_root = str(cfg.data.image_root)
    print(f"[masks] {n_total} training images -> {out_dir} (workers={args.num_workers})")

    init_args = (image_root, str(out_dir), bool(args.overwrite))
    results: list[tuple[str, float, str]] = []
    if args.num_workers <= 1:
        _init_worker(*init_args)
        for i, task in enumerate(tasks):
            results.append(_process_one(task))
            if (i + 1) % 2000 == 0:
                print(f"[masks]   {i + 1}/{n_total}")
    else:
        with Pool(args.num_workers, initializer=_init_worker, initargs=init_args) as pool:
            for i, r in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=64)):
                results.append(r)
                if (i + 1) % 2000 == 0:
                    print(f"[masks]   {i + 1}/{n_total}")

    # Coverage: overall + by bin x gender.
    res_df = pd.DataFrame(results, columns=["bin", "gender", "status"])
    res_df["masked"] = res_df["status"].isin(["masked", "skipped"])
    n_masked = int(res_df["masked"].sum())
    status_counts = res_df["status"].value_counts().to_dict()
    by_cell = (
        res_df.assign(gender_label=res_df["gender"].map({0.0: "female", 1.0: "male"}))
        .groupby(["bin", "gender_label"])["masked"]
        .agg(n="size", masked="sum")
        .reset_index()
    )
    by_cell["coverage"] = by_cell["masked"] / by_cell["n"]
    coverage = {
        "n_total": n_total,
        "n_masked": n_masked,
        "coverage_pct": 100.0 * n_masked / n_total if n_total else 0.0,
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "by_bin_gender": by_cell.to_dict(orient="records"),
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))
    meta = {
        "source": "mediapipe_face_mesh.region_masks[face]",
        "content": "binary face mask (uint8 0/255 PNG; True>127)",
        "layout": "one PNG per image mirroring the image relative path (ext -> .png)",
        "config": str(args.config),
        "n_total": n_total,
        "n_masked": n_masked,
        "coverage_pct": coverage["coverage_pct"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[masks] done: {n_masked}/{n_total} masked "
        f"({coverage['coverage_pct']:.1f}%)  statuses={coverage['status_counts']}"
    )
    print("[masks] low-coverage cells (<90%):")
    low = by_cell[by_cell["coverage"] < 0.90].sort_values("coverage")
    if low.empty:
        print("  (none)")
    else:
        for _, r in low.iterrows():
            print(
                f"  bin={r['bin']:>10} {r['gender_label']:>6}: "
                f"{int(r['masked'])}/{int(r['n'])} ({100 * r['coverage']:.0f}%)"
            )
    print(f"[masks] wrote {out_dir / 'coverage.json'} + {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
