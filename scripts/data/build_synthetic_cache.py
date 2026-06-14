#!/usr/bin/env python
"""Precompute the synthetic-occlusion ranking cache (Stage 4 prerequisite).

For a balanced occlusion_bin x gender sample of TRAIN anchors, generate one
``clean | mild | strong`` view triple per anchor and write them to disk with a
manifest. Training then loads these views instead of running MediaPipe on the
fly. Only same-anchor pairs are produced (identity/pose/lighting/gender fixed —
only occlusion changes), and only MediaPipe-valid pairs are kept.

Examples
--------
    # Default: tail-focused, balanced sample of train anchors.
    python -m scripts.data.build_synthetic_cache \
        --config configs/baseline.yaml \
        --cache-dir data/synthetic_cache/v1

    # Scale up / widen the regime.
    python -m scripts.data.build_synthetic_cache --config configs/baseline.yaml \
        --cache-dir data/synthetic_cache/v2 --max-per-bin-gender 400 --target-min 0.0
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

from face_occlusion.data.metadata import add_path_metadata
from face_occlusion.data.normalize import assign_occlusion_bin, normalize_target
from face_occlusion.data.splits import load_split
from face_occlusion.data.synthetic_cache import (
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    coverage_table,
    mask_filename,
    select_balanced_anchors,
    view_filenames,
)
from face_occlusion.data.synthetic_occlusion import (
    DEFAULT_OCCLUDER_TYPES,
    DEFAULT_REGION_WEIGHTS,
    DEFAULT_SEVERITY_BANDS,
    SyntheticOcclusionGenerator,
    build_generator_from_config,
)
from face_occlusion.utils import load_config, seed_everything


def _bin_labels(bins: list[float]) -> list[str]:
    return [f"{lo:.2f}_{hi:.2f}" for lo, hi in zip(bins[:-1], bins[1:])]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path, help="Output cache directory.")
    p.add_argument(
        "--max-per-bin-gender",
        type=int,
        default=200,
        help="Cap on anchors sampled per occlusion_bin x gender cell.",
    )
    p.add_argument(
        "--target-min",
        type=float,
        default=0.10,
        help="Only cache anchors with normalized target >= this (tail-focused by default).",
    )
    p.add_argument("--target-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality", type=int, default=95, help="WEBP quality for saved views (1-100).")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional hard cap on total anchors (for quick smoke runs).",
    )
    return p.parse_args()


def _build_generator(cfg, seed: int) -> SyntheticOcclusionGenerator:
    generator = build_generator_from_config(cfg)
    if generator is None:
        # Build from defaults so the cache works even when training opt-in is off.
        generator = SyntheticOcclusionGenerator(
            severity_bands=DEFAULT_SEVERITY_BANDS,
            region_weights=DEFAULT_REGION_WEIGHTS,
            occluder_types=DEFAULT_OCCLUDER_TYPES,
            seed=seed,
        )
    return generator


def _prepare_train_anchors(cfg, target_min: float, target_max: float) -> pd.DataFrame:
    """Train-split rows with normalized target + occlusion_bin, filtered by target."""
    id_col = cfg.data.id_col
    target_col = cfg.data.target_col
    df = add_path_metadata(pd.read_csv(cfg.data.train_csv), filename_col=id_col)
    df[target_col] = normalize_target(df[target_col], cfg.data.target_scale)

    # Restrict to TRAIN rows so no validation image is ever used as a source.
    split_path = Path(cfg.split.split_path)
    if split_path.exists():
        split = load_split(split_path)[[id_col, "split"]]
        df = df.merge(split, on=id_col, how="inner")
        df = df[df["split"] == "train"].reset_index(drop=True)
    else:
        print(f"[cache] WARNING: split {split_path} not found; using ALL rows as anchors.")

    bins = list(cfg.split.occlusion_bins)
    labels = _bin_labels(bins)
    idx = assign_occlusion_bin(df[target_col].to_numpy(), bins)
    df["occlusion_bin"] = [labels[i] for i in idx]
    mask = (df[target_col] >= target_min) & (df[target_col] <= target_max)
    return df[mask].reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed_everything(int(args.seed))

    size = int(cfg.augmentation.resize)
    image_root = Path(cfg.data.image_root)
    id_col = cfg.data.id_col
    gender_col = cfg.data.gender_col
    image_col = cfg.data.image_col

    # Ensure the MediaPipe model is present (downloads to models/mediapipe/ if needed).
    from face_occlusion.data.synthetic_occlusion import ensure_mediapipe_model

    ensure_mediapipe_model()

    generator = _build_generator(cfg, args.seed)
    anchors = _prepare_train_anchors(cfg, args.target_min, args.target_max)
    if anchors.empty:
        raise ValueError("No train anchors match the requested target range.")

    rng = np.random.default_rng(int(args.seed))
    selected = select_balanced_anchors(
        anchors,
        bin_col="occlusion_bin",
        gender_col=gender_col,
        max_per_bin_gender=int(args.max_per_bin_gender),
        rng=rng,
    )
    if args.limit is not None:
        selected = selected.iloc[: int(args.limit)]
    print(f"[cache] selected {len(selected)} anchors from {len(anchors)} eligible train rows")

    cache_dir = args.cache_dir
    (cache_dir / "views").mkdir(parents=True, exist_ok=True)
    (cache_dir / "masks").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    n_invalid = 0
    for write_idx, (_, row) in enumerate(selected.iterrows()):
        rel = str(row[image_col])
        path = image_root / rel
        with Image.open(path) as im:
            img = im.convert("RGB").resize((size, size), Image.BILINEAR)
        # Deterministic per-anchor RNG keyed by the source id, so re-runs and the
        # on-the-fly dataset path stay reproducible.
        anchor_rng = np.random.default_rng([int(args.seed), write_idx])
        pair = generator.generate_pair(img, rng=anchor_rng)
        if not (pair.valid and pair.mild is not None and pair.strong is not None):
            n_invalid += 1
            continue

        names = view_filenames(write_idx)
        img.save(cache_dir / names["clean"], quality=int(args.quality))
        pair.mild.image.save(cache_dir / names["mild"], quality=int(args.quality))
        pair.strong.image.save(cache_dir / names["strong"], quality=int(args.quality))
        # Save the face mask (for label-preserving background augmentation).
        mask_rel = mask_filename(write_idx)
        face_mask = pair.region_masks.get("face")
        if face_mask is not None:
            Image.fromarray((np.asarray(face_mask) > 0).astype("uint8") * 255, mode="L").save(
                cache_dir / mask_rel
            )
        records.append(
            {
                "id": str(row[id_col]),
                "occlusion_bin": str(row["occlusion_bin"]),
                "gender": float(row[gender_col]),
                "clean_path": names["clean"],
                "mild_path": names["mild"],
                "strong_path": names["strong"],
                "mask_path": mask_rel if face_mask is not None else None,
                "mild_severity": float(pair.mild.severity),
                "strong_severity": float(pair.strong.severity),
                "mild_occluder_type": pair.metadata.get("mild_occluder_type"),
                "strong_occluder_type": pair.metadata.get("strong_occluder_type"),
            }
        )

    manifest = pd.DataFrame(records, columns=list(MANIFEST_COLUMNS))
    manifest.to_csv(cache_dir / MANIFEST_FILENAME, index=False)

    print(f"[cache] wrote {len(manifest)} valid pairs ({n_invalid} skipped) to {cache_dir}")
    print(f"[cache] manifest: {cache_dir / MANIFEST_FILENAME}")
    bins = list(cfg.split.occlusion_bins)
    print("[cache] coverage by occlusion_bin x gender:")
    print(coverage_table(manifest, bin_order=_bin_labels(bins)).to_string(index=False))


if __name__ == "__main__":
    main()
