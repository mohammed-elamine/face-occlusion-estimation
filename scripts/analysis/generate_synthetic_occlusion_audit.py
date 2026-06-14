#!/usr/bin/env python
"""Visual audit for the synthetic occlusion generator (Stage 3).

Samples a handful of training images, runs the generator, and saves a grid +
per-sample images and a CSV summarising severities and occluder types so the
generator can be inspected before being plugged into training.

Example
-------
    python scripts/analysis/generate_synthetic_occlusion_audit.py \\
        --config configs/occlusion_aware_contrastive/00_baseline.yaml \\
        --num-samples 16 \\
        --output-dir outputs/reports/synthetic_occlusion_audit
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from face_occlusion.data.metadata import add_path_metadata
from face_occlusion.data.synthetic_occlusion import (
    DEFAULT_OCCLUDER_TYPES,
    DEFAULT_REGION_WEIGHTS,
    DEFAULT_SEVERITY_BANDS,
    OVERLAP_FLAG_KEYS,
    OVERLAP_METRIC_KEYS,
    SyntheticOcclusionGenerator,
    SyntheticOcclusionView,
    build_generator_from_config,
)
from face_occlusion.utils.config import load_config

GROUP_SUMMARY_COLUMNS = (
    "occlusion_bin",
    "gender",
    "database",
    "mediapipe_valid",
    "synthetic_valid",
    "mild_occluder_type",
    "strong_occluder_type",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Training config YAML.")
    p.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of training images to audit.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/synthetic_occlusion_audit"),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the resize target (defaults to cfg.augmentation.resize).",
    )
    p.add_argument("--target-min", type=float, default=None, help="Minimum normalized target.")
    p.add_argument("--target-max", type=float, default=None, help="Maximum normalized target.")
    p.add_argument("--database", type=str, default=None, help="Audit only one database.")
    p.add_argument("--gender", type=float, default=None, help="Audit only one gender value.")
    p.add_argument(
        "--coverage-only",
        action="store_true",
        help="Skip the visual grid and per-sample PNGs; only compute coverage stats. "
        "Use for large samples to measure the bin x gender gate cheaply.",
    )
    return p.parse_args()


def _normalize_target_scale(values: pd.Series, scale: str) -> pd.Series:
    if scale == "unit":
        return values.astype(float)
    if scale == "percent":
        return values.astype(float) / 100.0
    if values.astype(float).max() > 1.5:
        return values.astype(float) / 100.0
    return values.astype(float)


def _bin_labels(bins: list[float]) -> list[str]:
    return [f"{lo:.2f}_{hi:.2f}" for lo, hi in zip(bins[:-1], bins[1:])]


def _prepare_audit_dataframe(
    df: pd.DataFrame,
    *,
    image_col: str,
    target_col: str,
    target_scale: str,
    occlusion_bins: list[float],
) -> pd.DataFrame:
    out = add_path_metadata(df, filename_col=image_col)
    if target_col in out.columns:
        out[target_col] = _normalize_target_scale(out[target_col], target_scale)
        out["occlusion_bin"] = pd.cut(
            out[target_col],
            bins=occlusion_bins,
            labels=_bin_labels(occlusion_bins),
            include_lowest=True,
            right=False,
        ).astype(str)
        out.loc[out[target_col] >= occlusion_bins[-1], "occlusion_bin"] = _bin_labels(
            occlusion_bins
        )[-1]
    else:
        out["occlusion_bin"] = "unknown"
    return out


def _apply_audit_filters(
    df: pd.DataFrame,
    *,
    target_col: str,
    gender_col: str,
    target_min: float | None,
    target_max: float | None,
    database: str | None,
    gender: float | None,
) -> pd.DataFrame:
    out = df
    if target_min is not None and target_col in out.columns:
        out = out[out[target_col].astype(float) >= float(target_min)]
    if target_max is not None and target_col in out.columns:
        out = out[out[target_col].astype(float) <= float(target_max)]
    if database is not None:
        out = out[out["database"].astype(str) == str(database)]
    if gender is not None and gender_col in out.columns:
        out = out[out[gender_col].astype(float) == float(gender)]
    return out


def _view_metrics(prefix: str, view: SyntheticOcclusionView | None) -> dict[str, object]:
    record: dict[str, object] = {}
    for key in OVERLAP_METRIC_KEYS:
        record[f"{prefix}_{key}"] = (
            float(view.metadata[key]) if view is not None and key in view.metadata else None
        )
    for key in OVERLAP_FLAG_KEYS:
        record[f"{prefix}_{key}"] = (
            bool(view.metadata[key]) if view is not None and key in view.metadata else None
        )
    return record


def _maybe_overlay_regions(ax, region_masks: dict[str, np.ndarray]) -> None:
    if not region_masks or "face" not in region_masks:
        ax.text(
            0.5,
            0.5,
            "regions\nFAILED",
            ha="center",
            va="center",
            fontsize=9,
            color="red",
            transform=ax.transAxes,
        )
        return
    palette = {
        "eyes": (1.0, 0.2, 0.2, 0.55),
        "mouth": (0.2, 0.6, 1.0, 0.55),
        "nose": (1.0, 0.85, 0.1, 0.55),
        "cheeks": (0.3, 0.85, 0.3, 0.30),
        "forehead_chin": (0.7, 0.4, 0.9, 0.30),
    }
    h, w = region_masks["face"].shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for name, rgba in palette.items():
        m = region_masks.get(name)
        if m is None:
            continue
        overlay[m] = rgba
    ax.imshow(overlay)


def _build_record(
    *,
    sample_index: int,
    row: pd.Series,
    image_path: Path,
    pair,
    target_col: str,
    gender_col: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "sample_index": int(sample_index),
        "image_path": str(image_path),
        "target": float(row[target_col]) if target_col in row else None,
        "occlusion_bin": str(row.get("occlusion_bin", "unknown")),
        "gender": float(row[gender_col]) if gender_col in row else None,
        "database": str(row.get("database", "")),
        "provider": pair.metadata.get("region_provider"),
        "mediapipe_valid": bool(pair.metadata.get("mediapipe_valid", False)),
        "failure_reason": pair.metadata.get("failure_reason"),
        "ordering_ok": bool(pair.metadata.get("ordering_ok", False)),
        "synthetic_valid": bool(pair.valid),
        "mild_severity": (float(pair.mild.severity) if pair.mild is not None else None),
        "strong_severity": (float(pair.strong.severity) if pair.strong is not None else None),
        "mild_occluder_type": pair.metadata.get("mild_occluder_type"),
        "strong_occluder_type": pair.metadata.get("strong_occluder_type"),
        "num_attempts_mild": int(pair.metadata.get("mild_attempts", 0)),
        "num_attempts_strong": int(pair.metadata.get("strong_attempts", 0)),
    }
    record.update(_view_metrics("mild", pair.mild))
    record.update(_view_metrics("strong", pair.strong))
    return record


def _write_group_summary(records: list[dict[str, object]], csv_path: Path) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        df.to_csv(csv_path, index=False)
        return
    grouped = df.copy()
    for col in GROUP_SUMMARY_COLUMNS:
        if col not in grouped.columns:
            grouped[col] = "missing"
    for col in ("mild_occluder_type", "strong_occluder_type", "failure_reason"):
        if col in grouped.columns:
            grouped[col] = grouped[col].fillna("none")
    summary = (
        grouped.groupby(list(GROUP_SUMMARY_COLUMNS), dropna=False)
        .agg(
            count=("sample_index", "size"),
            valid_rate=("synthetic_valid", "mean"),
            mean_mild_severity=("mild_severity", "mean"),
            mean_strong_severity=("strong_severity", "mean"),
            mean_mild_important_region_overlap=("mild_important_region_overlap", "mean"),
            mean_strong_important_region_overlap=("strong_important_region_overlap", "mean"),
            mean_mild_background_overlap_ratio=("mild_background_overlap_ratio", "mean"),
            mean_strong_background_overlap_ratio=("strong_background_overlap_ratio", "mean"),
            mean_num_attempts_mild=("num_attempts_mild", "mean"),
            mean_num_attempts_strong=("num_attempts_strong", "mean"),
        )
        .reset_index()
        .sort_values(["count", "valid_rate"], ascending=[False, True])
    )
    summary.to_csv(csv_path, index=False)


_GENDER_LABELS = {0.0: "female", 1.0: "male"}
# High-occlusion bins are the ones synthetic ranking most needs to cover, and
# the ones MediaPipe is most likely to fail on (review R8).
_HIGH_OCC_BINS = ("0.40_0.60", "0.60_1.00")


def build_coverage_summary(records: list[dict[str, object]]) -> pd.DataFrame:
    """MediaPipe / ordering success rates grouped by occlusion_bin x gender.

    This is the Stage 3->4 gate: if MediaPipe success (and hence synthetic
    coverage) collapses in the high-occlusion bins, synthetic ranking cannot
    inject signal where it is needed, regardless of the loss implementation.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.copy()
    for col in ("mediapipe_valid", "ordering_ok", "synthetic_valid"):
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].astype(float)
    if "gender" not in df.columns:
        df["gender"] = float("nan")
    summary = (
        df.groupby(["occlusion_bin", "gender"], dropna=False)
        .agg(
            count=("sample_index", "size"),
            mediapipe_valid_rate=("mediapipe_valid", "mean"),
            ordering_ok_rate=("ordering_ok", "mean"),
            synthetic_valid_rate=("synthetic_valid", "mean"),
        )
        .reset_index()
    )
    summary["gender_label"] = summary["gender"].map(
        lambda g: _GENDER_LABELS.get(float(g), str(g)) if pd.notna(g) else "unknown"
    )
    return summary


def _print_coverage_gate(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("[audit] coverage summary is empty (no records).")
        return
    print("\n[audit] MediaPipe coverage by occlusion_bin x gender (Stage 3->4 gate):")
    print(f"  {'bin':<12}{'gender':<8}{'n':>5}{'mp_valid':>10}{'order_ok':>10}{'synth_valid':>12}")
    for _, r in summary.sort_values(["occlusion_bin", "gender"]).iterrows():
        print(
            f"  {str(r['occlusion_bin']):<12}{str(r['gender_label']):<8}"
            f"{int(r['count']):>5}{r['mediapipe_valid_rate']:>10.2f}"
            f"{r['ordering_ok_rate']:>10.2f}{r['synthetic_valid_rate']:>12.2f}"
        )
    high = summary[summary["occlusion_bin"].isin(_HIGH_OCC_BINS)]
    if not high.empty:
        worst = float(high["mediapipe_valid_rate"].min())
        print(
            f"  -> high-occlusion bins {_HIGH_OCC_BINS}: "
            f"min MediaPipe-valid rate = {worst:.2f}. "
            "Low values mean synthetic ranking will under-cover the hard cases."
        )


def _write_failure_grid(
    failures: list[tuple[int, Image.Image, str]],
    output_path: Path,
) -> None:
    if not failures:
        return
    n = len(failures)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (sample_idx, image, reason) in zip(axes_arr, failures):
        ax.imshow(image)
        ax.set_title(f"#{sample_idx} {reason}", fontsize=8, color="red")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    # Resolve image size (training resolution).
    size = args.image_size or int(cfg.augmentation.resize)

    # Build the generator from cfg if synthetic_occlusion.enabled, else from
    # defaults so an audit works regardless of training opt-in state.
    generator = build_generator_from_config(cfg)
    if generator is None:
        generator = SyntheticOcclusionGenerator(
            severity_bands=DEFAULT_SEVERITY_BANDS,
            region_weights=DEFAULT_REGION_WEIGHTS,
            occluder_types=DEFAULT_OCCLUDER_TYPES,
            seed=args.seed,
        )

    # Sample images from the configured training CSV after adding diagnostics
    # metadata and applying optional targeted-audit filters.
    image_col = cfg.data.image_col
    target_col = getattr(cfg.data, "target_col", "FaceOcclusion")
    gender_col = getattr(cfg.data, "gender_col", "gender")
    target_scale = getattr(cfg.data, "target_scale", "auto")
    default_bins = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
    occlusion_bins = list(getattr(cfg.split, "occlusion_bins", default_bins))
    df = _prepare_audit_dataframe(
        pd.read_csv(cfg.data.train_csv),
        image_col=image_col,
        target_col=target_col,
        target_scale=target_scale,
        occlusion_bins=occlusion_bins,
    )
    df = _apply_audit_filters(
        df,
        target_col=target_col,
        gender_col=gender_col,
        target_min=args.target_min,
        target_max=args.target_max,
        database=args.database,
        gender=args.gender,
    )
    if len(df) == 0:
        raise ValueError("No training rows match the requested audit filters.")
    if len(df) < args.num_samples:
        print(
            f"[audit] warning: requested {args.num_samples} samples but only "
            f"{len(df)} rows match filters."
        )
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.num_samples, len(df)), replace=False)
    rows = df.iloc[idx].reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / timestamp
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(cfg.data.image_root)

    n_rows = len(rows)
    # The visual grid scales with the sample size, so a large gate sample would
    # produce an unwieldy figure. --coverage-only computes stats only.
    make_grid = not args.coverage_only
    if make_grid:
        fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

    records = []
    failures: list[tuple[int, Image.Image, str]] = []
    n_valid = 0
    for i, row in rows.iterrows():
        rel = str(row[image_col])
        path = image_root / rel
        with Image.open(path) as im:
            img = im.convert("RGB").resize((size, size), Image.BILINEAR)
        sample_rng = np.random.default_rng(args.seed + int(i))
        pair = generator.generate_pair(img, rng=sample_rng)

        if make_grid:
            # Grid row: original | region overlay | mild | strong
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"#{i} original", fontsize=8)
            axes[i, 1].imshow(img)
            _maybe_overlay_regions(axes[i, 1], pair.region_masks)
            if pair.metadata.get("mediapipe_valid", False):
                axes[i, 1].set_title("MediaPipe regions", fontsize=8)
            else:
                reason = pair.metadata.get("failure_reason") or "unknown"
                axes[i, 1].set_title(f"regions FAILED: {reason}", fontsize=8, color="red")
            if pair.mild is not None:
                axes[i, 2].imshow(pair.mild.image)
                axes[i, 2].set_title(f"mild ρ={pair.mild.severity:.3f}", fontsize=8)
            else:
                axes[i, 2].imshow(img)
                axes[i, 2].set_title("mild FAILED", fontsize=8, color="red")
            if pair.strong is not None:
                axes[i, 3].imshow(pair.strong.image)
                axes[i, 3].set_title(f"strong ρ={pair.strong.severity:.3f}", fontsize=8)
            else:
                axes[i, 3].imshow(img)
                axes[i, 3].set_title("strong FAILED", fontsize=8, color="red")
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])

            # Per-sample PNGs
            img.save(samples_dir / f"sample_{i:03d}_original.png")
            if pair.mild is not None:
                pair.mild.image.save(samples_dir / f"sample_{i:03d}_mild.png")
            if pair.strong is not None:
                pair.strong.image.save(samples_dir / f"sample_{i:03d}_strong.png")
            if not pair.valid:
                failures.append(
                    (int(i), img.copy(), str(pair.metadata.get("failure_reason") or "unknown"))
                )

        if pair.valid:
            n_valid += 1
        records.append(
            _build_record(
                sample_index=int(i),
                row=row,
                image_path=path,
                pair=pair,
                target_col=target_col,
                gender_col=gender_col,
            )
        )

    if make_grid:
        fig.suptitle(
            f"MediaPipe synthetic occlusion audit — {n_valid}/{n_rows} valid pairs "
            f"(generator: {generator.region_provider})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "audit_grid.png", dpi=120)
        plt.close(fig)

    csv_path = out_dir / "synthetic_occlusion_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    group_csv_path = out_dir / "synthetic_occlusion_group_summary.csv"
    _write_group_summary(records, group_csv_path)
    coverage = build_coverage_summary(records)
    coverage_csv_path = out_dir / "coverage_by_bin_gender.csv"
    coverage.to_csv(coverage_csv_path, index=False)
    if make_grid:
        failure_grid_path = out_dir / "failure_grid.png"
        _write_failure_grid(failures, failure_grid_path)
        print(f"[audit] wrote {out_dir}/audit_grid.png")
        if failures:
            print(f"[audit] wrote {failure_grid_path}")
    print(f"[audit] wrote {csv_path}")
    print(f"[audit] wrote {group_csv_path}")
    print(f"[audit] wrote {coverage_csv_path}")
    print(f"[audit] valid pairs: {n_valid}/{n_rows}")
    _print_coverage_gate(coverage)


if __name__ == "__main__":
    main()
