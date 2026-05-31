#!/usr/bin/env python
"""Analyze saved validation predictions without loading a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from face_occlusion.metrics.challenge_metric import challenge_score, weighted_mse

DEFAULT_BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
REQUIRED_COLUMNS = [
    "target",
    "pred_raw",
    "pred_clipped",
    "gender",
    "database",
    "group_id",
    "filename",
]
OPTIONAL_COLUMNS = ["abs_error", "path", "source_subfolder", "face_id", "image_id"]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _warn(message: str) -> None:
    print(f"[analyze] Warning: {message}")


def _bin_labels(bins: list[float]) -> list[str]:
    return [f"{bins[i]:.2f}_{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]


def _auto_detect_split(split_dir: Path) -> Path | None:
    if not split_dir.exists():
        _warn(f"split directory not found, skipping seen/unseen group metrics: {split_dir}")
        return None
    split_files = sorted(split_dir.glob("*.csv"))
    if len(split_files) == 1:
        return split_files[0]
    if not split_files:
        _warn(f"no split CSV found in {split_dir}; skipping seen/unseen group metrics.")
    else:
        _warn(
            f"multiple split CSVs found in {split_dir}; pass --split-csv to choose one. "
            "Skipping seen/unseen group metrics."
        )
    return None


def _resolve_paths(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Path, Path, Path | None]:
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    if experiment_dir:
        pred_path = (
            Path(args.predictions)
            if args.predictions
            else experiment_dir / "predictions" / "val_predictions.csv"
        )
        output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "reports"
        if not pred_path.exists():
            raise FileNotFoundError(f"Validation predictions not found for experiment: {pred_path}")
        split_csv = (
            Path(args.split_csv)
            if args.split_csv
            else _auto_detect_split(experiment_dir / "splits")
        )
    else:
        if not args.predictions or not args.output_dir:
            parser.error(
                "Provide --experiment-dir, or provide both --predictions and --output-dir."
            )
        pred_path = Path(args.predictions)
        output_dir = Path(args.output_dir)
        if not pred_path.exists():
            raise FileNotFoundError(f"Validation predictions not found: {pred_path}")
        split_csv = (
            Path(args.split_csv)
            if args.split_csv
            else _auto_detect_split(pred_path.parent.parent / "splits")
        )

    if split_csv and not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    return pred_path, output_dir, split_csv


def _validate_prediction_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in validation predictions: {missing}")
    missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df.columns]
    if missing_optional:
        _warn(f"optional columns missing: {missing_optional}")


def _add_occlusion_bin(df: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    out = df.copy()
    out["occlusion_bin"] = pd.cut(
        out["target"],
        bins=bins,
        labels=_bin_labels(bins),
        include_lowest=True,
        right=False,
    )
    out.loc[out["target"] == bins[-1], "occlusion_bin"] = _bin_labels(bins)[-1]
    return out


def _add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["target", "pred_raw", "pred_clipped", "gender"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["target", "pred_raw", "pred_clipped", "gender"]].isna().any().any():
        raise ValueError("target, pred_raw, pred_clipped and gender must be numeric.")

    out["error"] = out["pred_clipped"] - out["target"]
    out["squared_error"] = out["error"] ** 2
    out["raw_error"] = out["pred_raw"] - out["target"]
    out["raw_abs_error"] = out["raw_error"].abs()
    computed_abs_error = out["error"].abs()
    if "abs_error" not in out.columns:
        _warn("abs_error column missing; recomputing it from pred_clipped and target.")
        out["abs_error"] = computed_abs_error
    else:
        out["abs_error"] = pd.to_numeric(out["abs_error"], errors="coerce").fillna(
            computed_abs_error
        )
    return out


def _add_face_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "face_id" not in out.columns:
        return out
    out["face_id"] = pd.to_numeric(out["face_id"], errors="coerce").astype("Int64")
    out["face_id_is_zero"] = out["face_id"].eq(0)
    return out


def _add_seen_status(df: pd.DataFrame, split_csv: Path | None) -> pd.DataFrame:
    out = df.copy()
    if split_csv is None:
        return out
    split = pd.read_csv(split_csv)
    if "group_id" not in split.columns or "split" not in split.columns:
        _warn("split CSV lacks group_id/split columns; skipping seen/unseen group metrics.")
        return out
    train_groups = set(split.loc[split["split"] == "train", "group_id"].astype(str))
    out["group_seen_status"] = (
        out["group_id"]
        .astype(str)
        .map(lambda group_id: "seen_in_train" if group_id in train_groups else "unseen_in_train")
    )
    return out


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def _summary_metrics(
    df: pd.DataFrame, pred_path: Path, split_csv: Path | None, top_k: int
) -> dict[str, Any]:
    score = challenge_score(
        df["pred_clipped"],
        df["target"],
        df["gender"],
        female_value=0.0,
        male_value=1.0,
        clip=False,
    )
    clipped_raw = (df["pred_raw"] < 0.0) | (df["pred_raw"] > 1.0)
    high_occ = df["target"] >= 0.4
    extreme_occ = df["target"] >= 0.6
    return {
        **score,
        "rows": int(len(df)),
        "weighted_mse": weighted_mse(df["pred_clipped"], df["target"], clip=False),
        "mae": float(df["abs_error"].mean()),
        "rmse": _rmse(df["error"]),
        "bias": float(df["error"].mean()),
        "mean_target": float(df["target"].mean()),
        "mean_pred_clipped": float(df["pred_clipped"].mean()),
        "mean_pred_raw": float(df["pred_raw"].mean()),
        "target_min": float(df["target"].min()),
        "target_max": float(df["target"].max()),
        "pred_raw_min": float(df["pred_raw"].min()),
        "pred_raw_max": float(df["pred_raw"].max()),
        "pred_clipped_min": float(df["pred_clipped"].min()),
        "pred_clipped_max": float(df["pred_clipped"].max()),
        "pct_pred_raw_below_0": float((df["pred_raw"] < 0.0).mean()),
        "pct_pred_raw_above_1": float((df["pred_raw"] > 1.0).mean()),
        "weighted_mse_raw": weighted_mse(df["pred_raw"], df["target"], clip=False),
        "weighted_mse_clipped": weighted_mse(df["pred_clipped"], df["target"], clip=False),
        "mae_raw": float(df["raw_abs_error"].mean()),
        "mae_clipped": float(df["abs_error"].mean()),
        "high_occlusion_rows": int(high_occ.sum()),
        "extreme_occlusion_rows": int(extreme_occ.sum()),
        "clipped_prediction_rows": int(clipped_raw.sum()),
        "top_k": int(top_k),
        "predictions": str(pred_path),
        "split_csv": str(split_csv) if split_csv else None,
    }


def _metrics_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        # Bias reveals systematic underprediction or overprediction within each group.
        row.update(
            {
                "count": int(len(group)),
                "weighted_mse": weighted_mse(group["pred_clipped"], group["target"], clip=False),
                "mae": float(group["abs_error"].mean()),
                "rmse": _rmse(group["error"]),
                "bias": float(group["error"].mean()),
                "mean_target": float(group["target"].mean()),
                "mean_pred": float(group["pred_clipped"].mean()),
                "mean_pred_raw": float(group["pred_raw"].mean()),
                "target_min": float(group["target"].min()),
                "target_max": float(group["target"].max()),
                "pred_min": float(group["pred_clipped"].min()),
                "pred_max": float(group["pred_clipped"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_grouped_report(
    df: pd.DataFrame,
    group_cols: list[str],
    output_path: Path,
) -> None:
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        _warn(f"skipping {output_path.name}; missing columns: {missing}")
        return
    _metrics_by(df, group_cols).to_csv(output_path, index=False)


def _write_grouped_reports(df: pd.DataFrame, output_dir: Path) -> None:
    reports = [
        (["gender"], "metrics_by_gender.csv"),
        (["occlusion_bin"], "metrics_by_occlusion_bin.csv"),
        (["database"], "metrics_by_database.csv"),
        (["database", "occlusion_bin"], "metrics_by_database_and_bin.csv"),
        (["gender", "occlusion_bin"], "metrics_by_gender_and_bin.csv"),
        (["database", "gender"], "metrics_by_database_and_gender.csv"),
        (["face_id"], "metrics_by_face_id.csv"),
        (["face_id_is_zero"], "metrics_by_face_id_is_zero.csv"),
        (["group_seen_status"], "metrics_by_group_seen_status.csv"),
        (
            ["database", "group_seen_status"],
            "metrics_by_database_and_group_seen_status.csv",
        ),
    ]
    for group_cols, filename in reports:
        _write_grouped_report(df, group_cols, output_dir / filename)


def _write_error_tables(df: pd.DataFrame, output_dir: Path, top_k: int) -> dict[str, pd.DataFrame]:
    tables = {
        "worst_errors": df.sort_values("abs_error", ascending=False).head(top_k),
        "worst_underpredictions": df.sort_values("error", ascending=True).head(top_k),
        "worst_overpredictions": df.sort_values("error", ascending=False).head(top_k),
        "high_occlusion_errors": df[df["target"] >= 0.4]
        .sort_values("abs_error", ascending=False)
        .head(top_k),
        "extreme_occlusion_errors": df[df["target"] >= 0.6]
        .sort_values("abs_error", ascending=False)
        .head(top_k),
        "clipped_predictions": df[(df["pred_raw"] < 0.0) | (df["pred_raw"] > 1.0)]
        .sort_values("raw_abs_error", ascending=False)
        .head(top_k),
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_plots(df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["target"], bins=50, ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel("target")
    _save_plot(plot_dir / "target_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["pred_clipped"], bins=50, ax=ax)
    ax.set_title("Clipped Prediction Distribution")
    ax.set_xlabel("pred_clipped")
    _save_plot(plot_dir / "prediction_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["pred_raw"], bins=50, ax=ax)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Raw Prediction Distribution")
    ax.set_xlabel("pred_raw")
    _save_plot(plot_dir / "pred_raw_distribution.png")

    sample = df.sample(min(len(df), 5000), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["target"], sample["pred_clipped"], s=10, alpha=0.25)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Predicted vs Target")
    ax.set_xlabel("target")
    ax.set_ylabel("pred_clipped")
    _save_plot(plot_dir / "pred_vs_target.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["error"], bins=50, ax=ax)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Error Distribution")
    ax.set_xlabel("pred_clipped - target")
    _save_plot(plot_dir / "error_distribution.png")

    bin_metrics = _metrics_by(df, ["occlusion_bin"])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=bin_metrics, x="occlusion_bin", y="mae", ax=ax)
    ax.set_title("MAE by Occlusion Bin")
    ax.set_xlabel("occlusion bin")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=30)
    _save_plot(plot_dir / "error_by_occlusion_bin.png")

    db_metrics = _metrics_by(df, ["database"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=db_metrics, x="database", y="mae", ax=ax)
    ax.set_title("MAE by Database")
    ax.set_xlabel("database")
    ax.set_ylabel("MAE")
    _save_plot(plot_dir / "error_by_database.png")

    gender_metrics = _metrics_by(df, ["gender"])
    gender_metrics["gender_label"] = gender_metrics["gender"].map(
        {0.0: "female (0)", 1.0: "male (1)"}
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=gender_metrics, x="gender_label", y="mae", ax=ax)
    ax.set_title("MAE by Gender")
    ax.set_xlabel("gender")
    ax.set_ylabel("MAE")
    _save_plot(plot_dir / "error_by_gender.png")


def _resolve_image_path(row: pd.Series, image_root: Path | None) -> Path | None:
    if "path" in row and pd.notna(row["path"]):
        path = Path(str(row["path"]))
        if path.exists():
            return path
    if image_root is not None and "filename" in row and pd.notna(row["filename"]):
        path = image_root / str(row["filename"])
        if path.exists():
            return path
    return None


def _write_image_grid(
    df: pd.DataFrame,
    output_path: Path,
    image_root: Path | None,
    max_images: int,
) -> None:
    images: list[tuple[Image.Image, str]] = []
    for _, row in df.head(max_images).iterrows():
        path = _resolve_image_path(row, image_root)
        if path is None:
            continue
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        title = (
            f"t={row['target']:.2f} p={row['pred_clipped']:.2f} e={row['abs_error']:.2f}\n"
            f"g={row['gender']:.0f} {row['database']}"
        )
        images.append((image, title))

    if not images:
        _warn(f"no readable images for grid: {output_path.name}")
        return

    cols = 4
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, (image, title) in zip(axes_array, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    _save_plot(output_path)


def _write_image_grids(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    image_root: Path | None,
    top_k: int,
) -> None:
    grid_k = min(top_k, 16)
    plot_dir = output_dir / "plots"
    for name in [
        "worst_errors",
        "worst_underpredictions",
        "worst_overpredictions",
        "high_occlusion_errors",
    ]:
        _write_image_grid(
            tables[name],
            plot_dir / f"{name}_grid.png",
            image_root=image_root,
            max_images=grid_k,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--bins", nargs="+", type=float, default=DEFAULT_BINS)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--save-image-grids", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    args.predictions, args.output_dir, args.split_csv = _resolve_paths(args, parser)
    if len(args.bins) < 2:
        parser.error("--bins must contain at least two values.")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    pred_path: Path = args.predictions
    output_dir: Path = args.output_dir
    split_csv: Path | None = args.split_csv
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    _validate_prediction_columns(df)
    df = _add_error_columns(df)
    df = _add_occlusion_bin(df, list(args.bins))
    df = _add_face_id_columns(df)
    df = _add_seen_status(df, split_csv)

    summary = _summary_metrics(df, pred_path, split_csv, top_k=args.top_k)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(_json_ready(summary), indent=2),
        encoding="utf-8",
    )

    _write_grouped_reports(df, output_dir)
    tables = _write_error_tables(df, output_dir, top_k=args.top_k)
    _write_plots(df, output_dir)

    if args.save_image_grids:
        image_root = Path(args.image_root) if args.image_root else None
        _write_image_grids(tables, output_dir, image_root=image_root, top_k=args.top_k)

    print(f"[analyze] Predictions: {pred_path}")
    print(f"[analyze] Reports:     {output_dir}")
    if split_csv:
        print(f"[analyze] Split CSV:   {split_csv}")


if __name__ == "__main__":
    main()
