#!/usr/bin/env python
"""Analyze saved validation predictions \u2014 complete post-analysis report generator.

Usage (recommended):
    python scripts/analyze_val_predictions.py \\
        --experiment-dir outputs/experiments/<run_id>

This generates:
    reports/report.html          \u2014 standalone HTML report
    reports/summary_metrics.json \u2014 key metrics as JSON
    reports/tables/              \u2014 grouped metrics and error tables (CSV)
    reports/plots/               \u2014 ordered diagnostic plots (PNG)
    reports/samples/             \u2014 image grids of difficult examples (PNG)

Options:
    --no-image-grids             Disable image grid generation
    --image-root PATH            Override image root directory
                                 (default: data/crops/Crop_224_5fp_100K)
    --top-k INT                  Rows in error tables (default: 100)
    --grid-k INT                 Images per grid (default: 16)
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
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
    # Challenge metric weights — used throughout grouped analyses
    out["weight"] = 1.0 / 30.0 + out["target"]
    out["weighted_sq_error"] = out["weight"] * out["squared_error"]
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
        .map(lambda gid: "seen_in_train" if gid in train_groups else "unseen_in_train")
    )
    return out


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def _compute_summary_metrics(
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
    global_weight_sum = float(df["weight"].sum())
    global_weighted_error_sum = float(df["weighted_sq_error"].sum())
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {col: key for col, key in zip(group_cols, keys)}
        weight_sum = float(group["weight"].sum())
        weighted_sq_sum = float(group["weighted_sq_error"].sum())
        row.update(
            {
                "count": int(len(group)),
                "weight_sum": weight_sum,
                "weight_ratio": (
                    weight_sum / global_weight_sum if global_weight_sum > 0 else float("nan")
                ),
                "weighted_error_sum": weighted_sq_sum,
                "weighted_error_contribution_ratio": (
                    weighted_sq_sum / global_weighted_error_sum
                    if global_weighted_error_sum > 0
                    else float("nan")
                ),
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


def _write_tables(df: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        (["gender"], "metrics_by_gender.csv"),
        (["occlusion_bin"], "metrics_by_occlusion_bin.csv"),
        (["database"], "metrics_by_database.csv"),
        (["database", "occlusion_bin"], "metrics_by_database_and_bin.csv"),
        (["gender", "occlusion_bin"], "metrics_by_gender_and_bin.csv"),
        (["database", "gender"], "metrics_by_database_and_gender.csv"),
        (["face_id_is_zero"], "metrics_by_face_id.csv"),
        (["group_seen_status"], "metrics_by_group_seen_status.csv"),
        (["database", "group_seen_status"], "metrics_by_database_and_group_seen_status.csv"),
    ]
    for group_cols, filename in reports:
        _write_grouped_report(df, group_cols, tables_dir / filename)


def _write_error_tables(df: pd.DataFrame, tables_dir: Path, top_k: int) -> dict[str, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "worst_errors": df.sort_values("abs_error", ascending=False).head(top_k),
        "worst_underpredictions": df.sort_values("error", ascending=True).head(top_k),
        "worst_overpredictions": df.sort_values("error", ascending=False).head(top_k),
        "high_occlusion_errors": (
            df[df["target"] >= 0.4].sort_values("abs_error", ascending=False).head(top_k)
        ),
        "extreme_occlusion_errors": (
            df[df["target"] >= 0.6].sort_values("abs_error", ascending=False).head(top_k)
        ),
        "clipped_predictions": (
            df[(df["pred_raw"] < 0.0) | (df["pred_raw"] > 1.0)]
            .sort_values("raw_abs_error", ascending=False)
            .head(top_k)
        ),
    }
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    return tables


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _annotate_bars(
    ax: plt.Axes,
    values: list[float],
    counts: list[int] | None = None,
    fmt: str = "{:.4f}",
) -> None:
    patches = ax.patches[: len(values)]
    for i, (bar, val) in enumerate(zip(patches, values)):
        y = bar.get_height()
        try:
            if math.isnan(float(val)):
                continue
        except (TypeError, ValueError):
            continue
        text = fmt.format(val)
        if counts is not None and i < len(counts):
            text += f"\nn={counts[i]}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + max(abs(y) * 0.01, 1e-5),
            text,
            ha="center",
            va="bottom",
            fontsize=7,
        )


# ─── Individual plots ─────────────────────────────────────────────────────────


def _plot_challenge_score_decomposition(summary: dict[str, Any], path: Path) -> None:
    err_f = float(summary.get("err_female") or float("nan"))
    err_m = float(summary.get("err_male") or float("nan"))
    no_nan = not (math.isnan(err_f) or math.isnan(err_m))
    mean_sub = (err_f + err_m) / 2 if no_nan else float("nan")
    gap = abs(err_f - err_m) if no_nan else float("nan")
    final = float(summary.get("score") or float("nan"))

    labels = ["err_female", "err_male", "mean_subgroup", "gender_gap", "final_score"]
    values = [err_f, err_m, mean_sub, gap, final]
    colors = ["#e07b91", "#5585b5", "#6acc65", "#f6a623", "#c44e52"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, val in zip(bars, values):
        if math.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(bar.get_height() * 0.01, 1e-6),
            f"{val:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title("Challenge Score Decomposition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted MSE")
    ax.tick_params(axis="x", rotation=15)
    fig.text(
        0.5,
        0.01,
        "Score = (err_female + err_male) / 2 + |err_female \u2212 err_male|",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555",
    )
    _save_plot(path)


def _plot_distributions(df: pd.DataFrame, path: Path, log_scale: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["target"], bins=50, ax=axes[0], color="#4878cf")
    axes[0].set_title("Target Distribution")
    axes[0].set_xlabel("target")
    sns.histplot(df["pred_clipped"], bins=50, ax=axes[1], color="#6acc65")
    axes[1].set_title("Clipped Prediction Distribution")
    axes[1].set_xlabel("pred_clipped")
    if log_scale:
        for ax in axes:
            ax.set_yscale("log")
        fig.suptitle("Target and Prediction Distributions \u2014 Log Count Scale", fontsize=11)
    else:
        fig.suptitle("Target and Prediction Distributions", fontsize=11)
    _save_plot(path)


def _plot_predicted_vs_target(df: pd.DataFrame, path: Path) -> None:
    sample = df.sample(min(len(df), 5000), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sample["target"], sample["pred_clipped"], s=10, alpha=0.25, color="#4878cf")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Predicted vs Target")
    ax.set_xlabel("target")
    ax.set_ylabel("pred_clipped")
    ax.legend(fontsize=8)
    _save_plot(path)


def _plot_calibration_by_occlusion_bin(df: pd.DataFrame, path: Path) -> None:
    bm = _metrics_by(df, ["occlusion_bin"])
    labels = bm["occlusion_bin"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, bm["mean_target"], width, label="mean_target", color="#4878cf")
    bars2 = ax.bar(x + width / 2, bm["mean_pred"], width, label="mean_pred", color="#6acc65")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Calibration by Occlusion Bin \u2014 Mean Target vs Mean Prediction")
    ax.set_ylabel("mean value")
    ax.legend()
    for bar, n in zip(bars2, bm["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _save_plot(path)


def _plot_weighted_error_contribution(df: pd.DataFrame, path: Path) -> None:
    bm = _metrics_by(df, ["occlusion_bin"])
    labels = bm["occlusion_bin"].astype(str).tolist()
    ratios = bm["weighted_error_contribution_ratio"].tolist()
    counts = bm["count"].tolist()
    wmse_vals = bm["weighted_mse"].tolist()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, ratios, color="#c44e52")
    ax.set_title("Weighted Error Contribution by Occlusion Bin")
    ax.set_xlabel("occlusion bin")
    ax.set_ylabel("fraction of total weighted error")
    ax.tick_params(axis="x", rotation=30)
    for bar, ratio, n, wmse in zip(ax.patches, ratios, counts, wmse_vals):
        y = bar.get_height()
        try:
            if math.isnan(float(ratio)):
                continue
        except (TypeError, ValueError):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.005,
            f"{ratio:.3f}\nWMSE={wmse:.4f}\nn={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    _save_plot(path)


def _plot_bias_by_occlusion_bin(df: pd.DataFrame, path: Path) -> None:
    bm = _metrics_by(df, ["occlusion_bin"])
    labels = bm["occlusion_bin"].astype(str).tolist()
    biases = bm["bias"].tolist()
    colors = ["#c44e52" if b > 0 else "#4878cf" for b in biases]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, biases, color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Bias by Occlusion Bin  (positive = overprediction, negative = underprediction)")
    ax.set_xlabel("occlusion bin")
    ax.set_ylabel("mean(pred \u2212 target)")
    ax.tick_params(axis="x", rotation=30)
    _annotate_bars(ax, biases, bm["count"].tolist(), fmt="{:.4f}")
    _save_plot(path)


def _plot_weighted_error_by_gender(df: pd.DataFrame, path: Path) -> None:
    gm = _metrics_by(df, ["gender"])
    gm["gender_label"] = (
        gm["gender"].map({0.0: "female (0)", 1.0: "male (1)"}).fillna(gm["gender"].astype(str))
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(gm["gender_label"], gm["weighted_mse"], color=["#e07b91", "#5585b5"])
    ax.set_title("Weighted MSE by Gender")
    ax.set_xlabel("gender")
    ax.set_ylabel("weighted MSE")
    _annotate_bars(ax, gm["weighted_mse"].tolist(), gm["count"].tolist())
    _save_plot(path)


def _plot_bias_by_gender(df: pd.DataFrame, path: Path) -> None:
    gm = _metrics_by(df, ["gender"])
    gm["gender_label"] = (
        gm["gender"].map({0.0: "female (0)", 1.0: "male (1)"}).fillna(gm["gender"].astype(str))
    )
    biases = gm["bias"].tolist()
    colors = ["#c44e52" if b > 0 else "#4878cf" for b in biases]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(gm["gender_label"], biases, color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Bias by Gender\n(positive = overprediction, negative = underprediction)")
    ax.set_xlabel("gender")
    ax.set_ylabel("mean(pred \u2212 target)")
    _annotate_bars(ax, biases, gm["count"].tolist())
    _save_plot(path)


def _plot_weighted_error_by_database(df: pd.DataFrame, path: Path) -> None:
    dm = _metrics_by(df, ["database"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dm["database"].astype(str), dm["weighted_mse"], color="#4878cf")
    ax.set_title("Weighted MSE by Database")
    ax.set_xlabel("database")
    ax.set_ylabel("weighted MSE")
    _annotate_bars(ax, dm["weighted_mse"].tolist(), dm["count"].tolist())
    _save_plot(path)


def _plot_bias_by_database(df: pd.DataFrame, path: Path) -> None:
    dm = _metrics_by(df, ["database"])
    biases = dm["bias"].tolist()
    colors = ["#c44e52" if b > 0 else "#4878cf" for b in biases]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dm["database"].astype(str), biases, color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Bias by Database\n(positive = overprediction, negative = underprediction)")
    ax.set_xlabel("database")
    ax.set_ylabel("mean(pred \u2212 target)")
    _annotate_bars(ax, biases, dm["count"].tolist())
    _save_plot(path)


def _plot_bias_by_gender_and_occlusion_bin(df: pd.DataFrame, path: Path) -> None:
    if "occlusion_bin" not in df.columns:
        return
    gbin = _metrics_by(df, ["gender", "occlusion_bin"])
    gbin["gender_label"] = (
        gbin["gender"].map({0.0: "female", 1.0: "male"}).fillna(gbin["gender"].astype(str))
    )
    bins_unique = gbin["occlusion_bin"].unique().tolist()
    genders_unique = sorted(gbin["gender_label"].unique().tolist())
    x = np.arange(len(bins_unique))
    n_genders = len(genders_unique)
    width = 0.7 / max(n_genders, 1)
    gender_colors = {"female": "#e07b91", "male": "#5585b5"}
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, gl in enumerate(genders_unique):
        subset = gbin[gbin["gender_label"] == gl].copy()
        subset = subset.set_index("occlusion_bin").reindex(bins_unique)
        offset = (i - n_genders / 2 + 0.5) * width
        ax.bar(
            x + offset,
            subset["bias"].values,
            width,
            label=gl,
            color=gender_colors.get(gl, f"C{i}"),
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bins_unique], rotation=30, ha="right")
    ax.set_title("Bias by Gender and Occlusion Bin\n(positive = overprediction)")
    ax.set_xlabel("occlusion bin")
    ax.set_ylabel("mean(pred \u2212 target)")
    ax.legend()
    _save_plot(path)


def _plot_weighted_error_by_group_seen(df: pd.DataFrame, path: Path) -> None:
    sm = _metrics_by(df, ["group_seen_status"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        sm["group_seen_status"].astype(str),
        sm["weighted_mse"],
        color=["#4878cf", "#6acc65"],
    )
    ax.set_title("Weighted MSE by Group Seen Status")
    ax.set_xlabel("seen status")
    ax.set_ylabel("weighted MSE")
    _annotate_bars(ax, sm["weighted_mse"].tolist(), sm["count"].tolist())
    _save_plot(path)


def _plot_raw_prediction_clipping(df: pd.DataFrame, path: Path) -> None:
    pct_below = float((df["pred_raw"] < 0.0).mean()) * 100
    pct_above = float((df["pred_raw"] > 1.0).mean()) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["pred_raw"], bins=60, ax=ax, color="#4878cf")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="clip bounds")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(
        f"Raw Prediction Distribution\n{pct_below:.2f}% below 0  |  {pct_above:.2f}% above 1"
    )
    ax.set_xlabel("pred_raw")
    ax.legend()
    _save_plot(path)


def _plot_error_distribution(df: pd.DataFrame, path: Path) -> None:
    bias = float(df["error"].mean())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["error"], bins=60, ax=ax, color="#4878cf")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(bias, color="red", linestyle="--", linewidth=1.2, label=f"bias = {bias:.4f}")
    ax.set_title("Error Distribution  (pred_clipped \u2212 target)")
    ax.set_xlabel("error")
    ax.legend(fontsize=9)
    _save_plot(path)


def _write_plots(df: pd.DataFrame, summary: dict[str, Any], plots_dir: Path) -> list[Path]:
    """Generate all diagnostic plots. Returns the list of created paths."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    created: list[Path] = []

    def _emit(filename: str, fn, *a, **kw) -> None:  # type: ignore[no-untyped-def]
        p = plots_dir / filename
        try:
            fn(*a, p, **kw)
            created.append(p)
        except Exception as exc:
            _warn(f"could not generate {filename}: {exc}")
            plt.close("all")

    _emit("01_challenge_score_decomposition.png", _plot_challenge_score_decomposition, summary)
    _emit("02_target_prediction_distribution.png", _plot_distributions, df)
    _emit("03_target_prediction_distribution_log.png", _plot_distributions, df, log_scale=True)
    _emit("04_predicted_vs_target.png", _plot_predicted_vs_target, df)
    _emit("05_calibration_by_occlusion_bin.png", _plot_calibration_by_occlusion_bin, df)
    _emit(
        "06_weighted_error_contribution_by_occlusion_bin.png",
        _plot_weighted_error_contribution,
        df,
    )
    _emit("07_bias_by_occlusion_bin.png", _plot_bias_by_occlusion_bin, df)
    _emit("08_weighted_error_by_gender.png", _plot_weighted_error_by_gender, df)
    _emit("09_bias_by_gender.png", _plot_bias_by_gender, df)
    _emit("10_weighted_error_by_database.png", _plot_weighted_error_by_database, df)
    _emit("11_bias_by_database.png", _plot_bias_by_database, df)
    _emit("12_bias_by_gender_and_occlusion_bin.png", _plot_bias_by_gender_and_occlusion_bin, df)
    if "group_seen_status" in df.columns:
        _emit(
            "13_weighted_error_by_group_seen_status.png",
            _plot_weighted_error_by_group_seen,
            df,
        )
    _emit("14_raw_prediction_clipping.png", _plot_raw_prediction_clipping, df)
    _emit("15_error_distribution.png", _plot_error_distribution, df)

    return created


def _resolve_image_path(row: pd.Series, image_root: Path | None) -> Path | None:
    if "path" in row.index and pd.notna(row["path"]):
        p = Path(str(row["path"]))
        if p.exists():
            return p
    if image_root is not None and "filename" in row.index and pd.notna(row["filename"]):
        p = image_root / str(row["filename"])
        if p.exists():
            return p
    return None


def _write_image_grid(
    df: pd.DataFrame,
    output_path: Path,
    image_root: Path | None,
    max_images: int,
) -> None:
    images: list[tuple[Image.Image, str]] = []
    for _, row in df.head(max_images).iterrows():
        img_path = _resolve_image_path(row, image_root)
        if img_path is None:
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        gender_val = row.get("gender", float("nan"))
        gender_str = "F" if gender_val == 0.0 else ("M" if gender_val == 1.0 else "?")
        db_str = str(row.get("database", "?"))
        seen_str = ""
        if "group_seen_status" in row.index and pd.notna(row.get("group_seen_status")):
            seen_str = " S" if str(row["group_seen_status"]) == "seen_in_train" else " U"
        title = (
            f"t={row['target']:.2f} p={row['pred_clipped']:.2f}\n"
            f"e={row['abs_error']:.2f} {gender_str} {db_str}{seen_str}"
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
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    _save_plot(output_path)


def _write_image_grids(
    tables: dict[str, pd.DataFrame],
    samples_dir: Path,
    image_root: Path | None,
    grid_k: int,
) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "worst_errors",
        "worst_underpredictions",
        "worst_overpredictions",
        "high_occlusion_errors",
        "extreme_occlusion_errors",
    ]:
        if name not in tables:
            continue
        _write_image_grid(
            tables[name],
            samples_dir / f"{name}_grid.png",
            image_root=image_root,
            max_images=grid_k,
        )


# ─── HTML report ──────────────────────────────────────────────────────────────


def _generate_auto_comments(summary: dict[str, Any], df: pd.DataFrame) -> list[str]:
    comments: list[str] = []

    bias = float(summary.get("bias") or 0.0)
    if abs(bias) < 0.005:
        comments.append(
            "Global bias is near zero \u2014 the model appears well-calibrated on average."
        )
    elif bias < 0:
        comments.append(
            f"Global bias is {bias:.4f}: the model tends to <strong>underpredict</strong> "
            "occlusion on average. This may indicate underestimation of highly occluded "
            "samples. Inspect the calibration plots to confirm."
        )
    else:
        comments.append(
            f"Global bias is {bias:.4f}: the model tends to <strong>overpredict</strong> "
            "occlusion on average."
        )

    pct_below = float(summary.get("pct_pred_raw_below_0") or 0.0)
    pct_above = float(summary.get("pct_pred_raw_above_1") or 0.0)
    if pct_below + pct_above < 0.01:
        comments.append(
            "Raw predictions rarely fall outside [0, 1] \u2014 clipping is not a major concern."
        )
    else:
        comments.append(
            f"{(pct_below + pct_above) * 100:.2f}% of raw predictions fall outside [0, 1]. "
            "Clipping may affect the metric; inspect the raw prediction plot."
        )

    err_f = summary.get("err_female", float("nan")) or float("nan")
    err_m = summary.get("err_male", float("nan")) or float("nan")
    try:
        err_f_f = float(err_f)
        err_m_f = float(err_m)
        no_nan = not (math.isnan(err_f_f) or math.isnan(err_m_f))
    except (TypeError, ValueError):
        no_nan = False
    if no_nan:
        gap = abs(err_f_f - err_m_f)
        mean_err = (err_f_f + err_m_f) / 2
        if mean_err > 0 and gap > mean_err * 0.2:
            worse = "female" if err_f_f > err_m_f else "male"
            comments.append(
                f"The gender gap ({gap:.5f}) is large relative to mean subgroup error "
                f"({mean_err:.5f}). The <strong>{worse}</strong> subgroup has higher error "
                "\u2014 this penalty is significant. Consider targeted data augmentation or "
                "reweighting."
            )
        else:
            comments.append(
                f"Gender gap ({gap:.5f}) is modest relative to the mean subgroup error "
                f"({mean_err:.5f}). Gender balance looks acceptable."
            )

    if "occlusion_bin" in df.columns:
        bm = _metrics_by(df, ["occlusion_bin"])
        high_mask = bm["mean_target"] >= 0.4
        low_mask = bm["mean_target"] < 0.2
        if high_mask.any() and low_mask.any():
            high_wmse = float(bm.loc[high_mask, "weighted_mse"].mean())
            low_wmse = float(bm.loc[low_mask, "weighted_mse"].mean())
            if not (math.isnan(high_wmse) or math.isnan(low_wmse)) and low_wmse > 0:
                if high_wmse > low_wmse * 2:
                    comments.append(
                        "High-occlusion bins have substantially larger weighted MSE than "
                        "low-occlusion bins. This suggests the model struggles most with "
                        "highly occluded faces \u2014 a likely target area for improvement."
                    )

    return comments


def _find_metrics_csv(experiment_dir: Path | None) -> Path | None:
    """Find the Lightning CSV logger metrics file for an experiment directory."""
    if experiment_dir is None:
        return None
    # Standard layout: logs/csv_logs/version_*/metrics.csv — take the latest version.
    candidates = sorted(experiment_dir.glob("logs/csv_logs/version_*/metrics.csv"))
    if candidates:
        return candidates[-1]
    # Fallback: any metrics.csv nested under logs/.
    candidates = sorted(experiment_dir.glob("logs/**/metrics.csv"))
    return candidates[-1] if candidates else None


def _read_training_metrics(metrics_path: Path) -> pd.DataFrame | None:
    """Parse Lightning CSV logger output into a per-epoch DataFrame.

    Lightning logs one row per step and leaves most columns as NaN for steps
    that don't update them.  This function coalesces multiple rows per epoch
    into a single row by taking the first non-NaN value per column.
    """
    df = pd.read_csv(metrics_path)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    if df.empty:
        return None
    df["epoch"] = df["epoch"].astype(int)
    result = (
        df.groupby("epoch")
        .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else float("nan"))
        .reset_index()
    )
    return result if not result.empty else None


def _write_training_dynamics_plots(metrics_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Generate per-epoch training-dynamics plots. Returns list of created paths."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    created: list[Path] = []

    epochs = metrics_df["epoch"].values

    def _get(col: str) -> np.ndarray | None:
        """Return column as float array, or None if missing / all-NaN."""
        if col not in metrics_df.columns:
            return None
        v = pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=float)
        return v if not np.all(np.isnan(v)) else None

    def _finish(fig: plt.Figure, path: Path, had_data: bool) -> None:
        if had_data:
            _save_plot(path)
            created.append(path)
        else:
            plt.close(fig)

    # ── Plot 20: global score + loss ─────────────────────────────────────────
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    had = False
    for col, label, ax, color in [
        ("val/score", "Val Score", ax0, "steelblue"),
        ("val/err_female", "Err Female", ax0, "coral"),
        ("val/err_male", "Err Male", ax0, "orchid"),
        ("val/loss", "Val Loss", ax1, "seagreen"),
        ("train/loss", "Train Loss", ax1, "tomato"),
        ("val/mae", "Val MAE", ax1, "goldenrod"),
    ]:
        v = _get(col)
        if v is not None:
            ax.plot(epochs, v, marker="o", markersize=3, label=label, color=color)
            had = True
    ax0.set_ylabel("Score / Error")
    ax0.set_title("Validation Score and Gender Errors by Epoch")
    ax0.legend(fontsize=8)
    ax1.set_ylabel("Loss / MAE")
    ax1.set_title("Training and Validation Loss by Epoch")
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, plots_dir / "20_training_global_metrics.png", had)

    # ── Plot 21: weighted MSE by occlusion bin ───────────────────────────────
    bin_err_cols = sorted(
        c for c in metrics_df.columns if c.startswith("val/bin_") and c.endswith("_err")
    )
    if bin_err_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        had = False
        for col in bin_err_cols:
            v = _get(col)
            if v is not None:
                label = col.removeprefix("val/bin_").removesuffix("_err")
                ax.plot(epochs, v, marker="o", markersize=3, label=label)
                had = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted MSE")
        ax.set_title("Weighted MSE by Occlusion Bin (per epoch)")
        ax.legend(title="Bin", fontsize=8, ncol=2)
        fig.tight_layout()
        _finish(fig, plots_dir / "21_training_weighted_mse_by_occlusion_bin.png", had)

    # ── Plot 22: bias by occlusion bin ───────────────────────────────────────
    bin_bias_cols = sorted(
        c for c in metrics_df.columns if c.startswith("val/bin_") and c.endswith("_bias")
    )
    if bin_bias_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        had = False
        for col in bin_bias_cols:
            v = _get(col)
            if v is not None:
                label = col.removeprefix("val/bin_").removesuffix("_bias")
                ax.plot(epochs, v, marker="o", markersize=3, label=label)
                had = True
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Bias (pred - target)")
        ax.set_title("Prediction Bias by Occlusion Bin (per epoch)")
        ax.legend(title="Bin", fontsize=8, ncol=2)
        fig.tight_layout()
        _finish(fig, plots_dir / "22_training_bias_by_occlusion_bin.png", had)
    else:
        _warn(
            "Bias-by-bin dynamics skipped: val/bin_*_bias not found in metrics CSV. "
            "These are logged by the updated LitModule."
        )

    # ── Plot 23: weighted MSE by gender ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    had = False
    for col, label in (("val/err_female", "Female"), ("val/err_male", "Male")):
        v = _get(col)
        if v is not None:
            ax.plot(epochs, v, marker="o", markersize=3, label=label)
            had = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE")
    ax.set_title("Weighted MSE by Gender (per epoch)")
    ax.legend()
    fig.tight_layout()
    _finish(fig, plots_dir / "23_training_weighted_mse_by_gender.png", had)

    # ── Plot 24: bias by gender ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    had = False
    for col, label in (("val/female_bias", "Female"), ("val/male_bias", "Male")):
        v = _get(col)
        if v is not None:
            ax.plot(epochs, v, marker="o", markersize=3, label=label)
            had = True
    if had:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Bias (pred - target)")
        ax.set_title("Prediction Bias by Gender (per epoch)")
        ax.legend()
        fig.tight_layout()
    _finish(fig, plots_dir / "24_training_bias_by_gender.png", had)
    if not had:
        _warn(
            "Bias-by-gender dynamics skipped: val/female_bias/val/male_bias not found. "
            "These are logged by the updated LitModule."
        )

    # ── Plot 25: weighted MSE by database ────────────────────────────────────
    db_err_cols = sorted(
        c for c in metrics_df.columns if c.startswith("val/database/") and c.endswith("_err")
    )
    if db_err_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        had = False
        for col in db_err_cols:
            v = _get(col)
            if v is not None:
                label = col.removeprefix("val/database/").removesuffix("_err")
                ax.plot(epochs, v, marker="o", markersize=3, label=label)
                had = True
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted MSE")
        ax.set_title("Weighted MSE by Database (per epoch)")
        ax.legend(title="Database")
        fig.tight_layout()
        _finish(fig, plots_dir / "25_training_weighted_mse_by_database.png", had)

    return created


def _write_html_report(
    output_dir: Path,
    summary: dict[str, Any],
    experiment_dir: Path | None,
    pred_path: Path,
    split_csv: Path | None,
    plots_dir: Path,
    samples_dir: Path,
    tables_dir: Path,
    df: pd.DataFrame,
    dynamics_plots: list[Path] | None = None,
) -> Path:
    report_path = output_dir / "report.html"

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(output_dir)).replace("\\", "/")
        except ValueError:
            return str(p)

    def _img(p: Path, caption: str = "", width: str = "100%") -> str:
        rp = _rel(p)
        if not p.exists():
            return f"<p><em>{caption} \u2014 not generated</em></p>"
        cap_html = (
            f"<figcaption style='font-size:.85em;color:#666;'>{caption}</figcaption>"
            if caption
            else ""
        )
        return (
            f"<figure style='margin:0 0 1em 0;'>"
            f"<img src='{rp}' style='max-width:{width};' alt='{caption}'/>"
            f"{cap_html}</figure>"
        )

    def _tlink(p: Path) -> str:
        rp = _rel(p)
        if not p.exists():
            return f"<li><em>{p.name} \u2014 not generated</em></li>"
        return f"<li><a href='{rp}'>{p.name}</a></li>"

    def _fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        try:
            fv = float(val)
            if math.isnan(fv):
                return "N/A"
            return f"{fv:.6f}" if abs(fv) < 10 else f"{fv:.4f}"
        except (TypeError, ValueError):
            return str(val)

    score_fields = [
        ("score", "Final Score"),
        ("err_female", "Err Female"),
        ("err_male", "Err Male"),
        ("gender_gap", "Gender Gap"),
        ("err_mean", "Mean Subgroup Error"),
    ]
    global_fields = [
        ("weighted_mse", "Weighted MSE"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("bias", "Bias"),
        ("mean_target", "Mean Target"),
        ("mean_pred_clipped", "Mean Pred (clipped)"),
        ("rows", "Total rows"),
    ]
    clipping_fields = [
        ("pct_pred_raw_below_0", "% Raw preds < 0"),
        ("pct_pred_raw_above_1", "% Raw preds > 1"),
        ("clipped_prediction_rows", "Rows clipped"),
        ("high_occlusion_rows", "High-occlusion rows (target \u2265 0.4)"),
        ("extreme_occlusion_rows", "Extreme-occlusion rows (target \u2265 0.6)"),
    ]

    ts = "border-collapse:collapse;width:100%;max-width:520px;margin-bottom:1em;"
    ths = "border:1px solid #ccc;padding:.4em .8em;background:#ebf8ff;text-align:left;"
    tds = "border:1px solid #ccc;padding:.4em .8em;text-align:left;"

    def _table(fields: list[tuple[str, str]]) -> str:
        body = "".join(
            f"<tr><td style='{tds}'>{label}</td>"
            f"<td style='{tds}'><strong>{_fmt(summary.get(key))}</strong></td></tr>"
            for key, label in fields
        )
        return (
            f"<table style='{ts}'><thead>"
            f"<tr><th style='{ths}'>Metric</th><th style='{ths}'>Value</th></tr>"
            f"</thead><tbody>{body}</tbody></table>"
        )

    auto_comments = _generate_auto_comments(summary, df)
    comments_html = "\n".join(f"<li>{c}</li>" for c in auto_comments)

    sample_grids = sorted(samples_dir.glob("*.png")) if samples_dir.exists() else []
    samples_html = (
        "\n".join(_img(p, p.stem.replace("_", " ").title(), "80%") for p in sample_grids)
        or "<p><em>Image grids not generated or images not found.</em></p>"
    )

    table_files = sorted(tables_dir.glob("*.csv")) if tables_dir.exists() else []
    tables_html = (
        "\n".join(_tlink(p) for p in table_files) or "<li><em>No tables generated.</em></li>"
    )

    exp_str = str(experiment_dir) if experiment_dir else "N/A"
    split_str = str(split_csv) if split_csv else "N/A"
    score_val = _fmt(summary.get("score"))

    # Training dynamics section -----------------------------------------------
    dyn_plot_names = [
        ("20_training_global_metrics.png", "Global metrics over epochs"),
        ("21_training_weighted_mse_by_occlusion_bin.png", "Weighted MSE by occlusion bin"),
        ("22_training_bias_by_occlusion_bin.png", "Bias by occlusion bin"),
        ("23_training_weighted_mse_by_gender.png", "Weighted MSE by gender"),
        ("24_training_bias_by_gender.png", "Bias by gender"),
        ("25_training_weighted_mse_by_database.png", "Weighted MSE by database"),
    ]
    if dynamics_plots:
        dyn_imgs = "\n".join(_img(plots_dir / fname, cap) for fname, cap in dyn_plot_names)
        dynamics_section = (
            f"<h2>12. Training Dynamics</h2>\n<div class='plot-grid'>{dyn_imgs}</div>"
        )
    else:
        dynamics_section = (
            "<h2>12. Training Dynamics</h2>"
            "<p><em>Training metrics not found. Run with "
            "<code>--experiment-dir</code> pointing to an experiment folder that "
            "contains <code>logs/csv_logs/version_*/metrics.csv</code>.</em></p>"
        )

    html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8"/>
          <meta name="viewport" content="width=device-width,initial-scale=1"/>
          <title>Face Occlusion \u2014 Validation Analysis</title>
          <style>
            body {{
              font-family: system-ui, -apple-system, sans-serif;
              max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222;
            }}
            h1 {{ color: #1a3a5c; }}
            h2 {{
              color: #2c5282; border-bottom: 2px solid #bee3f8;
              padding-bottom: .3em; margin-top: 2.5em;
            }}
            .score-box {{
              background: #ebf8ff; border: 2px solid #3182ce; border-radius: .5em;
              padding: 1em 1.5em; display: inline-block; margin-bottom: 1.5em;
            }}
            .score-value {{ font-size: 2.2em; font-weight: bold; color: #2b6cb0; }}
            .comments {{
              background: #fffff0; border-left: 4px solid #f6c90e;
              padding: .8em 1em; margin: 1em 0;
            }}
            .plot-grid {{
              display: grid;
              grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
              gap: 1em;
            }}
            figure {{
              border: 1px solid #e2e8f0; border-radius: .4em;
              padding: .5em; background: #fff; margin: 0;
            }}
            a {{ color: #2b6cb0; }}
            ul {{ padding-left: 1.2em; }}
            .meta td {{ padding: .25em .7em; font-size: .9em; color: #444; }}
            .meta td:first-child {{ font-weight: bold; }}
          </style>
        </head>
        <body>

        <h1>Face Occlusion \u2014 Validation Post-Analysis Report</h1>

        <h2>1. Overview</h2>
        <table class="meta">
          <tr><td>Experiment</td><td>{exp_str}</td></tr>
          <tr><td>Predictions</td><td>{pred_path}</td></tr>
          <tr><td>Split CSV</td><td>{split_str}</td></tr>
          <tr><td>Report</td><td>{report_path}</td></tr>
          <tr><td>Total rows</td><td>{summary.get("rows", "N/A")}</td></tr>
        </table>

        <h2>2. Challenge Score Decomposition</h2>
        <div class="score-box">
          <div style="font-size:.9em;color:#555;">Final Score</div>
          <div class="score-value">{score_val}</div>
          <div style="font-size:.82em;color:#666;margin-top:.3em;">
            Score = (Err_Female + Err_Male) / 2 + |Err_Female &minus; Err_Male|
          </div>
        </div>
        {_table(score_fields)}
        {_img(plots_dir / "01_challenge_score_decomposition.png", "Challenge Score Decomposition")}

        <h2>3. Key Global Metrics</h2>
        {_table(global_fields)}

        <div class="comments">
          <strong>Automatic observations:</strong>
          <ul>{comments_html}</ul>
          <em style="font-size:.85em;">
            Positive bias = model predicts higher than target on average.<br/>
            Negative bias = model predicts lower than target on average.<br/>
            These observations are heuristic \u2014 inspect the plots to confirm.
          </em>
        </div>

        <h2>4. Prediction and Target Distributions</h2>
        <div class="plot-grid">
          {
        _img(plots_dir / "02_target_prediction_distribution.png", "Distributions - linear scale")
    }
          {
        _img(plots_dir / "03_target_prediction_distribution_log.png", "Distributions - log scale")
    }
          {_img(plots_dir / "14_raw_prediction_clipping.png", "Raw prediction clipping")}
        </div>
        {_table(clipping_fields)}

        <h2>5. Calibration and High-Occlusion Behavior</h2>
        <div class="plot-grid">
          {_img(plots_dir / "04_predicted_vs_target.png", "Predicted vs target")}
          {_img(plots_dir / "05_calibration_by_occlusion_bin.png", "Calibration by occlusion bin")}
          {
        _img(
            plots_dir / "06_weighted_error_contribution_by_occlusion_bin.png",
            "Weighted error contribution by bin",
        )
    }
          {_img(plots_dir / "07_bias_by_occlusion_bin.png", "Bias by occlusion bin")}
        </div>

        <h2>6. Gender Diagnostics</h2>
        <div class="plot-grid">
          {_img(plots_dir / "08_weighted_error_by_gender.png", "Weighted MSE by gender")}
          {_img(plots_dir / "09_bias_by_gender.png", "Bias by gender")}
          {
        _img(
            plots_dir / "12_bias_by_gender_and_occlusion_bin.png",
            "Bias by gender and occlusion bin",
        )
    }
        </div>

        <h2>7. Database Diagnostics</h2>
        <div class="plot-grid">
          {_img(plots_dir / "10_weighted_error_by_database.png", "Weighted MSE by database")}
          {_img(plots_dir / "11_bias_by_database.png", "Bias by database")}
        </div>

        <h2>8. Seen / Unseen Identity Diagnostics</h2>
        <div class="plot-grid">
          {
        _img(
            plots_dir / "13_weighted_error_by_group_seen_status.png",
            "Weighted error by group seen status",
        )
    }
        </div>

        <h2>9. Error Distribution</h2>
        {_img(plots_dir / "15_error_distribution.png", "Error distribution")}

        <h2>10. Difficult Examples</h2>
        <div class="plot-grid">
          {samples_html}
        </div>

        <h2>11. Generated Tables</h2>
        <ul>
          {tables_html}
        </ul>

        {dynamics_section}

        </body>
        </html>
    """)

    report_path.write_text(html, encoding="utf-8")
    return report_path


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved validation predictions and generate a complete post-analysis report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Recommended: experiment-folder based
              python scripts/analyze_val_predictions.py \\
                  --experiment-dir outputs/experiments/<run_id>

              # Disable image grids
              python scripts/analyze_val_predictions.py \\
                  --experiment-dir outputs/experiments/<run_id> --no-image-grids

              # Override image root
              python scripts/analyze_val_predictions.py \\
                  --experiment-dir outputs/experiments/<run_id> \\
                  --image-root /data/crops/Crop_224_5fp_100K

              # Backward-compatible explicit paths
              python scripts/analyze_val_predictions.py \\
                  --predictions outputs/experiments/<run_id>/predictions/val_predictions.csv \\
                  --output-dir  outputs/experiments/<run_id>/reports
        """),
    )
    parser.add_argument("--experiment-dir", default=None, metavar="PATH")
    parser.add_argument("--predictions", default=None, metavar="PATH")
    parser.add_argument("--output-dir", default=None, metavar="PATH")
    parser.add_argument("--split-csv", default=None, metavar="PATH")
    parser.add_argument(
        "--bins",
        nargs="+",
        type=float,
        default=DEFAULT_BINS,
        metavar="FLOAT",
        help="Occlusion bin edges (default: 0.0 0.05 0.10 0.20 0.40 0.60 1.0)",
    )
    parser.add_argument(
        "--image-root",
        default="data/crops/Crop_224_5fp_100K",
        metavar="PATH",
        help="Root directory for image lookup (default: data/crops/Crop_224_5fp_100K)",
    )
    parser.add_argument(
        "--no-image-grids",
        action="store_true",
        help="Disable image grid generation (grids are on by default)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        metavar="INT",
        help="Rows to include in error tables (default: 100)",
    )
    parser.add_argument(
        "--grid-k",
        type=int,
        default=16,
        metavar="INT",
        help="Images per grid tile (default: 16)",
    )
    args = parser.parse_args()
    args.predictions, args.output_dir, args.split_csv = _resolve_paths(args, parser)
    if len(args.bins) < 2:
        parser.error("--bins must contain at least two values.")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if args.grid_k < 1:
        parser.error("--grid-k must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    pred_path: Path = args.predictions
    output_dir: Path = args.output_dir
    split_csv: Path | None = args.split_csv
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    save_image_grids = not args.no_image_grids
    image_root = Path(args.image_root)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    samples_dir = output_dir / "samples"

    for d in (output_dir, tables_dir, plots_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    _validate_prediction_columns(df)
    df = _add_error_columns(df)
    df = _add_occlusion_bin(df, list(args.bins))
    df = _add_face_id_columns(df)
    df = _add_seen_status(df, split_csv)

    summary = _compute_summary_metrics(df, pred_path, split_csv, top_k=args.top_k)
    summary["report_html"] = str(output_dir / "report.html")

    (output_dir / "summary_metrics.json").write_text(
        json.dumps(_json_ready(summary), indent=2),
        encoding="utf-8",
    )

    _write_tables(df, tables_dir)
    tables = _write_error_tables(df, tables_dir, top_k=args.top_k)
    _write_plots(df, summary, plots_dir)

    # Training dynamics from the Lightning CSV logger metrics file.
    dynamics_plots: list[Path] | None = None
    metrics_csv = _find_metrics_csv(experiment_dir)
    if metrics_csv is not None:
        metrics_df = _read_training_metrics(metrics_csv)
        if metrics_df is not None:
            dynamics_plots = _write_training_dynamics_plots(metrics_df, plots_dir)
            print(f"[analyze] Metrics CSV:  {metrics_csv} ({len(metrics_df)} epochs)")
        else:
            _warn(f"Could not parse training metrics from {metrics_csv}")
    elif experiment_dir is not None:
        _warn(
            "Training metrics CSV not found in experiment directory. "
            "Training dynamics plots will be skipped."
        )

    if save_image_grids:
        if not image_root.exists():
            _warn(f"--image-root not found: {image_root}  \u2014 skipping image grids.")
        else:
            _write_image_grids(tables, samples_dir, image_root=image_root, grid_k=args.grid_k)

    _write_html_report(
        output_dir=output_dir,
        summary=summary,
        experiment_dir=experiment_dir,
        pred_path=pred_path,
        split_csv=split_csv,
        plots_dir=plots_dir,
        samples_dir=samples_dir,
        tables_dir=tables_dir,
        df=df,
        dynamics_plots=dynamics_plots,
    )

    print(f"[analyze] Predictions:  {pred_path}")
    print(f"[analyze] Reports:      {output_dir}")
    print(f"[analyze] HTML report:  {output_dir / 'report.html'}")
    if split_csv:
        print(f"[analyze] Split CSV:    {split_csv}")
    score = summary.get("score")
    if score is not None:
        try:
            if not math.isnan(float(score)):
                print(f"[analyze] Score:        {float(score):.5f}")
        except (TypeError, ValueError):
            pass


if __name__ == "__main__":
    main()
