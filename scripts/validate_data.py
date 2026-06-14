#!/usr/bin/env python
"""Validate the configured dataset and write a lightweight data report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import pandas as pd
from PIL import Image

from face_occlusion.data.metadata import add_path_metadata
from face_occlusion.utils import load_config


def _counts(series: pd.Series) -> dict:
    return series.value_counts(dropna=False).sort_index().to_dict()


def _check_images(df: pd.DataFrame, image_col: str, image_root: Path, max_images: int) -> dict:
    paths = df[image_col].astype(str).tolist()
    if max_images and max_images < len(paths):
        paths = paths[:max_images]
    missing = []
    unreadable = []
    for rel in paths:
        path = image_root / rel
        if not path.exists():
            missing.append(rel)
            continue
        try:
            with Image.open(path) as img:
                img.convert("RGB")
        except Exception as exc:
            unreadable.append({"path": rel, "error": str(exc)})
    return {
        "checked": len(paths),
        "missing": len(missing),
        "unreadable": len(unreadable),
        "missing_examples": missing[:5],
        "unreadable_examples": unreadable[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max-image-check",
        type=int,
        default=0,
        help="If > 0, sample N images to verify they open. 0 = check all.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # The report is JSON-serializable so it can be inspected locally or in CI logs.
    report: dict = {"errors": [], "warnings": [], "stats": {}}

    train_csv = Path(cfg.data.train_csv)
    test_csv = Path(cfg.data.test_csv)
    image_root = Path(cfg.data.image_root)

    if not train_csv.exists():
        report["errors"].append(f"train_csv missing: {train_csv}")
    if not test_csv.exists():
        report["errors"].append(f"test_csv missing: {test_csv}")
    if not image_root.exists():
        report["errors"].append(f"image_root missing: {image_root}")

    if train_csv.exists():
        train = pd.read_csv(train_csv)
        report["stats"]["train_rows"] = len(train)

        for col in (cfg.data.image_col, cfg.data.target_col, cfg.data.gender_col):
            if col not in train.columns:
                report["errors"].append(f"train missing column '{col}'")

        if cfg.data.image_col in train.columns:
            train = add_path_metadata(train, filename_col=cfg.data.image_col)
            dups = train[cfg.data.image_col].duplicated().sum()
            if dups:
                report["warnings"].append(f"{dups} duplicate {cfg.data.image_col} in train")
            report["stats"]["train_database_counts"] = _counts(train["database"])
            report["stats"]["train_unique_group_ids"] = int(train["group_id"].nunique())
            report["stats"]["train_face_id_counts"] = _counts(train["face_id"])

        if cfg.data.target_col in train.columns:
            t = train[cfg.data.target_col].astype(float)
            report["stats"]["target_min"] = float(t.min())
            report["stats"]["target_max"] = float(t.max())
            report["stats"]["target_mean"] = float(t.mean())
            report["stats"]["target_nan"] = int(t.isna().sum())
            report["stats"]["target_quantiles"] = {
                str(q): float(t.quantile(q)) for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            }
            if "database" in train.columns:
                report["stats"]["target_mean_by_database"] = (
                    train.groupby("database")[cfg.data.target_col].mean().to_dict()
                )

        if cfg.data.gender_col in train.columns and cfg.data.target_col in train.columns:
            report["stats"]["gender_encoding"] = {
                "female": cfg.data.get("female_value", 0.0),
                "male": cfg.data.get("male_value", 1.0),
            }
            report["stats"]["gender_counts"] = _counts(train[cfg.data.gender_col])
            stats_by_g = (
                train.groupby(cfg.data.gender_col)[cfg.data.target_col]
                .agg(["count", "mean", "min", "max"])
                .reset_index()
            )
            report["stats"]["by_gender"] = stats_by_g.to_dict(orient="records")

        if cfg.data.image_col in train.columns:
            # Image existence + openability catches broken paths before training.
            image_check = _check_images(
                train, cfg.data.image_col, image_root, max_images=args.max_image_check
            )
            report["stats"]["train_images"] = image_check
            if image_check["missing_examples"]:
                report["warnings"].append(
                    {"train_missing_examples": image_check["missing_examples"]}
                )
            if image_check["unreadable_examples"]:
                report["warnings"].append(
                    {"train_unreadable_examples": image_check["unreadable_examples"]}
                )

    if test_csv.exists():
        test = pd.read_csv(test_csv)
        report["stats"]["test_rows"] = len(test)
        if cfg.data.image_col not in test.columns:
            report["errors"].append(f"test missing column '{cfg.data.image_col}'")
        else:
            test = add_path_metadata(test, filename_col=cfg.data.image_col)
            report["stats"]["test_database_counts"] = _counts(test["database"])
            report["stats"]["test_unique_group_ids"] = int(test["group_id"].nunique())
            report["stats"]["test_face_id_counts"] = _counts(test["face_id"])
            image_check = _check_images(
                test, cfg.data.image_col, image_root, max_images=args.max_image_check
            )
            report["stats"]["test_images"] = image_check
            if image_check["missing_examples"]:
                report["warnings"].append(
                    {"test_missing_examples": image_check["missing_examples"]}
                )
            if image_check["unreadable_examples"]:
                report["warnings"].append(
                    {"test_unreadable_examples": image_check["unreadable_examples"]}
                )

            if train_csv.exists() and cfg.data.image_col in train.columns:
                report["stats"]["train_test_overlapping_group_ids"] = int(
                    len(set(train["group_id"]) & set(test["group_id"]))
                )

    out_dir = Path(cfg.project.output_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print("=== Data validation summary ===")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to: {out_path}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
