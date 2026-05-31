#!/usr/bin/env python
"""Validate the configured dataset and write a lightweight data report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from face_occlusion.utils import load_config


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
            dups = train[cfg.data.image_col].duplicated().sum()
            if dups:
                report["warnings"].append(f"{dups} duplicate {cfg.data.image_col} in train")

        if cfg.data.target_col in train.columns:
            t = train[cfg.data.target_col].astype(float)
            report["stats"]["target_min"] = float(t.min())
            report["stats"]["target_max"] = float(t.max())
            report["stats"]["target_mean"] = float(t.mean())
            report["stats"]["target_nan"] = int(t.isna().sum())

        if cfg.data.gender_col in train.columns and cfg.data.target_col in train.columns:
            stats_by_g = (
                train.groupby(cfg.data.gender_col)[cfg.data.target_col]
                .agg(["count", "mean", "min", "max"])
                .reset_index()
            )
            report["stats"]["by_gender"] = stats_by_g.to_dict(orient="records")

        # Image existence + openability catches broken paths before training.
        if cfg.data.image_col in train.columns:
            paths = train[cfg.data.image_col].tolist()
            if args.max_image_check and args.max_image_check < len(paths):
                paths = paths[: args.max_image_check]
            missing = []
            unreadable = []
            for rel in paths:
                p = image_root / str(rel)
                if not p.exists():
                    missing.append(str(rel))
                    continue
                try:
                    with Image.open(p) as img:
                        img.convert("RGB")
                except Exception as exc:
                    unreadable.append({"path": str(rel), "error": str(exc)})
            report["stats"]["images_checked"] = len(paths)
            report["stats"]["missing_images"] = len(missing)
            report["stats"]["unreadable_images"] = len(unreadable)
            if missing[:5]:
                report["warnings"].append({"missing_examples": missing[:5]})
            if unreadable[:5]:
                report["warnings"].append({"unreadable_examples": unreadable[:5]})

    if test_csv.exists():
        test = pd.read_csv(test_csv)
        report["stats"]["test_rows"] = len(test)
        if cfg.data.image_col not in test.columns:
            report["errors"].append(f"test missing column '{cfg.data.image_col}'")

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
