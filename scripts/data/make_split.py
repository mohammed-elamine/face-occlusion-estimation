#!/usr/bin/env python
"""Generate the configured train/validation split."""

from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import pandas as pd

from face_occlusion.data.splits import make_stratified_split, save_split
from face_occlusion.utils import load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--strategy",
        choices=["row_stratified", "group_stratified"],
        default=None,
        help="Optional split strategy override for one-off split generation.",
    )
    parser.add_argument(
        "--split-path",
        default=None,
        help="Optional output path override. Useful for keeping row and group splits separate.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.project.seed))

    # This global split can be reused by many configs for fair comparison.
    df = pd.read_csv(cfg.data.train_csv)
    strategy = args.strategy or cfg.split.get("strategy", "row_stratified")
    split_path = args.split_path or cfg.split.split_path
    split = make_stratified_split(
        df,
        target_col=cfg.data.target_col,
        gender_col=cfg.data.gender_col,
        id_col=cfg.data.id_col,
        bins=list(cfg.split.occlusion_bins),
        val_size=float(cfg.split.get("val_size", cfg.data.val_size)),
        seed=int(cfg.split.get("random_state", cfg.project.seed)),
        strategy=strategy,
        stratify_by=list(cfg.split.get("stratify_by", ["gender", "occlusion_bin"])),
        group_col=cfg.split.get("group_col", "group_id"),
    )
    save_split(split, split_path)

    counts = split["split"].value_counts().to_dict()
    print(f"Strategy: {strategy}")
    print(f"Split written to: {split_path}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
