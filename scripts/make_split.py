#!/usr/bin/env python
"""Generate a gender x occlusion-bin stratified train/val split."""

from __future__ import annotations

import argparse

import pandas as pd

from face_occlusion.data.splits import make_stratified_split, save_split
from face_occlusion.utils import load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.project.seed))

    df = pd.read_csv(cfg.data.train_csv)
    split = make_stratified_split(
        df,
        target_col=cfg.data.target_col,
        gender_col=cfg.data.gender_col,
        id_col=cfg.data.id_col,
        bins=list(cfg.split.occlusion_bins),
        val_size=float(cfg.data.val_size),
        seed=int(cfg.project.seed),
    )
    save_split(split, cfg.split.split_path)

    counts = split["split"].value_counts().to_dict()
    print(f"Split written to: {cfg.split.split_path}")
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
