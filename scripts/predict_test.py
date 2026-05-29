#!/usr/bin/env python
"""Generate test-set predictions and a submission file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.inference import predict_dataframe
from face_occlusion.training import FaceOcclusionLitModule
from face_occlusion.utils import load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.project.seed))

    dm = FaceOcclusionDataModule(cfg)
    dm.setup("predict")
    loader = dm.test_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = FaceOcclusionLitModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
    df = predict_dataframe(module.model, loader, device=device)

    out_dir = Path(cfg.project.output_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extended file for analysis.
    ext_path = out_dir / "test_predictions_extended.csv"
    df.to_csv(ext_path, index=False)

    # Submission file: mirror train.csv columns -> filename, FaceOcclusion, gender.
    submission = pd.DataFrame(
        {
            cfg.data.image_col: df["image_id"],
            cfg.data.target_col: df["pred_clipped"],
            cfg.data.gender_col: df["gender"],
        }
    )
    sub_path = out_dir / "test_predictions.csv"
    submission.to_csv(sub_path, index=False)

    print(f"[predict] Submission: {sub_path}")
    print(f"[predict] Extended:   {ext_path}")


if __name__ == "__main__":
    main()
