#!/usr/bin/env python
"""Generate test-set predictions and a submission file."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import pandas as pd
import torch

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.inference import predict_dataframe
from face_occlusion.training import FaceOcclusionLitModule
from face_occlusion.utils import load_config, seed_everything


def _default_output_dir(cfg, checkpoint: str | Path) -> Path:
    ckpt_path = Path(checkpoint)
    # Experiment checkpoints live in <run_dir>/checkpoints; keep predictions beside them.
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.name != "outputs":
        return ckpt_path.parent.parent / "predictions"
    return Path(cfg.project.output_dir) / "predictions"


def build_submission(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Build the challenge submission with a dummy gender column required by the platform."""

    dummy_gender = cfg.data.get("submission_dummy_gender", cfg.data.get("female_value", 0))
    return pd.DataFrame(
        {
            cfg.data.image_col: df["filename"],
            cfg.data.target_col: df["pred_clipped"],
            cfg.data.gender_col: dummy_gender,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional prediction output directory. Defaults to the checkpoint run "
            "folder when possible."
        ),
    )
    parser.add_argument(
        "--recalibration",
        default=None,
        metavar="MAPPING_JSON",
        help=(
            "Optional post-hoc recalibration mapping (from fit_recalibration.py) applied "
            "to raw predictions before clipping. Omit for no recalibration."
        ),
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation: average the prediction over the image + its hflip. "
        "Defaults to the config's inference.tta.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Seed identically to scripts.training.train so the eval pipeline is
    # reproducible (review R6).
    deterministic = bool(cfg.project.get("deterministic", False))
    seed_everything(int(cfg.project.seed), deterministic=deterministic)

    dm = FaceOcclusionDataModule(cfg)
    dm.setup("predict")
    loader = dm.test_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = FaceOcclusionLitModule.load_from_checkpoint(args.checkpoint, cfg=cfg)

    recalibration = None
    if args.recalibration:
        from face_occlusion.calibration import load_mapping

        recalibration = load_mapping(args.recalibration)
        print(f"[predict] Applying recalibration: {args.recalibration}")
    inference_cfg = cfg.get("inference", {}) if hasattr(cfg, "get") else {}
    tta = bool(args.tta or (inference_cfg.get("tta", False) if inference_cfg else False))
    if tta:
        print("[predict] Test-time augmentation: image + horizontal flip")
    df = predict_dataframe(
        module.model, loader, device=device, recalibration=recalibration, tta=tta
    )

    out_dir = (
        Path(args.output_dir) if args.output_dir else _default_output_dir(cfg, args.checkpoint)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extended file keeps metadata and raw predictions for analysis.
    ext_path = out_dir / "test_predictions_extended.csv"
    df.to_csv(ext_path, index=False)

    # Submission file mirrors the challenge columns: filename, prediction, dummy gender.
    submission = build_submission(df, cfg)
    sub_path = out_dir / "test_predictions.csv"
    submission.to_csv(sub_path, index=False)

    print(f"[predict] Submission: {sub_path}")
    print(f"[predict] Extended:   {ext_path}")


if __name__ == "__main__":
    main()
