#!/usr/bin/env python
"""Build a challenge submission by averaging several models' test predictions.

Ensembling decorrelated, individually-tied models is the lever that beat the single-model
champion on this challenge. This driver fuses members given as **experiment run folders**:

  python -m scripts.inference.predict_ensemble \
      --members outputs/experiments/<champion> \
                outputs/experiments/<sigmoid> \
                outputs/experiments/<dldl> \
      [--weights 1 1 1] [--tta] [--output-dir outputs/ensemble_submission]

For each member it needs that member's **test** predictions
(``predictions/test_predictions_extended.csv``, produced by ``scripts.inference.predict_test``).
If they are missing it will generate them when the member's checkpoint is present
(``checkpoints/best.ckpt``) — otherwise it tells you to run ``predict_test`` for that member
first (checkpoints live on the training pod). It also prints the ensemble **val/score** from
each member's on-disk ``val_predictions.csv`` (no checkpoints needed) so the expected number is
confirmed before you trust the submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import pandas as pd

from face_occlusion.inference import ensemble_average, score_val_ensemble
from face_occlusion.utils import load_config

from .predict_test import build_submission


def _member_test_predictions(
    run_dir: Path, checkpoint_name: str, tta: bool, regenerate: bool
) -> pd.DataFrame:
    """Return a member's per-image test predictions, generating them if needed/possible."""
    ext_path = run_dir / "predictions" / "test_predictions_extended.csv"
    if ext_path.exists() and not regenerate:
        return pd.read_csv(ext_path)

    ckpt = run_dir / "checkpoints" / checkpoint_name
    cfg_path = run_dir / "config.yaml"
    if not (ckpt.exists() and cfg_path.exists()):
        raise FileNotFoundError(
            f"{run_dir.name}: no test predictions at {ext_path} and cannot generate them "
            f"(missing {ckpt} or {cfg_path}). Run `python -m scripts.inference.predict_test "
            f"--config {cfg_path} --checkpoint {ckpt}` on the machine holding the checkpoints "
            f"(the pod), then re-run this command."
        )

    # Generate test predictions for this member from its own config + checkpoint.
    import torch

    from face_occlusion.data.datamodule import FaceOcclusionDataModule
    from face_occlusion.inference import predict_dataframe
    from face_occlusion.training import FaceOcclusionLitModule
    from face_occlusion.utils import seed_everything

    cfg = load_config(str(cfg_path))
    seed_everything(
        int(cfg.project.seed),
        deterministic=bool(cfg.project.get("deterministic", False)),
    )
    dm = FaceOcclusionDataModule(cfg)
    dm.setup("predict")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = FaceOcclusionLitModule.load_from_checkpoint(str(ckpt), cfg=cfg)
    inference_cfg = cfg.get("inference", {}) if hasattr(cfg, "get") else {}
    use_tta = bool(tta or (inference_cfg.get("tta", False) if inference_cfg else False))
    df = predict_dataframe(module.model, dm.test_dataloader(), device=device, tta=use_tta)
    ext_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ext_path, index=False)
    print(f"[ensemble] generated {ext_path} ({'TTA' if use_tta else 'no TTA'})")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members",
        nargs="+",
        required=True,
        help="Experiment run folders to ensemble (each holds config.yaml + predictions/).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-member weights (default: equal). Must match --members length.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ensemble_submission",
        help="Where to write the submission + extended ensemble CSV.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config for submission column names / dummy gender (default: first member's).",
    )
    parser.add_argument("--checkpoint-name", default="best.ckpt")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use image+hflip TTA when generating any missing member test predictions.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run per-member test inference even if cached predictions exist.",
    )
    args = parser.parse_args()

    members = [Path(m) for m in args.members]
    if len(members) < 2:
        parser.error("ensembling needs at least two members")
    if args.weights is not None and len(args.weights) != len(members):
        parser.error(f"--weights ({len(args.weights)}) must match --members ({len(members)})")
    weights = args.weights
    wlabel = " ".join(f"{w:g}" for w in weights) if weights is not None else "equal"
    print(f"[ensemble] {len(members)} members, weights: {wlabel}")
    for i, m in enumerate(members):
        print(f"  member_{i}: {m.name}")

    # 1) Confirm the expected ensemble val/score from on-disk artifacts (no checkpoints).
    try:
        score, _ = score_val_ensemble(members, weights)
        gap = abs(score["err_female"] - score["err_male"])
        print(
            f"[ensemble] val/score = {score['score']:.5f}  "
            f"(err_F {score['err_female']:.5f}, err_M {score['err_male']:.5f}, gap {gap:.5f})"
        )
    except FileNotFoundError as exc:
        print(f"[ensemble] skipping val verification ({exc})")

    # 2) Gather each member's test predictions (generate from checkpoint if available).
    try:
        frames = [
            _member_test_predictions(m, args.checkpoint_name, args.tta, args.regenerate)
            for m in members
        ]
    except FileNotFoundError as exc:
        raise SystemExit(f"[ensemble] {exc}")

    # 3) Average pred_clipped across members, aligned on image_id.
    ensemble = ensemble_average(
        frames,
        weights,
        value_col="pred_clipped",
        keep_cols=["image_id", "filename", "path"],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_path = out_dir / "ensemble_test_predictions.csv"
    ensemble.to_csv(ext_path, index=False)

    cfg = load_config(args.config) if args.config else load_config(str(members[0] / "config.yaml"))
    submission = build_submission(ensemble, cfg)
    sub_path = out_dir / "test_predictions.csv"
    submission.to_csv(sub_path, index=False)

    print(f"[ensemble] Submission: {sub_path}  ({len(submission)} rows)")
    print(f"[ensemble] Extended:   {ext_path}")


if __name__ == "__main__":
    main()
