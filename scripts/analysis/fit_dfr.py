#!/usr/bin/env python
"""Deep Feature Reweighting (DFR): refit the champion's linear head on gender×occ-balanced
features to remove the gender shortcut (see tmp/model_study/05_gender_gap.md).

    python -m scripts.analysis.fit_dfr --config <run>/config.yaml \
        --checkpoint <run>/checkpoints/best.ckpt [--ridge 1.0] [--predict-test]

Freezes the encoder, extracts pooled features for train+val, fits a closed-form weighted-ridge
head on group-balanced train features, then reports the original-vs-DFR challenge score / gender
gap on val and writes ``<out>/predictions/val_predictions.csv`` (training schema → gate with
``compare_experiments`` / ``bootstrap_metrics``). Linear-head models only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from face_occlusion.data.datamodule import FaceOcclusionDataModule
from face_occlusion.data.dataset import FaceOcclusionDataset
from face_occlusion.data.transforms import build_eval_transform
from face_occlusion.metrics.challenge_metric import challenge_score
from face_occlusion.training import FaceOcclusionLitModule
from face_occlusion.training.dfr import (
    apply_head,
    fit_ridge_head,
    group_balance_weights,
    occlusion_group_ids,
)
from face_occlusion.utils import load_config, seed_everything

from ..inference.predict_test import build_submission


def _eval_loader(ds, cfg) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _extract(model, loader, device, *, with_labels: bool):
    model.eval().to(device)
    feats, ys, gs = [], [], []
    meta = {
        k: [] for k in ("image_id", "filename", "path", "database", "source_subfolder", "group_id")
    }
    face_ids = []
    for batch in loader:
        feats.append(model.pooled_features(batch["image"].to(device)).float().cpu().numpy())
        for k in meta:
            meta[k] += list(batch[k])
        face_ids.append(batch["face_id"].numpy())
        if with_labels:
            ys.append(batch["target"].numpy())
            gs.append(batch["gender"].numpy())
    meta["face_id"] = np.concatenate(face_ids)
    X = np.concatenate(feats)
    y = np.concatenate(ys) if with_labels else None
    g = np.concatenate(gs) if with_labels else None
    return X, y, g, meta


def _score(pred_clipped, y, g, fv, mv) -> dict:
    gstr = np.array([f"{float(x):.1f}" for x in g])
    s = challenge_score(pred_clipped, y, gstr, female_value=fv, male_value=mv)
    w = 1 / 30 + y
    bulk = y < 0.4
    fmask, mmask = gstr == fv, gstr == mv

    def err(m):
        return float((w[m] * (pred_clipped[m] - y[m]) ** 2).sum() / w[m].sum())

    bulk_gap = abs(err(fmask & bulk) - err(mmask & bulk))
    return {
        "score": float(s["score"]),
        "err_female": float(s["err_female"]),
        "err_male": float(s["err_male"]),
        "gap": abs(float(s["err_female"]) - float(s["err_male"])),
        "bulk_gap": bulk_gap,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument(
        "--balance",
        choices=["gender_occ", "gender", "none"],
        default="gender_occ",
        help="Group-balancing for the head refit (default gender×occlusion).",
    )
    ap.add_argument(
        "--no-metric-weight",
        action="store_true",
        help="Drop the challenge weight w=1/30+y from the refit (keep only group balance).",
    )
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(
        int(cfg.project.seed), deterministic=bool(cfg.project.get("deterministic", False))
    )
    act = str(cfg.model.output_activation)
    fv = str(cfg.data.get("female_value", "0.0"))
    mv = str(cfg.data.get("male_value", "1.0"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    module = FaceOcclusionLitModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model = module.model
    if model.head is not None:
        raise SystemExit("[dfr] only linear-head models are supported (champion).")

    dm = FaceOcclusionDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    eval_tf = build_eval_transform(cfg)
    common = dict(
        image_root=cfg.data.image_root,
        image_col=cfg.data.image_col,
        target_col=cfg.data.target_col,
        gender_col=cfg.data.gender_col,
        id_col=cfg.data.id_col,
        target_scale=cfg.data.target_scale,
    )
    train_ds = FaceOcclusionDataset(dm.train_df, transform=eval_tf, mode="val", **common)

    print(f"[dfr] extracting features (device={device}) ...")
    Xtr, ytr, gtr, _ = _extract(model, _eval_loader(train_ds, cfg), device, with_labels=True)
    Xva, yva, gva, va_meta = _extract(model, _eval_loader(dm.val_ds, cfg), device, with_labels=True)
    print(f"[dfr] features: train {Xtr.shape}, val {Xva.shape}")

    # Group-balanced sample weights × the challenge weight, on TRAIN.
    edges = np.asarray(list(cfg.split.occlusion_bins), dtype=float)
    if args.balance == "gender_occ":
        gids = occlusion_group_ids(ytr, gtr, edges, female_value=float(fv), male_value=float(mv))
    elif args.balance == "gender":
        gids = occlusion_group_ids(
            ytr, gtr, [0.0, 1.01], female_value=float(fv), male_value=float(mv)
        )
    else:
        gids = np.zeros(len(ytr), dtype=int)
    sw = group_balance_weights(gids)
    if not args.no_metric_weight:
        sw = sw * (1 / 30 + ytr)
        sw = sw / sw.mean()

    # Fit target: identity head -> y; sigmoid head -> logit(y).
    fit_target = ytr
    if act == "sigmoid":
        yc = np.clip(ytr, 1e-4, 1 - 1e-4)
        fit_target = np.log(yc / (1 - yc))
    weight, bias = fit_ridge_head(Xtr, fit_target, sample_weight=sw, ridge=args.ridge)

    def predict(X):
        raw = apply_head(X, weight, bias)
        if act == "sigmoid":
            raw = 1 / (1 + np.exp(-raw))
        return np.clip(raw, 0.0, 1.0), raw

    # Original head (same features) for an apples-to-apples comparison.
    clf = model.backbone.get_classifier()
    w0 = clf.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(clf.bias.detach().cpu().numpy().reshape(-1)[0])
    orig_raw = apply_head(Xva, w0, b0)
    if act == "sigmoid":
        orig_raw = 1 / (1 + np.exp(-orig_raw))
    orig_pred = np.clip(orig_raw, 0.0, 1.0)
    dfr_pred, dfr_raw = predict(Xva)

    m_orig = _score(orig_pred, yva, gva, fv, mv)
    m_dfr = _score(dfr_pred, yva, gva, fv, mv)
    print("\n== VAL: original head vs DFR head (same features) ==")
    print("            score     err_F     err_M     gap       bulk_gap")
    for name, m in (("original", m_orig), ("DFR     ", m_dfr)):
        print(
            f"  {name} {m['score']:.6f} {m['err_female']:.6f} {m['err_male']:.6f} "
            f"{m['gap']:.6f} {m['bulk_gap']:.6f}"
        )
    print(
        f"  Δ(DFR-orig) score={m_dfr['score'] - m_orig['score']:+.6f} "
        f"gap={m_dfr['gap'] - m_orig['gap']:+.6f} "
        f"bulk_gap={m_dfr['bulk_gap'] - m_orig['bulk_gap']:+.6f}"
    )

    run = Path(args.checkpoint).parent.parent.name
    out = Path(args.out_dir) if args.out_dir else Path("outputs/dfr") / f"{run}_dfr"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "dfr_head.npz",
        weight=weight,
        bias=bias,
        activation=act,
        balance=args.balance,
        ridge=args.ridge,
    )
    pd.DataFrame(
        {
            "image_id": va_meta["image_id"],
            "filename": va_meta["filename"],
            "path": va_meta["path"],
            "gender": gva,
            "target": yva,
            "pred_raw": dfr_raw,
            "pred_clipped": dfr_pred,
            "abs_error": np.abs(dfr_pred - yva),
            "database": va_meta["database"],
            "source_subfolder": va_meta["source_subfolder"],
            "group_id": va_meta["group_id"],
            "face_id": va_meta["face_id"],
        }
    ).to_csv(out / "predictions" / "val_predictions.csv", index=False)
    (out / "dfr_metrics.json").write_text(json.dumps({"original": m_orig, "dfr": m_dfr}, indent=2))
    print(f"\n[dfr] wrote {out}/predictions/val_predictions.csv + dfr_head.npz + dfr_metrics.json")

    if args.predict_test:
        dm.setup("predict")
        Xte, _, _, te_meta = _extract(
            model, _eval_loader(dm.test_ds, cfg), device, with_labels=False
        )
        te_pred, te_raw = predict(Xte)
        ext = pd.DataFrame(
            {
                "image_id": te_meta["image_id"],
                "filename": te_meta["filename"],
                "path": te_meta["path"],
                "pred_raw": te_raw,
                "pred_clipped": te_pred,
                "database": te_meta["database"],
                "source_subfolder": te_meta["source_subfolder"],
                "group_id": te_meta["group_id"],
                "face_id": te_meta["face_id"],
            }
        )
        ext.to_csv(out / "predictions" / "test_predictions_extended.csv", index=False)
        build_submission(ext, cfg).to_csv(out / "predictions" / "test_predictions.csv", index=False)
        print(f"[dfr] wrote test predictions ({len(ext)} rows) under {out}/predictions/")


if __name__ == "__main__":
    main()
