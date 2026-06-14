#!/usr/bin/env python
"""Fit and gate a post-hoc recalibration of a run's predictions (no retraining).

The regressor under-predicts the mid-high occlusion band. This fits a monotonic map
``g(ŷ) -> y`` (challenge-weighted isotonic) that corrects it, and adjudicates it with the
evaluation gate (paired-Δ significance + balanced/test-matched lenses + leakage-free
score). It is also a *diagnostic*: compare high-occlusion error before/after — if it
recovers, the failure was correctable bias; if not, the model lacks discrimination and a
training-side fix is needed.

Honesty: the score the gate sees comes from **out-of-fold** recalibration (identities
never shared between the isotonic-fit set and the held-out set). The single mapping saved
for inference is fit on all of validation (deploy never sees labels, so no leak there).

Note: the test set has no gender (the submission adds a dummy), so only the GLOBAL map is
deployable. ``--per-gender on`` adds a diagnostic only.

Usage:
    python -m scripts.analysis.fit_recalibration --experiment-dir outputs/experiments/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from face_occlusion.calibration import fit_weighted_isotonic, oof_recalibrate, save_mapping
from face_occlusion.metrics.bootstrap import bootstrap_per_bin, bootstrap_score_delta
from face_occlusion.metrics.challenge_metric import weighted_mse
from face_occlusion.metrics.eval_lenses import lens_weights

DEFAULT_BINS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]


def _auto_split_csv(experiment_dir: Path) -> Path | None:
    split_dir = experiment_dir / "splits"
    if not split_dir.exists():
        return None
    csvs = sorted(split_dir.glob("*.csv"))
    return csvs[0] if len(csvs) == 1 else None


def _seen_status(df: pd.DataFrame, split_csv: Path | None) -> pd.Series | None:
    if split_csv is None or not split_csv.exists():
        return None
    split = pd.read_csv(split_csv)
    if "group_id" not in split.columns or "split" not in split.columns:
        return None
    train_groups = set(split.loc[split["split"] == "train", "group_id"].astype(str))
    return (
        df["group_id"]
        .astype(str)
        .map(lambda g: "seen_in_train" if g in train_groups else "unseen_in_train")
    )


def _ci(metric_ci) -> dict[str, float]:
    return {"point": metric_ci.point, "lo": metric_ci.lo, "hi": metric_ci.hi}


def _delta_block(
    recal: np.ndarray, raw: np.ndarray, df: pd.DataFrame, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Paired Δ (recalibrated − raw) for the official metric and each lens."""
    targets = df["target"].to_numpy(dtype=float)
    genders = df["gender"].to_numpy()
    gids = df["group_id"].to_numpy() if "group_id" in df.columns else None
    unit = "group" if gids is not None else "row"
    out: dict[str, Any] = {}
    for lens in ("official", "balanced", "test_matched"):
        sw = lens_weights(lens, targets)
        d = bootstrap_score_delta(
            recal,
            raw,
            targets,
            genders,
            group_ids=gids,
            unit=unit,
            sample_weight=sw,
            n_boot=n_boot,
            seed=seed,
        )
        out[lens] = {"score": _ci(d["score"]), "gender_gap": _ci(d["gender_gap"])}
    return out


def _significance(block: dict[str, float]) -> str:
    lo, hi = block["lo"], block["hi"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "?"
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "ns"


def main() -> None:
    p = argparse.ArgumentParser(description="Fit + gate a post-hoc recalibration map.")
    p.add_argument("--experiment-dir", required=True, metavar="PATH")
    p.add_argument("--pred-col", default="pred_raw", choices=["pred_raw", "pred_clipped"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--slope-cap", type=float, default=3.0)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--per-gender", choices=["off", "on"], default="off")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run = Path(args.experiment_dir)
    pred_path = run / "predictions" / "val_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"[recal] predictions not found: {pred_path}")
    df = pd.read_csv(pred_path)
    raw = df[args.pred_col].to_numpy(dtype=float)
    raw_clipped = np.clip(raw, 0.0, 1.0)
    targets = df["target"].to_numpy(dtype=float)
    groups = df["group_id"].to_numpy() if "group_id" in df.columns else np.arange(len(df))

    fit_kw = dict(slope_cap=args.slope_cap, min_samples=args.min_samples)

    # 1. Honest out-of-fold recalibration (what the gate scores).
    oof = oof_recalibrate(raw, targets, groups, n_folds=args.n_folds, seed=args.seed, **fit_kw)
    oof_clipped = np.clip(oof, 0.0, 1.0)

    # 2. Final deploy mapping: fit on all validation rows.
    mapping = fit_weighted_isotonic(raw, targets, meta={"source_run": run.name}, **fit_kw)
    out_dir = run / "calibration"
    save_mapping(mapping, out_dir / "mapping.json")

    # 3. Recalibrated predictions CSV (new file; never mutate the original).
    recal_df = df.copy()
    recal_df["pred_recal_oof"] = oof
    recal_df["pred_clipped"] = oof_clipped  # so analyze/compare can read it directly
    recal_df["pred_raw"] = raw
    recal_df.to_csv(run / "predictions" / "val_predictions_recalibrated.csv", index=False)

    # 4. Gate report ---------------------------------------------------------
    deltas = _delta_block(oof_clipped, raw_clipped, df, n_boot=args.n_boot, seed=args.seed)

    # Per-bin weighted MSE before/after (diagnostic: did the tail recover?).
    pb_raw = bootstrap_per_bin(
        raw_clipped,
        targets,
        edges=DEFAULT_BINS,
        group_ids=groups,
        unit="group",
        n_boot=args.n_boot,
        seed=args.seed,
    )
    pb_recal = bootstrap_per_bin(
        oof_clipped,
        targets,
        edges=DEFAULT_BINS,
        group_ids=groups,
        unit="group",
        n_boot=args.n_boot,
        seed=args.seed,
    )

    hi = targets >= 0.40
    high_occ_raw = weighted_mse(raw_clipped[hi], targets[hi]) if hi.any() else float("nan")
    high_occ_recal = weighted_mse(oof_clipped[hi], targets[hi]) if hi.any() else float("nan")

    # Leakage-free Δ (unseen identities only).
    seen = _seen_status(df, _auto_split_csv(run))
    leakage_free = None
    if seen is not None:
        mask = (seen == "unseen_in_train").to_numpy()
        if mask.sum() > 0 and "group_id" in df.columns:
            sub = df[mask]
            d = bootstrap_score_delta(
                oof_clipped[mask],
                raw_clipped[mask],
                sub["target"].to_numpy(dtype=float),
                sub["gender"].to_numpy(),
                group_ids=sub["group_id"].to_numpy(),
                unit="group",
                n_boot=args.n_boot,
                seed=args.seed,
            )
            leakage_free = {"n_unseen": int(mask.sum()), "score": _ci(d["score"])}

    # Per-gender diagnostic (report-only; NOT deployable — test gender is unknown).
    per_gender = None
    if args.per_gender == "on":
        per_gender = {}
        for gval, gname in ((0.0, "female"), (1.0, "male")):
            gmask = df["gender"].to_numpy() == gval
            if gmask.sum() < 50:
                continue
            g_oof = oof_recalibrate(
                raw[gmask],
                targets[gmask],
                groups[gmask],
                n_folds=args.n_folds,
                seed=args.seed,
                **fit_kw,
            )
            per_gender[gname] = {
                "n": int(gmask.sum()),
                "raw_wmse": weighted_mse(raw_clipped[gmask], targets[gmask]),
                "recal_wmse": weighted_mse(np.clip(g_oof, 0, 1), targets[gmask]),
            }

    report = {
        "run": run.name,
        "pred_col": args.pred_col,
        "n_rows": int(len(df)),
        "n_boot": args.n_boot,
        "mapping": str(out_dir / "mapping.json"),
        "raw_weighted_mse": weighted_mse(raw_clipped, targets),
        "recal_oof_weighted_mse": weighted_mse(oof_clipped, targets),
        "raw_bias": float((raw_clipped - targets).mean()),
        "recal_bias": float((oof_clipped - targets).mean()),
        "high_occ_err_raw": high_occ_raw,
        "high_occ_err_recal": high_occ_recal,
        "paired_delta": deltas,
        "leakage_free": leakage_free,
        "per_gender": per_gender,
        "decision": {lens: _significance(deltas[lens]["score"]) for lens in deltas},
    }
    (out_dir / "recalibration_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "recalibration_report.md").write_text(
        _to_markdown(report, pb_raw, pb_recal), encoding="utf-8"
    )

    # Console summary.
    print(f"[recal] run: {run.name}")
    print(
        f"[recal] weighted MSE: raw {report['raw_weighted_mse']:.6f} "
        f"-> recal {report['recal_oof_weighted_mse']:.6f}  "
        f"(bias {report['raw_bias']:+.4f} -> {report['recal_bias']:+.4f})"
    )
    print(f"[recal] high-occ err (y>=0.4): {high_occ_raw:.5f} -> {high_occ_recal:.5f}")
    for lens in ("official", "balanced", "test_matched"):
        b = deltas[lens]["score"]
        print(
            f"[recal] Δ{lens:13s}: {b['point']:+.6f} [{b['lo']:+.6f}, {b['hi']:+.6f}]  "
            f"-> {report['decision'][lens]}"
        )
    if leakage_free:
        b = leakage_free["score"]
        print(
            f"[recal] Δ leakage-free (n={leakage_free['n_unseen']}): "
            f"{b['point']:+.6f} [{b['lo']:+.6f}, {b['hi']:+.6f}]"
        )
    print(f"[recal] mapping saved: {out_dir / 'mapping.json'}")
    print(f"[recal] report:        {out_dir / 'recalibration_report.md'}")


def _to_markdown(report: dict, pb_raw: dict, pb_recal: dict) -> str:
    d = report["paired_delta"]
    lines = [
        f"# Recalibration gate — `{report['run']}`",
        "",
        f"- Predictions column: `{report['pred_col']}`, rows: {report['n_rows']}, "
        f"bootstrap: {report['n_boot']} (group unit).",
        "- Δ = recalibrated (out-of-fold) − raw on the same rows. **better** = CI below 0.",
        "",
        "## Headline",
        "",
        f"- Weighted MSE: **{report['raw_weighted_mse']:.6f} → "
        f"{report['recal_oof_weighted_mse']:.6f}**",
        f"- Bias: {report['raw_bias']:+.4f} → {report['recal_bias']:+.4f}",
        f"- High-occlusion err (y≥0.4): **{report['high_occ_err_raw']:.5f} → "
        f"{report['high_occ_err_recal']:.5f}** "
        "(the diagnostic: recovered ⇒ bias-only; not ⇒ discrimination gap)",
        "",
        "## Paired Δ by lens",
        "",
        "| lens | Δ score [95% CI] | decision | Δ gender_gap |",
        "|---|---|---|---|",
    ]
    for lens in ("official", "balanced", "test_matched"):
        s = d[lens]["score"]
        g = d[lens]["gender_gap"]
        lines.append(
            f"| {lens} | {s['point']:+.6f} [{s['lo']:+.6f}, {s['hi']:+.6f}] "
            f"| {_significance(s)} | {g['point']:+.6f} |"
        )
    lf = report.get("leakage_free")
    if lf:
        s = lf["score"]
        lines += [
            "",
            f"**Leakage-free Δ** (unseen identities, n={lf['n_unseen']}): "
            f"{s['point']:+.6f} [{s['lo']:+.6f}, {s['hi']:+.6f}]",
        ]
    lines += [
        "",
        "## Per-bin weighted MSE (raw → recal)",
        "",
        "| bin | n | raw | recal |",
        "|---|---|---|---|",
    ]
    for label in pb_raw:
        n = pb_raw[label]["count"]
        r = pb_raw[label]["weighted_mse"].point
        c = pb_recal[label]["weighted_mse"].point
        lines.append(f"| {label} | {n} | {r:.5f} | {c:.5f} |")
    if report.get("per_gender"):
        lines += ["", "## Per-gender diagnostic (report-only; not deployable)", ""]
        for gname, v in report["per_gender"].items():
            lines.append(f"- {gname} (n={v['n']}): {v['raw_wmse']:.6f} → {v['recal_wmse']:.6f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
