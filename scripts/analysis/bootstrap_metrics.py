#!/usr/bin/env python
"""Bootstrap confidence intervals for the challenge metric on a predictions CSV.

The tiny, leaked validation tail makes a bare ``val/score`` point estimate
uninterpretable: tail-metric movements are mostly sampling noise. This script
resamples the validation rows and reports percentile CIs so an ablation delta
can be judged as signal vs noise.

Examples
--------
    # Single split, row bootstrap.
    python -m scripts.analysis.bootstrap_metrics \
        --predictions outputs/experiments/<run>/predictions/val_predictions.csv

    # Honest CI under identity leakage: cluster bootstrap on group_id.
    python -m scripts.analysis.bootstrap_metrics \
        --predictions <run>/predictions/val_predictions.csv --unit group

    # Dual-split report: leaderboard proxy (row) vs leakage-free (group).
    python -m scripts.analysis.bootstrap_metrics \
        --predictions <row_run>/predictions/val_predictions.csv \
        --group-predictions <group_run>/predictions/val_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import pandas as pd

from face_occlusion.metrics.bootstrap import MetricCI, bootstrap_challenge_metrics

# Order in which metrics are printed.
_METRIC_ORDER = [
    "score",
    "err_mean",
    "gender_gap",
    "err_female",
    "err_male",
    "high_occ_err",
    "high_occ_gender_gap",
]


def _bootstrap_from_csv(
    path: Path,
    *,
    unit: str,
    n_boot: int,
    ci: float,
    seed: int,
    pred_col: str,
) -> dict[str, MetricCI]:
    df = pd.read_csv(path)
    for col in (pred_col, "target", "gender"):
        if col not in df.columns:
            raise KeyError(f"{path} is missing required column {col!r}")
    group_ids = df["group_id"].to_numpy() if "group_id" in df.columns else None
    return bootstrap_challenge_metrics(
        df[pred_col].to_numpy(),
        df["target"].to_numpy(),
        # Match training: gender stored as the "0.0"/"1.0" string label.
        df["gender"].astype(str).to_numpy(),
        group_ids=group_ids,
        unit=unit,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )


def _fmt(ci: MetricCI) -> str:
    if ci.point != ci.point:  # NaN
        return "    nan"
    return f"{ci.point:.4f} [{ci.lo:.4f}, {ci.hi:.4f}]"


def _print_report(columns: dict[str, dict[str, MetricCI]], ci_level: float) -> None:
    headers = list(columns)
    width = max(28, *(len(h) for h in headers))
    print(f"\nBootstrap {int(ci_level * 100)}% CIs (point [lo, hi])\n")
    print("metric".ljust(22) + "".join(h.ljust(width) for h in headers))
    print("-" * (22 + width * len(headers)))
    for m in _METRIC_ORDER:
        row = m.ljust(22)
        for h in headers:
            row += _fmt(columns[h][m]).ljust(width)
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--group-predictions",
        type=Path,
        default=None,
        help="Optional second CSV (e.g. group_stratified split) for a dual-split report.",
    )
    parser.add_argument(
        "--unit",
        choices=["row", "group"],
        default="row",
        help="Bootstrap unit: row (i.i.d.) or group (cluster on group_id, honest under leakage).",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pred-col",
        default="pred_raw",
        help="Prediction column to score (clipped internally, matching training).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    columns: dict[str, dict[str, MetricCI]] = {}
    columns["row_split"] = _bootstrap_from_csv(
        args.predictions,
        unit=args.unit,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=args.seed,
        pred_col=args.pred_col,
    )
    if args.group_predictions is not None:
        columns["group_split"] = _bootstrap_from_csv(
            args.group_predictions,
            unit=args.unit,
            n_boot=args.n_boot,
            ci=args.ci,
            seed=args.seed,
            pred_col=args.pred_col,
        )

    _print_report(columns, args.ci)

    if args.output is not None:
        payload = {
            split: {m: ci.as_dict() for m, ci in metrics.items()}
            for split, metrics in columns.items()
        }
        payload["_meta"] = {
            "unit": args.unit,
            "n_boot": args.n_boot,
            "ci": args.ci,
            "seed": args.seed,
            "pred_col": args.pred_col,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
