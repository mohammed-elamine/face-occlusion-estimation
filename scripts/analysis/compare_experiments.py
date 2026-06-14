#!/usr/bin/env python
"""Compare experiments with confidence intervals and a paired significance test.

Point-estimate ``val/score`` deltas between runs are not trustworthy: the metric's
mass sits in a handful of high-occlusion rows, so marginal CIs overlap for almost any
pair. This tool instead computes, for every run vs a baseline, a **paired bootstrap of
the score difference** on the *same* validation rows (``bootstrap_score_delta``). Because
both runs see identical images, their per-row errors are correlated and the *difference*
is measured far more precisely than either score alone — a delta CI that excludes 0 means
the runs genuinely differ.

It also reports each run's score under the three evaluation lenses
(official / balanced / test-matched) so robustness across distributions is visible.
Everything is post-hoc on saved ``val_predictions.csv`` — no retraining, no checkpoints.

Usage:
    python -m scripts.analysis.compare_experiments \\
        --runs outputs/experiments/<run_a> outputs/experiments/<run_b> ... \\
        --baseline <run_a> --out-dir tmp/comparison_reports/latest

    # or discover runs under a root by glob
    python -m scripts.analysis.compare_experiments \\
        --experiments-root outputs/experiments --glob '*owa*' \\
        --out-dir tmp/comparison_reports/ordinal_warmup
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from face_occlusion.metrics.bootstrap import (
    bootstrap_challenge_metrics,
    bootstrap_score_delta,
)
from face_occlusion.metrics.eval_lenses import LENS_NAMES, lens_weights

KEY_ID = "image_id"
_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_")


def _cfg_knobs(run_dir: Path) -> dict[str, Any]:
    """Best-effort extraction of the config knobs that distinguish runs."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    c = yaml.safe_load(cfg_path.read_text()) or {}
    model = c.get("model", {}) or {}
    training = c.get("training", {}) or {}
    losses = c.get("losses", {}) or {}
    ordc = losses.get("ordinal", {}) or {}
    consc = losses.get("consistency", {}) or {}
    rankc = losses.get("ranking", {}) or {}
    return {
        "backbone": str(model.get("backbone", "?")).split(".")[0],
        "lr": training.get("learning_rate"),
        "wd": training.get("weight_decay"),
        "ord_w": (
            ordc.get("weight") if (model.get("use_ordinal_head") and ordc.get("enabled")) else 0
        ),
        "ord_warm": ordc.get("warmup_epochs", 0) if ordc.get("enabled") else 0,
        "cons": consc.get("weight") if consc.get("enabled") else 0,
        "rank": rankc.get("weight") if rankc.get("enabled") else 0,
        "sampler": bool((c.get("sampler", {}) or {}).get("enabled", False)),
        "synth": bool((c.get("synthetic_occlusion", {}) or {}).get("enabled", False)),
    }


def _load_run(run_dir: Path, pred_col: str) -> dict[str, Any] | None:
    pred_path = run_dir / "predictions" / "val_predictions.csv"
    if not pred_path.exists():
        print(f"[compare] skip {run_dir.name}: no val_predictions.csv")
        return None
    vp = pd.read_csv(pred_path)
    if pred_col not in vp.columns:
        print(f"[compare] skip {run_dir.name}: missing column {pred_col}")
        return None
    if KEY_ID not in vp.columns:
        # Fall back to row order (shared split guarantees alignment).
        vp = vp.reset_index().rename(columns={"index": KEY_ID})
    vp = vp.set_index(KEY_ID, drop=False)
    summary: dict[str, Any] = {}
    sm_path = run_dir / "reports" / "summary_metrics.json"
    if sm_path.exists():
        summary = json.loads(sm_path.read_text())
    return {
        "name": run_dir.name,
        "label": _TIMESTAMP_PREFIX.sub("", run_dir.name),
        "vp": vp,
        "pred_col": pred_col,
        "summary": summary,
        "knobs": _cfg_knobs(run_dir),
    }


def _lens_scores(run: dict[str, Any], *, n_boot: int, ci: float, seed: int) -> dict[str, dict]:
    vp = run["vp"]
    preds = vp[run["pred_col"]].to_numpy(dtype=float)
    targets = vp["target"].to_numpy(dtype=float)
    genders = vp["gender"].to_numpy()
    has_groups = "group_id" in vp.columns
    gids = vp["group_id"].to_numpy() if has_groups else None
    unit = "group" if has_groups else "row"
    out = {}
    for name in LENS_NAMES:
        sw = lens_weights(name, targets)
        res = bootstrap_challenge_metrics(
            preds,
            targets,
            genders,
            group_ids=gids,
            unit=unit,
            sample_weight=sw,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
        )
        out[name] = {"score": res["score"], "high_occ_err": res["high_occ_err"]}
    return out


def _aligned(base_vp: pd.DataFrame, run_vp: pd.DataFrame, pred_col: str):
    """Intersect on image_id and return aligned (preds_base, preds_run, targets, genders, gids)."""
    ids = base_vp.index.intersection(run_vp.index)
    b = base_vp.loc[ids]
    r = run_vp.loc[ids]
    gids = b["group_id"].to_numpy() if "group_id" in b.columns else None
    return (
        b[pred_col].to_numpy(dtype=float),
        r[pred_col].to_numpy(dtype=float),
        b["target"].to_numpy(dtype=float),
        b["gender"].to_numpy(),
        gids,
        len(ids),
    )


def _paired_delta(base: dict, run: dict, *, n_boot: int, ci: float, seed: int):
    """Paired Δscore = score(run) - score(baseline) with CI (same rows)."""
    pb, pr, t, g, gids, n = _aligned(base["vp"], run["vp"], run["pred_col"])
    unit = "group" if gids is not None else "row"
    d = bootstrap_score_delta(
        pr,
        pb,
        t,
        g,
        group_ids=gids,
        unit=unit,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
    )
    return d["score"], d["high_occ_err"], n


def _significant(delta) -> str:
    if not np.isfinite(delta.lo) or not np.isfinite(delta.hi):
        return "?"
    if delta.lo > 0:
        return "worse"  # run - baseline > 0 means run scores higher (worse)
    if delta.hi < 0:
        return "better"
    return "ns"  # not significant


def compare(
    runs: list[Path], baseline: str | None, *, pred_col: str, n_boot: int, ci: float, seed: int
) -> tuple[pd.DataFrame, dict]:
    loaded = [r for r in (_load_run(d, pred_col) for d in runs) if r is not None]
    if not loaded:
        raise SystemExit("[compare] no usable runs found")

    # Default baseline = best (lowest) official score.
    if baseline is None:
        base = min(loaded, key=lambda r: r["summary"].get("score", float("inf")))
    else:
        match = [r for r in loaded if baseline in (r["name"], r["label"])]
        if not match:
            raise SystemExit(f"[compare] baseline {baseline!r} not among runs")
        base = match[0]
    print(f"[compare] baseline = {base['name']}")

    rows = []
    for run in loaded:
        lenses = _lens_scores(run, n_boot=n_boot, ci=ci, seed=seed)
        is_base = run["name"] == base["name"]
        if is_base:
            dscore = dhi = None
            n_shared = len(run["vp"])
        else:
            dscore, dhi, n_shared = _paired_delta(base, run, n_boot=n_boot, ci=ci, seed=seed)
        row = {
            "label": run["label"],
            "is_baseline": is_base,
            **run["knobs"],
            "official": lenses["official"]["score"].point,
            "official_lo": lenses["official"]["score"].lo,
            "official_hi": lenses["official"]["score"].hi,
            "balanced": lenses["balanced"]["score"].point,
            "test_matched": lenses["test_matched"]["score"].point,
            "high_occ_err": lenses["official"]["high_occ_err"].point,
            "delta_vs_base": None if is_base else dscore.point,
            "delta_lo": None if is_base else dscore.lo,
            "delta_hi": None if is_base else dscore.hi,
            "significance": "baseline" if is_base else _significant(dscore),
            "n_shared": n_shared,
            "name": run["name"],
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("official").reset_index(drop=True)
    meta = {"baseline": base["name"], "n_boot": n_boot, "ci": ci, "pred_col": pred_col}
    return df, meta


def _to_markdown(df: pd.DataFrame, meta: dict) -> str:
    lines = [
        "# Experiment comparison (paired-Δ significance)",
        "",
        f"- Baseline: `{meta['baseline']}`",
        f"- Paired bootstrap: {meta['n_boot']} resamples, {int(meta['ci'] * 100)}% CI, "
        f"prediction column `{meta['pred_col']}`, group-cluster unit.",
        "- `significance` is the paired Δscore (run − baseline): **better** = CI below 0, "
        "**worse** = CI above 0, **ns** = CI spans 0 (indistinguishable).",
        "",
        "| run | score [95% CI] | balanced | test | high-occ | Δ vs base [95% CI] | sig |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        sci = f"{r['official']:.6f} [{r['official_lo']:.6f}, {r['official_hi']:.6f}]"
        if r["is_baseline"]:
            delta = "— (baseline)"
        else:
            delta = f"{r['delta_vs_base']:+.6f} [{r['delta_lo']:+.6f}, {r['delta_hi']:+.6f}]"
        lines.append(
            f"| {r['label']} | {sci} | {r['balanced']:.6f} | {r['test_matched']:.6f} "
            f"| {r['high_occ_err']:.5f} | {delta} | {r['significance']} |"
        )
    return "\n".join(lines) + "\n"


def _forest_plot(df: pd.DataFrame, meta: dict, path: Path) -> None:
    sub = df[~df["is_baseline"]].copy()
    if sub.empty:
        return
    y = np.arange(len(sub))
    point = sub["delta_vs_base"].to_numpy(dtype=float)
    lo = sub["delta_lo"].to_numpy(dtype=float)
    hi = sub["delta_hi"].to_numpy(dtype=float)
    xerr = np.array([point - lo, hi - point])
    colors = [
        "#2ca02c" if s == "better" else ("#d62728" if s == "worse" else "#888")
        for s in sub["significance"]
    ]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(sub) + 2))
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(point, y, xerr=xerr, fmt="o", ecolor="#999", capsize=3, linestyle="none")
    for i, c in enumerate(colors):
        ax.plot(point[i], y[i], "o", color=c, markersize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel("Δ score vs baseline (negative = better)")
    ax.set_title(f"Paired Δscore vs {meta['baseline'][:30]} (95% CI)", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _discover(root: Path, pattern: str) -> list[Path]:
    return sorted(d for d in root.glob(pattern) if d.is_dir())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare experiments with CIs and a paired-Δ significance test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--runs", nargs="*", default=None, metavar="DIR", help="Experiment dirs.")
    p.add_argument("--experiments-root", default=None, metavar="DIR")
    p.add_argument("--glob", default="*", metavar="PATTERN", help="Glob under --experiments-root.")
    p.add_argument("--baseline", default=None, help="Run name/label to compare against.")
    p.add_argument("--pred-col", default="pred_clipped", choices=["pred_clipped", "pred_raw"])
    p.add_argument("--out-dir", default="tmp/comparison_reports/latest", metavar="DIR")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.runs and not args.experiments_root:
        p.error("provide --runs or --experiments-root")
    return args


def main() -> None:
    args = parse_args()
    if args.runs:
        runs = [Path(r) for r in args.runs]
    else:
        runs = _discover(Path(args.experiments_root), args.glob)
    if not runs:
        raise SystemExit("[compare] no run directories matched")

    df, meta = compare(
        runs,
        args.baseline,
        pred_col=args.pred_col,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "comparison.md").write_text(_to_markdown(df, meta), encoding="utf-8")
    _forest_plot(df, meta, out_dir / "delta_forest.png")

    print(f"[compare] {len(df)} runs vs baseline {meta['baseline']}")
    print(f"[compare] wrote {out_dir / 'comparison.csv'}")
    print(f"[compare] wrote {out_dir / 'comparison.md'}")
    print(f"[compare] wrote {out_dir / 'delta_forest.png'}")
    sig = df[df["significance"].isin(["better", "worse"])]
    if len(sig):
        print("[compare] significant vs baseline:")
        for _, r in sig.iterrows():
            print(f"    {r['label']}: Δ={r['delta_vs_base']:+.6f} ({r['significance']})")
    else:
        print("[compare] no run differs from baseline at this CI (all 'ns').")


if __name__ == "__main__":
    main()
