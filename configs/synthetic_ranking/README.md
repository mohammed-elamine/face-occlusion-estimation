# Synthetic monotonic ranking ablation (Stage 4)

Tests whether a RankNet loss on synthetic `clean < mild < strong` views improves
high-occlusion regression without hurting calibration. The two configs differ
**only** in the ranking toggle (and the synthetic views it needs), so the delta
isolates the ranking effect.

| Config | Ranking | Synthetic views |
|---|---|---|
| `00_baseline.yaml` | off | off |
| `01_ranking_w010.yaml` | on (λ=0.1, warmup 2) | on (cache) |

## 1. Gate the coverage first

Synthetic ranking can only help where MediaPipe succeeds. Run the coverage audit
on the high-occlusion rows and read the bin×gender table before investing:

```bash
python -m scripts.analysis.generate_synthetic_occlusion_audit \
  --config configs/baseline.yaml --target-min 0.40 --num-samples 300 \
  --coverage-only --output-dir tmp/gate_audit
```

If MediaPipe-valid rates collapse in `0.40–0.60` / `0.60–1.00`, ranking will
under-cover the hard cases — that is the finding, not a number to chase.

## 2. Build the cache

`01_ranking_w010.yaml` points at `synthetic_occlusion.cache_dir:
data/synthetic_cache/v1`. Build it once (TRAIN rows only, balanced by
bin×gender):

```bash
python -m scripts.data.build_synthetic_cache \
  --config configs/baseline.yaml \
  --cache-dir data/synthetic_cache/v1 \
  --target-min 0.10 --max-per-bin-gender 200
```

The printed coverage table shows how many valid pairs landed per bin×gender —
check the male high-occlusion cells are not starved.

## 3. Train both and compare

```bash
python -m scripts.training.train --config configs/synthetic_ranking/00_baseline.yaml
python -m scripts.training.train --config configs/synthetic_ranking/01_ranking_w010.yaml
```

Then compare with confidence intervals on **both** splits (never raw tail
deltas):

```bash
python -m scripts.analysis.bootstrap_metrics \
  --predictions outputs/experiments/<ranking_run>/predictions/val_predictions.csv \
  --unit group        # honest CI under identity leakage
```

## Accept / reject

Accept ranking only if it **lowers real high-occlusion error within CI** on both
splits without regressing low-bin calibration. Watch:

- `val/high_occ_0.40_1.00_*` (from `bootstrap_metrics`: `high_occ_err`) — must improve.
- `val/bin_0.00_0.05_weighted_mse` — must NOT regress (calibration guard; ranking
  lands on the calibrated head).
- `train/rank_ordering_acc` — should rise toward 1.0; if it stays low, the
  ordering signal is too noisy to help.

Ordering accuracy improving while real high-occ error does not is the classic
"learned the artifact, not the occlusion" failure — reject in that case.
