# Analysis Scripts

Post-training diagnostics that work from saved experiment outputs.

```bash
python -m scripts.analysis.analyze_val_predictions \
  --experiment-dir outputs/experiments/<run_id>
```

Reports include summary metrics, grouped metrics, worst-error tables, plots,
and optional image grids.
