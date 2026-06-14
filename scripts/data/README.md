# Data Scripts

Utilities for inspecting data and managing validation splits.

```bash
python -m scripts.data.validate_data --config configs/baseline.yaml
python -m scripts.data.make_split --config configs/baseline.yaml
```

Splits are saved so multiple experiments can use the same train/validation
protocol.
