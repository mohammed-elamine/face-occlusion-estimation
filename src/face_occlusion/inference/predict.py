"""Inference helpers that keep predictions tied to image metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_dataframe(model, loader: DataLoader, device: str = "cpu") -> pd.DataFrame:
    model.eval().to(device)
    rows: list[dict] = []
    for batch in loader:
        images = batch["image"].to(device)
        preds = model(images).detach().cpu().numpy().reshape(-1)
        genders = batch["gender"].detach().cpu().numpy().reshape(-1)
        image_ids = list(batch["image_id"])
        paths = list(batch["path"])
        # One output row per image keeps predictions easy to join with metadata later.
        for i, pid in enumerate(image_ids):
            rows.append(
                {
                    "image_id": pid,
                    "path": paths[i],
                    "gender": float(genders[i]),
                    "pred_raw": float(preds[i]),
                    "pred_clipped": float(np.clip(preds[i], 0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)
