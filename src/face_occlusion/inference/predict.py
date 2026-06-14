"""Inference helpers that keep predictions tied to image metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from face_occlusion.calibration import IsotonicMapping


@torch.no_grad()
def predict_dataframe(
    model,
    loader: DataLoader,
    device: str = "cpu",
    recalibration: IsotonicMapping | None = None,
) -> pd.DataFrame:
    """Run the model over ``loader`` and return one row per image.

    When ``recalibration`` is given, the monotonic map is applied to the *raw* prediction
    **before** clipping (so the calibrator sees the model's true under-prediction); the
    untouched model output is still kept in ``pred_raw``. ``recalibration=None`` (default)
    reproduces the original behaviour byte-for-byte.
    """
    model.eval().to(device)
    rows: list[dict] = []
    for batch in loader:
        images = batch["image"].to(device)
        # Models return a structured OcclusionModelOutput; the regression
        # score lives on ``y_pred``.
        preds = model(images).y_pred.detach().cpu().numpy().reshape(-1)
        recal = recalibration.apply(preds) if recalibration is not None else preds
        image_ids = list(batch["image_id"])
        filenames = list(batch["filename"])
        paths = list(batch["path"])
        databases = list(batch["database"])
        source_subfolders = list(batch["source_subfolder"])
        group_ids = list(batch["group_id"])
        face_ids = batch["face_id"].detach().cpu().numpy().reshape(-1)
        # One output row per image keeps predictions easy to join with metadata later.
        for i, pid in enumerate(image_ids):
            row = {
                "image_id": pid,
                "filename": filenames[i],
                "path": paths[i],
                "pred_raw": float(preds[i]),
                "pred_clipped": float(np.clip(recal[i], 0.0, 1.0)),
                "database": databases[i],
                "source_subfolder": source_subfolders[i],
                "group_id": group_ids[i],
                "face_id": int(face_ids[i]),
            }
            if recalibration is not None:
                row["pred_recal"] = float(recal[i])
            rows.append(row)
    return pd.DataFrame(rows)
