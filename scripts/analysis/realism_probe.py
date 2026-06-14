#!/usr/bin/env python
"""Measure the synthetic-vs-real occlusion gap (a realism probe).

If synthetic occluders look obviously different from real ones, the ranking loss
can learn the *artifact* instead of occlusion — a shortcut that won't transfer to
the real test set. This script quantifies that gap: it trains a simple classifier
to tell **synthetic-occluded** faces from **real-occluded** faces and reports the
ROC AUC.

    AUC ~ 0.5  -> the two are hard to tell apart (good: small gap)
    AUC ~ 1.0  -> trivially separable (bad: big gap / shortcut risk)

The synthetic occluders come from the generator described by ``--config`` (so the
same script measures geometric vs realistic occluders just by changing the
config). Real and synthetic sets are matched by occlusion level so the classifier
keys on *style*, not on *how covered* the face is.

Example
-------
    python -m scripts.analysis.realism_probe \
        --config configs/synthetic_ranking/02_ranking_realistic_masks.yaml \
        --num-per-class 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from face_occlusion.data.metadata import add_path_metadata
from face_occlusion.data.normalize import normalize_target
from face_occlusion.data.synthetic_occlusion import build_generator_from_config
from face_occlusion.utils import load_config


def probe_auc(features: np.ndarray, labels: np.ndarray, *, seed: int = 0) -> float:
    """ROC AUC of a logistic-regression synthetic-vs-real classifier.

    ``features`` is ``(N, D)``, ``labels`` is ``(N,)`` with 1 = real, 0 = synthetic.
    A held-out split keeps the estimate honest. Pure + deterministic, so it is
    unit-tested without any image model.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_tr, y_tr)
    scores = clf.predict_proba(x_te)[:, 1]
    return float(roc_auc_score(y_te, scores))


def _build_feature_extractor(backbone: str):
    import timm
    import torch

    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)

    @torch.no_grad()
    def extract(images: list[Image.Image]) -> np.ndarray:
        feats = []
        for i in range(0, len(images), 64):
            batch = torch.stack([transform(im) for im in images[i : i + 64]])
            feats.append(model(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    return extract


def _load_crops(df: pd.DataFrame, image_root: Path, image_col: str, size: int) -> list[Image.Image]:
    out = []
    for rel in df[image_col]:
        with Image.open(image_root / str(rel)) as im:
            out.append(im.convert("RGB").resize((size, size), Image.BILINEAR))
    return out


def _make_synthetic_crops(clean_crops, generator, seed) -> list[Image.Image]:
    """Generate strong synthetic-occluded views from clean faces (valid only)."""
    out = []
    for i, img in enumerate(clean_crops):
        pair = generator.generate_pair(img, rng=np.random.default_rng([seed, i]))
        if pair.valid and pair.strong is not None:
            out.append(pair.strong.image)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-per-class", type=int, default=300)
    parser.add_argument(
        "--real-min", type=float, default=0.20, help="Min occlusion for the real set."
    )
    parser.add_argument(
        "--clean-max", type=float, default=0.05, help="Max occlusion for synthetic sources."
    )
    parser.add_argument("--backbone", default="resnet18", help="Frozen feature extractor (timm).")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generator = build_generator_from_config(cfg)
    if generator is None:
        raise ValueError("synthetic_occlusion.enabled must be true in the config to probe it.")

    image_root = Path(cfg.data.image_root)
    image_col = cfg.data.image_col
    df = add_path_metadata(pd.read_csv(cfg.data.train_csv), filename_col=cfg.data.id_col)
    df["y"] = normalize_target(df[cfg.data.target_col], cfg.data.target_scale)
    rng = np.random.default_rng(args.seed)

    real_pool = df[df["y"] >= args.real_min]
    real_df = real_pool.sample(min(args.num_per_class, len(real_pool)), random_state=args.seed)
    real_crops = _load_crops(real_df, image_root, image_col, args.size)

    # Oversample clean sources because not every one yields a valid synthetic view.
    clean_pool = df[df["y"] <= args.clean_max]
    n_src = min(len(clean_pool), int(args.num_per_class * 2.5))
    clean_df = clean_pool.sample(n_src, random_state=args.seed + 1)
    clean_crops = _load_crops(clean_df, image_root, image_col, args.size)
    synth_crops = _make_synthetic_crops(clean_crops, generator, args.seed)[: args.num_per_class]

    n = min(len(real_crops), len(synth_crops))
    real_crops, synth_crops = real_crops[:n], synth_crops[:n]
    print(f"[probe] real-occluded: {len(real_crops)}  synthetic-occluded: {len(synth_crops)}")
    if n < 20:
        raise ValueError("Too few samples to probe; lower thresholds or raise --num-per-class.")

    extract = _build_feature_extractor(args.backbone)
    feats = extract(real_crops + synth_crops)
    labels = np.array([1] * len(real_crops) + [0] * len(synth_crops))
    perm = rng.permutation(len(labels))
    auc = probe_auc(feats[perm], labels[perm], seed=args.seed)

    print(f"\n[probe] synthetic-vs-real AUC = {auc:.3f}")
    print("  ~0.5 = indistinguishable (small gap, good); ~1.0 = easily separable (big gap).")
    print("  Note: real occluders are varied (masks/hands/hair) while synthetic are masks,")
    print("  so some separability is expected; track the trend as realism improves.")


if __name__ == "__main__":
    main()
