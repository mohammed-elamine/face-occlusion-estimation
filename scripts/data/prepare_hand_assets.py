#!/usr/bin/env python
"""Prepare hand occluder assets from a local 11k Hands download.

Segments hands off their white background, auto-derives palm/wrist anchors, and
writes RGBA cut-outs + a manifest the hand occluder fits onto faces.

**Licensing.** 11k Hands images are released for "reasonable academic fair use"
(not a redistribution licence), so the produced assets are written to a
git-ignored directory by default (``assets_private/occluders/hands``) and are NOT
committed — only this script is. Each user generates their own assets locally.

Acquisition (one-off, ~632 MB):
    # download the "Hands.zip" from https://sites.google.com/view/11khands
    # (Google Drive) and the HandInfo.csv metadata, then:
    python -m scripts.data.prepare_hand_assets \
        --hands-dir /path/to/Hands --metadata /path/to/HandInfo.csv --num 50
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
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_OUTPUT_DIR = Path("assets_private/occluders/hands")


def segment_white_background(rgb: np.ndarray) -> np.ndarray:
    """Boolean hand mask from a hand photographed on a uniform white background.

    A pixel is background when it is bright AND nearly grey (low saturation).
    The largest connected component is kept and holes are filled, so stray
    specks and ring/nail highlights do not fragment the hand.
    """
    arr = rgb.astype(np.float32)
    brightness = arr.max(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    background = (brightness > 200) & (saturation < 25)
    hand = ~background

    hand_u8 = (hand.astype(np.uint8)) * 255
    if cv2 is not None:
        kernel = np.ones((5, 5), np.uint8)
        hand_u8 = cv2.morphologyEx(hand_u8, cv2.MORPH_OPEN, kernel)
        hand_u8 = cv2.morphologyEx(hand_u8, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((hand_u8 > 0).astype(np.uint8))
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hand_u8 = np.where(labels == largest, 255, 0).astype(np.uint8)
            # Fill interior holes by flood-filling the background and inverting.
            ff = hand_u8.copy()
            mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
            cv2.floodFill(ff, mask, (0, 0), 255)
            hand_u8 = hand_u8 | cv2.bitwise_not(ff)
    return hand_u8 > 0


def derive_hand_anchors(alpha: np.ndarray) -> dict[str, list[float]]:
    """Auto-derive ``palm_center`` and ``wrist_point`` from a hand mask.

    The wrist/arm enters from whichever image border the mask touches most; the
    wrist anchor is the centroid of the mask along that border. The palm centre
    is the overall mask centroid. No manual labels needed.
    """
    ys, xs = np.where(alpha)
    palm = [float(xs.mean()), float(ys.mean())]
    h, w = alpha.shape
    borders = {
        "top": alpha[0, :].sum(),
        "bottom": alpha[h - 1, :].sum(),
        "left": alpha[:, 0].sum(),
        "right": alpha[:, w - 1].sum(),
    }
    edge = max(borders, key=borders.get)
    band = max(2, int(0.06 * (h if edge in ("top", "bottom") else w)))
    if edge == "bottom":
        sub = alpha[h - band :, :]
        wy, wx = np.where(sub)
        wrist = [float(wx.mean()), float(h - band + wy.mean())]
    elif edge == "top":
        wy, wx = np.where(alpha[:band, :])
        wrist = [float(wx.mean()), float(wy.mean())]
    elif edge == "left":
        wy, wx = np.where(alpha[:, :band])
        wrist = [float(wx.mean()), float(wy.mean())]
    else:  # right
        wy, wx = np.where(alpha[:, w - band :])
        wrist = [float(w - band + wx.mean()), float(wy.mean())]
    return {"palm": palm, "wrist": wrist}


def _to_rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgba = np.dstack([rgb, (alpha.astype(np.uint8)) * 255])
    return Image.fromarray(rgba, mode="RGBA")


def _curate(meta: pd.DataFrame, num: int, seed: int) -> pd.DataFrame:
    """Sample a diverse subset: prefer clean palmar/dorsal flats, balance skin tone."""
    df = meta.copy()
    for col, bad in (("accessories", 1), ("nailPolish", 1), ("irregularities", 1)):
        if col in df.columns:
            df = df[df[col] != bad]
    if "skinColor" in df.columns:
        # Even sampling across skin tones for harmonization diversity.
        groups = [g.sample(min(len(g), num), random_state=seed) for _, g in df.groupby("skinColor")]
        df = pd.concat(groups) if groups else df
    return df.sample(min(num, len(df)), random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands-dir", required=True, type=Path, help="Local 11k Hands images.")
    parser.add_argument("--metadata", type=Path, default=None, help="HandInfo.csv (optional).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=400, help="Resize the longer side to this.")
    args = parser.parse_args()

    if args.metadata and args.metadata.exists():
        meta = pd.read_csv(args.metadata)
        name_col = "imageName" if "imageName" in meta.columns else meta.columns[-1]
        chosen = _curate(meta, args.num, args.seed)[name_col].tolist()
    else:
        chosen = sorted(p.name for p in args.hands_dir.glob("*.jpg"))[: args.num]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchors: dict[str, dict] = {}
    kept = 0
    for name in chosen:
        path = args.hands_dir / name
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        scale = args.size / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        rgb = np.asarray(img)
        alpha = segment_white_background(rgb)
        if alpha.sum() < 0.05 * alpha.size:  # failed segmentation
            continue
        out_name = f"hand_{kept:03d}.png"
        _to_rgba(rgb, alpha).save(args.output_dir / out_name)
        anchors[out_name.replace(".png", "")] = {
            "template": out_name,
            "size": list(img.size),
            "points": derive_hand_anchors(alpha),
        }
        kept += 1

    (args.output_dir / "anchors.json").write_text(json.dumps(anchors, indent=2))
    (args.output_dir / "PROVENANCE.md").write_text(
        "# Hand occluder assets\n\n"
        "Generated locally from the **11k Hands** dataset "
        "(https://sites.google.com/view/11khands, Afifi 2019), which is released "
        "for *reasonable academic fair use* — NOT a redistribution licence. These "
        "files are therefore git-ignored and not committed; regenerate them with "
        "`scripts/data/prepare_hand_assets.py`. Cite Afifi (2019) when using.\n"
    )
    print(f"[hands] wrote {kept} hand assets to {args.output_dir}")


if __name__ == "__main__":
    main()
