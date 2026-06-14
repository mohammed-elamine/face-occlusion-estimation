"""Landmark-driven realistic hand occluder (Hand2Face-style cut-paste).

Reuses the mask pipeline's pattern (load assets → fit onto landmarks → composite)
but with two differences suited to hands:

* a **similarity transform** (translate + rotate + uniform scale, no shear) from
  two anchors — ``palm`` and ``wrist`` — so the hand keeps its natural shape;
* **anatomical placements**: the hand is placed at a plausible spot (chin, mouth,
  cheek, forehead) and oriented so the **wrist exits the frame** (the arm comes
  from outside — the strongest realism cue), then **skin-tone matched** to the face.

Hand assets are produced locally by ``scripts/data/prepare_hand_assets.py`` (from
11k Hands; not redistributed). Returns an empty occluder when assets or landmarks
are missing, so the generator's accept/reject simply retries or fails the view.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .synthetic_compositing import CompositingConfig, composite_occluder

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_DEFAULT_ASSET_DIR = Path("assets_private/occluders/hands")

_CHIN = 152
_NOSE_BRIDGE = 168

# Plausible hand-over-face placements: (name, weight). Target points + wrist-exit
# directions are computed from landmarks per face in ``_placement_targets``.
_PLACEMENTS = ("chin", "mouth", "left_cheek", "right_cheek", "forehead")
_PLACEMENT_WEIGHTS = (0.30, 0.25, 0.18, 0.18, 0.09)


@lru_cache(maxsize=4)
def load_hand_assets(asset_dir: str | None = None) -> dict[str, tuple[np.ndarray, dict]]:
    """Load ``{name: (rgba_array, {'palm':[x,y], 'wrist':[x,y]})}`` for hand cut-outs."""
    d = Path(asset_dir) if asset_dir else _DEFAULT_ASSET_DIR
    manifest = d / "anchors.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No hand assets at {d}. Generate them with scripts.data.prepare_hand_assets."
        )
    anchors = json.loads(manifest.read_text())
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for name, info in anchors.items():
        rgba = np.asarray(Image.open(d / info["template"]).convert("RGBA"))
        out[name] = (rgba, info["points"])
    if not out:
        raise ValueError(f"No hand assets found in {d}")
    return out


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 matrix mapping src[0]->dst[0], src[1]->dst[1] (rotation+scale+translation)."""
    s = src[1] - src[0]
    d = dst[1] - dst[0]
    s_len = float(np.hypot(*s)) + 1e-6
    scale = float(np.hypot(*d)) / s_len
    ang = np.arctan2(d[1], d[0]) - np.arctan2(s[1], s[0])
    c, sn = np.cos(ang) * scale, np.sin(ang) * scale
    rot = np.array([[c, -sn], [sn, c]], dtype=np.float32)
    t = dst[0] - rot @ src[0]
    return np.array([[rot[0, 0], rot[0, 1], t[0]], [rot[1, 0], rot[1, 1], t[1]]], dtype=np.float32)


def _placement_targets(
    landmarks: np.ndarray, region_masks: dict[str, np.ndarray], image_size: tuple[int, int]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-placement ``(target_point, wrist_exit_unit_vector)`` from the face geometry."""
    lm = np.asarray(landmarks, dtype=np.float32)
    xmin, xmax = float(lm[:, 0].min()), float(lm[:, 0].max())
    ymin, ymax = float(lm[:, 1].min()), float(lm[:, 1].max())
    cx = 0.5 * (xmin + xmax)
    face_h = ymax - ymin
    chin = lm[_CHIN]
    bridge = lm[_NOSE_BRIDGE]

    def centroid(name: str, fallback: np.ndarray) -> np.ndarray:
        m = region_masks.get(name)
        if m is None or not m.any():
            return fallback
        ys, xs = np.where(m)
        return np.array([xs.mean(), ys.mean()], dtype=np.float32)

    mouth = centroid("mouth", np.array([cx, chin[1] - 0.18 * face_h], dtype=np.float32))
    forehead = np.array([cx, max(0.0, bridge[1] - 0.28 * face_h)], dtype=np.float32)
    left_cheek = np.array([xmin + 0.28 * (xmax - xmin), mouth[1]], dtype=np.float32)
    right_cheek = np.array([xmax - 0.28 * (xmax - xmin), mouth[1]], dtype=np.float32)

    def unit(v: tuple[float, float]) -> np.ndarray:
        a = np.array(v, dtype=np.float32)
        return a / (np.linalg.norm(a) + 1e-6)

    # exit = direction the wrist/arm leaves toward (off the face, to a border).
    return {
        "chin": (chin.astype(np.float32), unit((0.0, 1.0))),
        "mouth": (mouth, unit((0.0, 1.0))),
        "left_cheek": (left_cheek, unit((-1.0, 0.4))),
        "right_cheek": (right_cheek, unit((1.0, 0.4))),
        "forehead": (forehead, unit((0.0, -1.0))),
    }


def fit_hand(
    image_size: tuple[int, int],
    *,
    target: np.ndarray,
    exit_dir: np.ndarray,
    hand_length: float,
    asset: tuple[np.ndarray, dict],
    rng: np.random.Generator,
    flip: bool = False,
) -> Image.Image:
    """Place a hand cut-out so its palm sits at ``target`` and its wrist exits along
    ``exit_dir``, scaled to ``hand_length``. Returns an RGBA layer at image size."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv (cv2) is required for hand fitting")
    width, height = image_size
    rgba, anchors = asset
    if flip:
        rgba = rgba[:, ::-1].copy()
        tw = rgba.shape[1]
        palm = np.array([tw - 1 - anchors["palm"][0], anchors["palm"][1]], dtype=np.float32)
        wrist = np.array([tw - 1 - anchors["wrist"][0], anchors["wrist"][1]], dtype=np.float32)
    else:
        palm = np.array(anchors["palm"], dtype=np.float32)
        wrist = np.array(anchors["wrist"], dtype=np.float32)

    # Small orientation jitter for variety.
    ang = np.deg2rad(rng.uniform(-12.0, 12.0))
    c, s = np.cos(ang), np.sin(ang)
    exit_j = np.array([c * exit_dir[0] - s * exit_dir[1], s * exit_dir[0] + c * exit_dir[1]])
    dst = np.stack([target, target + exit_j * float(hand_length)]).astype(np.float32)
    src = np.stack([palm, wrist]).astype(np.float32)

    matrix = _similarity_transform(src, dst)
    warped = cv2.warpAffine(
        rgba,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA")


def build_realistic_hand_sampler(
    *,
    asset_dir: str | None = None,
    hand_names: tuple[str, ...] | None = None,
    compositing: CompositingConfig | None = None,
):
    """Build an occluder sampler ``(image, rng, scale, ctx) -> (image, mask, info)``.

    ``ctx`` must expose ``.landmarks`` and ``.region_masks``. The hand is placed at
    a plausible spot with the wrist exiting the frame, sized from ``scale`` (bigger
    ⇒ heavier occlusion, reaching the strong band), and skin-tone matched to the
    face (reference = the ``cheeks`` region).
    """
    assets = load_hand_assets(asset_dir)
    if hand_names:
        assets = {k: v for k, v in assets.items() if k in hand_names}
        if not assets:
            raise ValueError(f"None of {hand_names} are available hand assets")
    names = tuple(assets)
    # Hands look most natural skin-matched; default compositing turns that on.
    comp_cfg = compositing or CompositingConfig(color_match=True, harmonize=False)

    def sampler(image, rng, scale, ctx):
        landmarks = getattr(ctx, "landmarks", None)
        region_masks = getattr(ctx, "region_masks", {}) or {}
        empty = np.zeros((image.height, image.width), dtype=bool)
        if landmarks is None or len(landmarks) <= _NOSE_BRIDGE:
            return image, empty, {"hand_failed": "no_landmarks"}

        targets = _placement_targets(landmarks, region_masks, image.size)
        placement = str(rng.choice(_PLACEMENTS, p=_PLACEMENT_WEIGHTS))
        target, exit_dir = targets[placement]
        name = names[int(rng.integers(0, len(names)))]

        lm = np.asarray(landmarks, dtype=np.float32)
        face_w = float(lm[:, 0].max() - lm[:, 0].min())
        # Map scale (0.45 mild … 2.2 strong) → hand length ~ 0.4–1.0 × face width.
        frac = 0.4 + 0.6 * float(np.clip((float(scale) - 0.45) / (2.2 - 0.45), 0.0, 1.0))
        hand_length = frac * face_w
        flip = placement == "right_cheek" or (placement != "left_cheek" and rng.random() < 0.5)

        occ_rgba = fit_hand(
            image.size,
            target=target,
            exit_dir=exit_dir,
            hand_length=hand_length,
            asset=assets[name],
            rng=rng,
            flip=flip,
        )
        reference = region_masks.get("cheeks")
        out, mask = composite_occluder(image, occ_rgba, rng, comp_cfg, reference_mask=reference)
        return out, mask, {"hand_asset": name, "placement": placement, "hand_frac": frac}

    return sampler
