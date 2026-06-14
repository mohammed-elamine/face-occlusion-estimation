"""Face-aware synthetic occlusion generator (Stage 3).

The generator produces a *pair* of synthetic views — mild and strong — for a
single aligned face crop. Synthetic samples deliberately receive **no
regression label**: only severity proxies are exposed via metadata. Stage 4
will consume the resulting ordering ``original < mild < strong`` to train a
monotonic ranking head; Stage 3 only generates and audits views.

Design notes
------------
* Runtime region localisation is MediaPipe-only. If MediaPipe cannot detect a
  credible face, the synthetic pair is marked invalid; there is no geometric
  fallback for training or audit generation.
* Severity is a weighted area proxy ``ρ = Σ_r w_r · |M ∩ R_r| / |face|``
  where ``M`` is the occluder mask. ``ρ`` is *not* a label.
* Acceptance–rejection sampling rejects pastes whose ``ρ`` falls outside the
  configured ``[min, max]`` band for the requested severity level. The
  generator retries up to ``max_attempts`` times per level.
* The whole generator is RNG-deterministic when given a seeded
  ``numpy.random.Generator``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .synthetic_compositing import CompositingConfig
from .synthetic_hand_occluder import build_realistic_hand_sampler
from .synthetic_mask_occluder import build_realistic_mask_sampler

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_REGION_WEIGHTS: dict[str, float] = {
    "eyes": 1.00,
    "mouth": 0.85,
    "nose": 0.75,
    "cheeks": 0.45,
    "forehead_chin": 0.35,
    "background": 0.00,
}

DEFAULT_OCCLUDER_TYPES: tuple[str, ...] = (
    "mask_like_lower_face",
    "sunglasses_like_eyes",
    "random_face_rectangle",
    "random_textured_polygon",
    "blurred_patch",
)

DEFAULT_SEVERITY_BANDS: dict[str, tuple[float, float]] = {
    "mild": (0.05, 0.15),
    "strong": (0.35, 0.60),
}

BACKGROUND_OVERLAP_WARNING_THRESHOLD = 0.35
HIGH_ATTEMPT_WARNING_FRACTION = 0.80
IMPORTANT_REGION_OVERLAP_WARNING_THRESHOLD = 0.05

REQUIRED_REGION_MASKS: tuple[str, ...] = (
    "face",
    "left_eye",
    "right_eye",
    "eyes",
    "nose",
    "mouth",
    "lower_face",
    "cheeks",
    "forehead_chin",
    "background",
)

OVERLAP_METRIC_KEYS: tuple[str, ...] = (
    "face_overlap_ratio",
    "background_overlap_ratio",
    "important_region_overlap",
    "eye_overlap_ratio",
    "mouth_overlap_ratio",
    "nose_overlap_ratio",
    "lower_face_overlap_ratio",
    "occluder_area_ratio",
    "occluder_face_area_ratio",
    "weighted_severity",
)

OVERLAP_FLAG_KEYS: tuple[str, ...] = (
    "mostly_background_occlusion",
    "low_important_region_overlap",
    "high_attempt_count",
)

MEDIAPIPE_INSTALL_MESSAGE = (
    "MediaPipe is required for synthetic_occlusion.region_provider='mediapipe'. "
    "Install it with `uv sync --extra synthetic` or `pip install mediapipe`."
)

MEDIAPIPE_MODEL_MESSAGE = (
    "MediaPipe Face Landmarker requires a .task model asset. Download it with "
    "`curl -L -o tmp/mediapipe/face_landmarker.task "
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task` or set "
    "FACE_OCCLUSION_MEDIAPIPE_FACE_LANDMARKER to the model path."
)

DEFAULT_MEDIAPIPE_MODEL_PATHS: tuple[Path, ...] = (
    Path("tmp/mediapipe/face_landmarker.task"),
    Path("assets/mediapipe/face_landmarker.task"),
    Path("data/mediapipe/face_landmarker.task"),
)

# MediaPipe Face Mesh landmark indices. The face oval list is ordered so it can
# be drawn directly as a polygon; smaller facial parts use convex hulls.
_FACE_OVAL_LANDMARKS = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

_LEFT_EYE_LANDMARKS = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)

_RIGHT_EYE_LANDMARKS = (
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)

_MOUTH_LANDMARKS = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
)

_NOSE_LANDMARKS = (
    1,
    2,
    4,
    5,
    6,
    19,
    45,
    48,
    64,
    94,
    97,
    98,
    115,
    129,
    131,
    168,
    195,
    197,
    220,
    275,
    278,
    294,
    326,
    327,
    344,
    358,
    360,
    419,
    440,
)

_MAX_REGION_LANDMARK_INDEX = max(
    _FACE_OVAL_LANDMARKS
    + _LEFT_EYE_LANDMARKS
    + _RIGHT_EYE_LANDMARKS
    + _MOUTH_LANDMARKS
    + _NOSE_LANDMARKS
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FaceRegionResult:
    """Output contract for face-region providers.

    ``landmarks`` contains pixel-space MediaPipe landmark coordinates with
    shape ``(N, 2)`` when detection succeeds. Failed results keep ``masks``
    empty and set a compact ``failure_reason`` for diagnostics.
    """

    valid: bool
    masks: dict[str, np.ndarray]
    landmarks: np.ndarray | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticOcclusionView:
    """One synthetic view (mild OR strong) with its occluder mask."""

    image: Image.Image
    occluder_mask: np.ndarray  # bool array, (H, W)
    severity: float
    level: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticOcclusionPair:
    """Pair of synthetic views derived from one source image."""

    original: Image.Image
    mild: SyntheticOcclusionView | None
    strong: SyntheticOcclusionView | None
    region_masks: dict[str, np.ndarray]
    valid: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------


def build_aligned_face_region_masks(size: int) -> dict[str, np.ndarray]:
    """Legacy/test-only approximate masks for an aligned face crop.

    Coordinates are tuned for the project's 224×224 aligned crops; they scale
    linearly with ``size``. Returned masks are ``bool`` arrays of shape
    ``(size, size)``. Runtime synthetic generation must not fall back to this
    helper when MediaPipe fails.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")

    s = size

    def _ellipse(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        ys, xs = np.mgrid[0:s, 0:s]
        return (((xs - cx * s) / (rx * s)) ** 2 + ((ys - cy * s) / (ry * s)) ** 2) <= 1.0

    def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        m = np.zeros((s, s), dtype=bool)
        m[int(y0 * s) : int(y1 * s), int(x0 * s) : int(x1 * s)] = True
        return m

    face = _ellipse(0.50, 0.55, 0.42, 0.55)
    left_eye = _ellipse(0.36, 0.42, 0.10, 0.05)
    right_eye = _ellipse(0.64, 0.42, 0.10, 0.05)
    eyes = (left_eye | right_eye) & face
    nose = _ellipse(0.50, 0.55, 0.07, 0.13) & face
    mouth = _ellipse(0.50, 0.75, 0.14, 0.07) & face
    lower_face = _rect(0.0, 0.55, 1.0, 0.98) & face
    forehead = _rect(0.0, 0.0, 1.0, 0.32) & face
    chin = _rect(0.0, 0.88, 1.0, 1.0) & face
    forehead_chin = forehead | chin
    cheeks = face & ~(eyes | nose | mouth | forehead_chin)
    background = ~face

    return {
        "face": face,
        "left_eye": left_eye & face,
        "right_eye": right_eye & face,
        "eyes": eyes,
        "nose": nose,
        "mouth": mouth,
        "lower_face": lower_face,
        "cheeks": cheeks,
        "forehead_chin": forehead_chin,
        "background": background,
    }


def _invalid_region_result(
    failure_reason: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> FaceRegionResult:
    return FaceRegionResult(
        valid=False,
        masks={},
        landmarks=None,
        failure_reason=failure_reason,
        metadata=metadata or {},
    )


def _resolve_mediapipe_model_asset_path(path: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env_path = os.environ.get("FACE_OCCLUSION_MEDIAPIPE_FACE_LANDMARKER")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(DEFAULT_MEDIAPIPE_MODEL_PATHS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or RGB numpy array to a uint8 RGB array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise ValueError(f"Expected an RGB image array, got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the monotonic-chain convex hull of ``points``."""
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def _cross(
        o: tuple[int, int],
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_mask(
    height: int,
    width: int,
    polygon: Iterable[tuple[int, int]],
    *,
    expand_px: int = 0,
) -> np.ndarray:
    points = list(polygon)
    if len(points) < 3:
        return np.zeros((height, width), dtype=bool)
    mask_img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_img).polygon(points, fill=255)
    if expand_px > 0:
        # A tiny dilation keeps thin landmark hulls, especially eyes, visible.
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=2 * expand_px + 1))
    return np.asarray(mask_img) > 0


def _mask_from_landmarks(
    points: np.ndarray,
    indices: Iterable[int],
    height: int,
    width: int,
    *,
    expand_px: int = 0,
    ordered: bool = False,
) -> np.ndarray:
    selected = [(int(points[i, 0]), int(points[i, 1])) for i in indices if 0 <= i < points.shape[0]]
    polygon = selected if ordered else _convex_hull(selected)
    return _polygon_mask(height, width, polygon, expand_px=expand_px)


class MediaPipeFaceRegionProvider:
    """Build face and semantic region masks from MediaPipe Face Mesh landmarks.

    The provider deliberately returns an invalid result instead of approximate
    geometry when detection or mask sanity checks fail. Region definitions are a
    pragmatic first pass: face/eyes/mouth/nose come from landmark polygons, and
    lower-face, cheeks, and forehead/chin are derived from landmark-relative
    horizontal bands inside the detected face mask.
    """

    def __init__(
        self,
        *,
        model_asset_path: str | Path | None = None,
        min_detection_confidence: float = 0.50,
        min_face_area_ratio: float = 0.10,
        max_face_area_ratio: float = 0.90,
        max_out_of_bounds_fraction: float = 0.05,
    ) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise ImportError(MEDIAPIPE_INSTALL_MESSAGE) from exc

        self.min_face_area_ratio = float(min_face_area_ratio)
        self.max_face_area_ratio = float(max_face_area_ratio)
        self.max_out_of_bounds_fraction = float(max_out_of_bounds_fraction)
        self._backend = (
            "solutions_face_mesh" if hasattr(mp, "solutions") else "tasks_face_landmarker"
        )
        self._face_mesh = None
        self._landmarker = None
        self._mp_image_cls = None
        self._mp_image_format = None
        if self._backend == "solutions_face_mesh":
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=float(min_detection_confidence),
            )
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as base_options_lib

            resolved_model_path = _resolve_mediapipe_model_asset_path(model_asset_path)
            if resolved_model_path is None:
                raise FileNotFoundError(MEDIAPIPE_MODEL_MESSAGE)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options_lib.BaseOptions(
                    model_asset_path=str(resolved_model_path)
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=float(min_detection_confidence),
                min_face_presence_confidence=float(min_detection_confidence),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._mp_image_cls = mp.Image
            self._mp_image_format = mp.ImageFormat

    # ------------------------------------------------------------------
    def extract(self, image: Image.Image | np.ndarray) -> FaceRegionResult:
        try:
            arr = _as_rgb_array(image)
            height, width = arr.shape[:2]
            if self._backend == "solutions_face_mesh":
                result = self._face_mesh.process(arr)
                if not result.multi_face_landmarks:
                    return _invalid_region_result("no_face_detected")
                face_landmarks = result.multi_face_landmarks[0].landmark
            else:
                mp_image = self._mp_image_cls(
                    image_format=self._mp_image_format.SRGB,
                    data=arr,
                )
                result = self._landmarker.detect(mp_image)
                if not result.face_landmarks:
                    return _invalid_region_result("no_face_detected")
                face_landmarks = result.face_landmarks[0]
            if not face_landmarks:
                return _invalid_region_result("missing_landmarks")

            normalized = np.array(
                [(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in face_landmarks],
                dtype=np.float32,
            )
            if normalized.shape[0] <= _MAX_REGION_LANDMARK_INDEX:
                return _invalid_region_result(
                    "missing_landmarks",
                    metadata={"landmark_count": int(normalized.shape[0])},
                )

            outside = (
                (normalized[:, 0] < -0.05)
                | (normalized[:, 0] > 1.05)
                | (normalized[:, 1] < -0.05)
                | (normalized[:, 1] > 1.05)
            )
            out_of_bounds_fraction = float(outside.mean())
            if out_of_bounds_fraction > self.max_out_of_bounds_fraction:
                return _invalid_region_result(
                    "region_out_of_bounds",
                    metadata={"out_of_bounds_fraction": out_of_bounds_fraction},
                )

            points = np.column_stack(
                [
                    np.clip(np.round(normalized[:, 0] * (width - 1)), 0, width - 1),
                    np.clip(np.round(normalized[:, 1] * (height - 1)), 0, height - 1),
                ]
            ).astype(np.int32)
            masks = self._build_masks(points, height, width)
            return self._validate_masks(
                masks,
                landmarks=points.astype(np.float32),
                image_size=(width, height),
                metadata={
                    "provider": "mediapipe",
                    "backend": self._backend,
                    "landmark_count": int(points.shape[0]),
                    "out_of_bounds_fraction": out_of_bounds_fraction,
                },
            )
        except Exception as exc:
            return _invalid_region_result(
                "unknown_error",
                metadata={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    def __call__(self, image: Image.Image | np.ndarray) -> FaceRegionResult:
        return self.extract(image)

    # ------------------------------------------------------------------
    def _build_masks(self, points: np.ndarray, height: int, width: int) -> dict[str, np.ndarray]:
        face = _mask_from_landmarks(
            points,
            _FACE_OVAL_LANDMARKS,
            height,
            width,
            ordered=True,
        )
        left_eye = (
            _mask_from_landmarks(
                points,
                _LEFT_EYE_LANDMARKS,
                height,
                width,
                expand_px=2,
            )
            & face
        )
        right_eye = (
            _mask_from_landmarks(
                points,
                _RIGHT_EYE_LANDMARKS,
                height,
                width,
                expand_px=2,
            )
            & face
        )
        eyes = (left_eye | right_eye) & face
        mouth = (
            _mask_from_landmarks(
                points,
                _MOUTH_LANDMARKS,
                height,
                width,
                expand_px=2,
            )
            & face
        )
        nose = (
            _mask_from_landmarks(
                points,
                _NOSE_LANDMARKS,
                height,
                width,
                expand_px=2,
            )
            & face
        )

        ys, _xs = np.mgrid[0:height, 0:width]
        eye_y = float(points[list(_LEFT_EYE_LANDMARKS + _RIGHT_EYE_LANDMARKS), 1].mean())
        mouth_y = float(points[list(_MOUTH_LANDMARKS), 1].mean())
        nose_y = float(points[list(_NOSE_LANDMARKS), 1].mean())
        forehead_cut = max(0.0, eye_y - 0.10 * height)
        chin_cut = min(float(height - 1), mouth_y + 0.08 * height)

        lower_face = face & (ys >= nose_y)
        forehead_chin = face & ((ys <= forehead_cut) | (ys >= chin_cut))
        cheeks = face & ~(eyes | nose | mouth | forehead_chin)
        background = ~face

        return {
            "face": face,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "eyes": eyes,
            "nose": nose,
            "mouth": mouth,
            "lower_face": lower_face,
            "cheeks": cheeks,
            "forehead_chin": forehead_chin,
            "background": background,
        }

    # ------------------------------------------------------------------
    def _validate_masks(
        self,
        masks: dict[str, np.ndarray],
        *,
        landmarks: np.ndarray,
        image_size: tuple[int, int],
        metadata: dict[str, Any],
    ) -> FaceRegionResult:
        width, height = image_size
        expected_shape = (height, width)
        for name in REQUIRED_REGION_MASKS:
            mask = masks.get(name)
            if mask is None or mask.shape != expected_shape or mask.dtype != bool:
                return _invalid_region_result(
                    "invalid_region_area",
                    metadata={**metadata, "invalid_region": name},
                )

        face_area = int(masks["face"].sum())
        face_area_ratio = face_area / float(height * width)
        metadata = {**metadata, "face_area_ratio": face_area_ratio}
        if face_area_ratio < self.min_face_area_ratio or face_area_ratio > self.max_face_area_ratio:
            return _invalid_region_result("invalid_face_mask", metadata=metadata)

        min_part_area = max(4, int(height * width * 0.0002))
        for name in ("left_eye", "right_eye", "eyes", "mouth", "nose"):
            area = int(masks[name].sum())
            if area < min_part_area:
                return _invalid_region_result(
                    "invalid_region_area",
                    metadata={**metadata, "invalid_region": name, "area": area},
                )
            if int(np.logical_and(masks[name], masks["face"]).sum()) == 0:
                return _invalid_region_result(
                    "invalid_region_area",
                    metadata={**metadata, "invalid_region": name, "area": area},
                )

        return FaceRegionResult(
            valid=True,
            masks=masks,
            landmarks=landmarks,
            failure_reason=None,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Severity proxy
# ---------------------------------------------------------------------------


def compute_severity(
    occluder_mask: np.ndarray,
    region_masks: dict[str, np.ndarray],
    region_weights: dict[str, float],
) -> float:
    """Weighted occluded face area / face area (clamped to ``[0, 1]``)."""
    face = region_masks["face"]
    face_area = float(face.sum())
    if face_area <= 0:
        return 0.0
    rho = 0.0
    for region, weight in region_weights.items():
        if weight == 0.0:
            continue
        region_mask = region_masks.get(region)
        if region_mask is None:
            continue
        overlap = float(np.logical_and(occluder_mask, region_mask).sum())
        rho += float(weight) * overlap / face_area
    return float(min(max(rho, 0.0), 1.0))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_occluder_overlap_metrics(
    occluder_mask: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute semantic occluder/region overlap diagnostics.

    ``important_region_overlap`` is measured as
    ``area(M ∩ (eyes ∪ mouth ∪ nose)) / area(M)``. This answers the audit
    question "how much of the pasted occluder actually lands on important face
    regions?" and makes large side/background patches easy to spot.
    """
    mask = np.asarray(occluder_mask, dtype=bool)
    image_area = float(mask.size)
    mask_area = float(mask.sum())

    empty = np.zeros_like(mask, dtype=bool)
    face = np.asarray(region_masks.get("face", empty), dtype=bool)
    background = np.asarray(region_masks.get("background", ~face), dtype=bool)
    eyes = np.asarray(region_masks.get("eyes", empty), dtype=bool)
    mouth = np.asarray(region_masks.get("mouth", empty), dtype=bool)
    nose = np.asarray(region_masks.get("nose", empty), dtype=bool)
    lower_face = np.asarray(region_masks.get("lower_face", empty), dtype=bool)
    important = (eyes | mouth | nose) & face

    face_overlap_area = float(np.logical_and(mask, face).sum())
    background_overlap_area = float(np.logical_and(mask, background).sum())
    important_overlap_area = float(np.logical_and(mask, important).sum())

    return {
        "face_overlap_ratio": _ratio(face_overlap_area, mask_area),
        "background_overlap_ratio": _ratio(background_overlap_area, mask_area),
        "important_region_overlap": _ratio(important_overlap_area, mask_area),
        "eye_overlap_ratio": _ratio(float(np.logical_and(mask, eyes).sum()), float(eyes.sum())),
        "mouth_overlap_ratio": _ratio(float(np.logical_and(mask, mouth).sum()), float(mouth.sum())),
        "nose_overlap_ratio": _ratio(float(np.logical_and(mask, nose).sum()), float(nose.sum())),
        "lower_face_overlap_ratio": _ratio(
            float(np.logical_and(mask, lower_face).sum()), float(lower_face.sum())
        ),
        "occluder_area_ratio": _ratio(mask_area, image_area),
        "occluder_face_area_ratio": _ratio(face_overlap_area, float(face.sum())),
    }


# ---------------------------------------------------------------------------
# Occluder primitives
# ---------------------------------------------------------------------------


def _random_natural_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample a plausible occluder colour (cloth/skin/dark/light)."""
    palette = np.array(
        [
            (20, 20, 20),  # near-black (cap/sunglasses)
            (40, 40, 50),  # dark grey
            (90, 80, 70),  # dark brown
            (200, 200, 210),  # light grey (mask)
            (240, 240, 245),  # white (mask)
            (60, 100, 170),  # blue (mask/scarf)
            (130, 70, 50),  # brown
            (160, 130, 100),  # tan (hand)
        ],
        dtype=np.uint8,
    )
    idx = int(rng.integers(0, len(palette)))
    base = palette[idx].astype(int)
    jitter = rng.integers(-15, 16, size=3)
    return tuple(int(np.clip(c, 0, 255)) for c in base + jitter)


def _paste_filled_polygon(
    image: Image.Image,
    polygon: list[tuple[int, int]],
    color: tuple[int, int, int],
    alpha: float,
) -> tuple[Image.Image, np.ndarray]:
    """Alpha-blend a filled polygon onto ``image``; return image + bool mask."""
    if not polygon:
        return image, np.zeros((image.height, image.width), dtype=bool)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(np.clip(alpha, 0.0, 1.0) * 255)
    draw.polygon(polygon, fill=(color[0], color[1], color[2], a))
    base = image.convert("RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    mask_img = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask_img).polygon(polygon, fill=255)
    return out, np.asarray(mask_img) > 0


def _paste_filled_ellipse(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> tuple[Image.Image, np.ndarray]:
    """Alpha-blend a filled ellipse onto ``image``; return image + bool mask."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    a = int(np.clip(alpha, 0.0, 1.0) * 255)
    ImageDraw.Draw(overlay).ellipse(bbox, fill=(color[0], color[1], color[2], a))
    base = image.convert("RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    mask_img = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask_img).ellipse(bbox, fill=255)
    return out, np.asarray(mask_img) > 0


def _apply_blur_patch(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    radius: float,
) -> tuple[Image.Image, np.ndarray]:
    """Apply a strong Gaussian blur to a rectangular patch (simulates degradation)."""
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 <= x0 or y1 <= y0:
        return image, np.zeros((image.height, image.width), dtype=bool)
    patch = image.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=radius))
    out = image.copy()
    out.paste(patch, (x0, y0))
    mask = np.zeros((image.height, image.width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return out, mask


# ---------------------------------------------------------------------------
# Occluder samplers
# ---------------------------------------------------------------------------


def _sample_mask_like_lower_face(
    image: Image.Image,
    rng: np.random.Generator,
    scale: float,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    w, h = image.size
    cx = w * 0.5
    cy = h * (0.72 + rng.uniform(-0.04, 0.04))
    rx = w * 0.22 * scale
    ry = h * 0.16 * scale
    bbox = (int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry))
    color = _random_natural_color(rng)
    alpha = float(rng.uniform(0.85, 1.0))
    out, mask = _paste_filled_ellipse(image, bbox, color, alpha)
    return out, mask, {"color": color, "alpha": alpha, "scale": float(scale)}


def _sample_sunglasses_like_eyes(
    image: Image.Image,
    rng: np.random.Generator,
    scale: float,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    w, h = image.size
    cx = w * 0.5
    cy = h * (0.42 + rng.uniform(-0.03, 0.03))
    half_w = w * 0.32 * scale
    half_h = h * 0.07 * scale
    bbox = (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))
    color = (int(rng.integers(10, 40)),) * 3
    alpha = float(rng.uniform(0.85, 1.0))
    out, mask = _paste_filled_ellipse(image, bbox, color, alpha)
    return out, mask, {"color": color, "alpha": alpha, "scale": float(scale)}


def _sample_random_face_rectangle(
    image: Image.Image,
    rng: np.random.Generator,
    scale: float,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    w, h = image.size
    # Sample within the face bbox (roughly [0.08, 0.92] × [0.05, 0.95]).
    cx = rng.uniform(0.20, 0.80) * w
    cy = rng.uniform(0.20, 0.85) * h
    half_w = rng.uniform(0.08, 0.22) * w * scale
    half_h = rng.uniform(0.08, 0.22) * h * scale
    bbox = (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))
    poly = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
    color = _random_natural_color(rng)
    alpha = float(rng.uniform(0.80, 1.0))
    out, mask = _paste_filled_polygon(image, poly, color, alpha)
    return out, mask, {"color": color, "alpha": alpha, "scale": float(scale)}


def _sample_random_textured_polygon(
    image: Image.Image,
    rng: np.random.Generator,
    scale: float,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    w, h = image.size
    cx = rng.uniform(0.25, 0.75) * w
    cy = rng.uniform(0.25, 0.85) * h
    n_vertices = int(rng.integers(5, 9))
    base_r = rng.uniform(0.10, 0.22) * min(w, h) * scale
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_vertices))
    radii = base_r * rng.uniform(0.6, 1.2, size=n_vertices)
    poly = [(int(cx + r * np.cos(a)), int(cy + r * np.sin(a))) for a, r in zip(angles, radii)]
    color = _random_natural_color(rng)
    alpha = float(rng.uniform(0.80, 1.0))
    out, mask = _paste_filled_polygon(image, poly, color, alpha)
    return out, mask, {"color": color, "alpha": alpha, "scale": float(scale)}


def _sample_blurred_patch(
    image: Image.Image,
    rng: np.random.Generator,
    scale: float,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    w, h = image.size
    cx = rng.uniform(0.25, 0.75) * w
    cy = rng.uniform(0.25, 0.85) * h
    half_w = rng.uniform(0.10, 0.22) * w * scale
    half_h = rng.uniform(0.10, 0.22) * h * scale
    bbox = (
        int(max(0, cx - half_w)),
        int(max(0, cy - half_h)),
        int(min(w, cx + half_w)),
        int(min(h, cy + half_h)),
    )
    radius = float(rng.uniform(6.0, 14.0)) * max(scale, 0.5)
    out, mask = _apply_blur_patch(image, bbox, radius)
    return out, mask, {"radius": radius, "scale": float(scale)}


_OCCLUDER_DISPATCH = {
    "mask_like_lower_face": _sample_mask_like_lower_face,
    "sunglasses_like_eyes": _sample_sunglasses_like_eyes,
    "random_face_rectangle": _sample_random_face_rectangle,
    "random_textured_polygon": _sample_random_textured_polygon,
    "blurred_patch": _sample_blurred_patch,
}

# Realistic occluders are landmark-driven and live in their own modules; they are
# valid occluder types but take a face context rather than going through the
# geometric dispatch above.
_REALISTIC_OCCLUDER_TYPES = ("realistic_mask", "realistic_hand")
_ALL_OCCLUDER_TYPES = tuple(_OCCLUDER_DISPATCH) + _REALISTIC_OCCLUDER_TYPES


@dataclass
class OccluderContext:
    """Face context passed to landmark-driven occluder samplers."""

    landmarks: np.ndarray | None
    region_masks: dict[str, np.ndarray]
    image_size: tuple[int, int]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticOcclusionGenerator:
    """Generate controlled mild/strong synthetic occlusions for one image.

    Parameters
    ----------
    severity_bands:
        Mapping ``level -> (min, max)`` for the accepted severity proxy.
    region_weights:
        Per-region multiplier used by :func:`compute_severity`.
    occluder_types:
        Subset of :data:`DEFAULT_OCCLUDER_TYPES` to sample from.
    max_attempts:
        Hard cap on acceptance-rejection retries per level.
    region_provider:
        Runtime provider name. Only ``"mediapipe"`` is supported.
    face_region_provider:
        Optional provider object used by tests; it must return
        :class:`FaceRegionResult` from ``extract(image)`` or ``__call__(image)``.
    """

    def __init__(
        self,
        *,
        severity_bands: dict[str, tuple[float, float]] | None = None,
        region_weights: dict[str, float] | None = None,
        occluder_types: Iterable[str] | None = None,
        max_attempts: int = 50,
        region_provider: str = "mediapipe",
        face_region_provider: Any | None = None,
        mask_params: dict[str, Any] | None = None,
        hand_params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.severity_bands = dict(severity_bands or DEFAULT_SEVERITY_BANDS)
        self.region_weights = dict(region_weights or DEFAULT_REGION_WEIGHTS)
        self.occluder_types = tuple(occluder_types or DEFAULT_OCCLUDER_TYPES)
        if not self.occluder_types:
            raise ValueError("occluder_types must be non-empty")
        for t in self.occluder_types:
            if t not in _ALL_OCCLUDER_TYPES:
                raise ValueError(f"Unknown occluder type {t!r}")
        # Landmark-driven realistic samplers are built only when requested. When
        # the config uses only realistic occluders we do not compose multiple
        # primitives (stacking two masks/hands is not meaningful) — one per view.
        self._realistic_samplers: dict[str, Any] = {}
        if "realistic_mask" in self.occluder_types:
            self._realistic_samplers["realistic_mask"] = build_realistic_mask_sampler(
                **(mask_params or {})
            )
        if "realistic_hand" in self.occluder_types:
            self._realistic_samplers["realistic_hand"] = build_realistic_hand_sampler(
                **(hand_params or {})
            )
        self._single_occluder = set(self.occluder_types) <= set(_REALISTIC_OCCLUDER_TYPES)
        for level, band in self.severity_bands.items():
            lo, hi = float(band[0]), float(band[1])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"Invalid severity band for {level!r}: must satisfy 0 <= min < max <= 1"
                )
        if max_attempts <= 0:
            raise ValueError(f"max_attempts must be > 0, got {max_attempts}")
        self.max_attempts = int(max_attempts)
        if region_provider != "mediapipe":
            raise ValueError(
                f"region_provider={region_provider!r} is not supported. "
                "Runtime synthetic occlusion is MediaPipe-only; there is no "
                "aligned_geometry fallback."
            )
        self.region_provider = region_provider
        self.face_region_provider = face_region_provider or MediaPipeFaceRegionProvider()
        self._default_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _region_result(self, image: Image.Image) -> FaceRegionResult:
        try:
            if hasattr(self.face_region_provider, "extract"):
                result = self.face_region_provider.extract(image)
            else:
                result = self.face_region_provider(image)
        except Exception as exc:
            return _invalid_region_result(
                "unknown_error",
                metadata={"error": str(exc), "provider": self.region_provider},
            )
        if not isinstance(result, FaceRegionResult):
            return _invalid_region_result(
                "unknown_error",
                metadata={
                    "error": "face region provider did not return FaceRegionResult",
                    "provider": self.region_provider,
                },
            )
        return result

    # ------------------------------------------------------------------
    def _sample_view(
        self,
        image: Image.Image,
        region_masks: dict[str, np.ndarray],
        landmarks: np.ndarray | None,
        level: str,
        rng: np.random.Generator,
        forced_type: str | None = None,
    ) -> tuple[SyntheticOcclusionView | None, int]:
        """Acceptance-rejection sampling for one severity level.

        Strong bands are hard to hit with a single geometric occluder under the
        default region weights (a fully face-covering occluder only reaches
        ~0.47), so we compose 2–3 primitives for high bands. Mild bands use a
        single primitive to stay visually subtle. Realistic occluders (e.g.
        masks) are always single — stacking two masks is not meaningful.

        ``forced_type`` pins the occluder type for every component, so a whole
        ``clean < mild < strong`` triple uses one type (e.g. all masks or all
        hands) — see :meth:`generate_pair`.
        """
        lo, hi = self.severity_bands[level]
        if hi <= 0.20:
            scale_low, scale_high, n_low, n_high = 0.45, 0.95, 1, 1
        elif lo >= 0.30:
            scale_low, scale_high, n_low, n_high = 1.20, 2.20, 2, 4
        else:
            scale_low, scale_high, n_low, n_high = 0.80, 1.60, 1, 2

        h, w = image.height, image.width
        ctx = OccluderContext(landmarks=landmarks, region_masks=region_masks, image_size=(w, h))
        for attempt in range(1, self.max_attempts + 1):
            n_components = 1 if self._single_occluder else int(rng.integers(n_low, n_high + 1))
            cur_img = image
            combined_mask = np.zeros((h, w), dtype=bool)
            primitives: list[dict[str, Any]] = []
            for _ in range(n_components):
                occluder_type = (
                    forced_type
                    or self.occluder_types[int(rng.integers(0, len(self.occluder_types)))]
                )
                scale = float(rng.uniform(scale_low, scale_high))
                if occluder_type in self._realistic_samplers:
                    realistic = self._realistic_samplers[occluder_type]
                    cur_img, occluder_mask, info = realistic(cur_img, rng, scale, ctx)
                else:
                    sampler = _OCCLUDER_DISPATCH[occluder_type]
                    cur_img, occluder_mask, info = sampler(cur_img, rng, scale)
                combined_mask = combined_mask | occluder_mask
                primitives.append({"type": occluder_type, "scale": scale, **info})
            severity = compute_severity(combined_mask, region_masks, self.region_weights)
            if lo <= severity <= hi:
                overlap_metrics = compute_occluder_overlap_metrics(combined_mask, region_masks)
                overlap_metrics["weighted_severity"] = float(severity)
                background_overlap = overlap_metrics["background_overlap_ratio"]
                important_overlap = overlap_metrics["important_region_overlap"]
                meta = {
                    "occluder_type": primitives[0]["type"]
                    if len(primitives) == 1
                    else "+".join(p["type"] for p in primitives),
                    "num_attempts": attempt,
                    "n_components": n_components,
                    "primitives": primitives,
                    "level": level,
                    **overlap_metrics,
                    "mostly_background_occlusion": bool(
                        background_overlap > BACKGROUND_OVERLAP_WARNING_THRESHOLD
                    ),
                    "low_important_region_overlap": bool(
                        important_overlap < IMPORTANT_REGION_OVERLAP_WARNING_THRESHOLD
                    ),
                    "high_attempt_count": bool(
                        attempt >= HIGH_ATTEMPT_WARNING_FRACTION * self.max_attempts
                    ),
                }
                return (
                    SyntheticOcclusionView(
                        image=cur_img,
                        occluder_mask=combined_mask,
                        severity=float(severity),
                        level=level,
                        metadata=meta,
                    ),
                    attempt,
                )
        # Failure path: return ``None`` so callers can decide whether to
        # surface it (audit) or silently fall back (training).
        return None, self.max_attempts

    # ------------------------------------------------------------------
    def generate_pair(
        self,
        image: Image.Image,
        rng: np.random.Generator | None = None,
    ) -> SyntheticOcclusionPair:
        """Generate one (mild, strong) pair satisfying ``mild < strong``."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image instance")
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        if w != h:
            raise ValueError(
                f"SyntheticOcclusionGenerator expects an aligned square crop; got {w}x{h}"
            )
        rng = rng if rng is not None else self._default_rng
        region_result = self._region_result(image)
        base_metadata = {
            "image_size": (w, h),
            "region_provider": self.region_provider,
            "mediapipe_valid": bool(region_result.valid),
            "failure_reason": region_result.failure_reason,
            "region_metadata": dict(region_result.metadata),
        }
        if not region_result.valid:
            return SyntheticOcclusionPair(
                original=image,
                mild=None,
                strong=None,
                region_masks=region_result.masks,
                valid=False,
                metadata={
                    **base_metadata,
                    "mild_attempts": 0,
                    "strong_attempts": 0,
                    "mild_severity": float("nan"),
                    "strong_severity": float("nan"),
                    "mild_occluder_type": None,
                    "strong_occluder_type": None,
                    "ordering_ok": False,
                },
            )
        region_masks = region_result.masks

        # Strong first: easier to fit mild under a known strong upper bound if
        # the bands overlap (they don't by default, but stay defensive).
        landmarks = region_result.landmarks
        # Fix the occluder type per anchor so a triple is single-type (all masks
        # or all hands). Across anchors both types appear at both severity levels,
        # which decorrelates occluder type from occlusion level for ranking.
        forced_type = None
        if self._single_occluder and len(self.occluder_types) > 1:
            forced_type = self.occluder_types[int(rng.integers(0, len(self.occluder_types)))]
        strong, strong_attempts = self._sample_view(
            image, region_masks, landmarks, "strong", rng, forced_type=forced_type
        )
        mild, mild_attempts = self._sample_view(
            image, region_masks, landmarks, "mild", rng, forced_type=forced_type
        )
        ordering_ok = mild is not None and strong is not None and mild.severity < strong.severity
        valid = mild is not None and strong is not None and ordering_ok
        failure_reason = None
        if not valid:
            failure_reason = (
                "ordering_failed"
                if mild is not None and strong is not None
                else "generation_failed"
            )

        metadata = {
            **base_metadata,
            "failure_reason": failure_reason,
            "mild_attempts": mild_attempts,
            "strong_attempts": strong_attempts,
            "mild_severity": mild.severity if mild is not None else float("nan"),
            "strong_severity": strong.severity if strong is not None else float("nan"),
            "mild_occluder_type": mild.metadata.get("occluder_type") if mild else None,
            "strong_occluder_type": strong.metadata.get("occluder_type") if strong else None,
            "ordering_ok": ordering_ok,
        }
        return SyntheticOcclusionPair(
            original=image,
            mild=mild,
            strong=strong,
            region_masks=region_masks,
            valid=valid,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def build_generator_from_config(cfg) -> SyntheticOcclusionGenerator | None:
    """Build a :class:`SyntheticOcclusionGenerator` from a config block.

    Returns ``None`` when ``synthetic_occlusion.enabled`` is False or missing,
    so callers can branch with a single ``if generator is None`` check.
    """
    so = cfg.get("synthetic_occlusion", {}) if hasattr(cfg, "get") else {}
    if not so or not so.get("enabled", False):
        return None
    sev = so.get("severity", {}) or {}
    bands: dict[str, tuple[float, float]] = {}
    for level in ("mild", "strong"):
        band = sev.get(level)
        if band is None:
            bands[level] = DEFAULT_SEVERITY_BANDS[level]
        else:
            bands[level] = (float(band["min"]), float(band["max"]))
    occluder_types = list(so.get("occluder_types") or DEFAULT_OCCLUDER_TYPES)

    # Realistic occluder params (optional). compositing.* overrides CompositingConfig.
    mask_params = None
    if "realistic_mask" in occluder_types:
        mask_params = _realistic_params(
            so.get("mask", {}), names_key="templates", arg="template_names"
        )
    hand_params = None
    if "realistic_hand" in occluder_types:
        hand_params = _realistic_params(so.get("hand", {}), names_key="hands", arg="hand_names")

    return SyntheticOcclusionGenerator(
        severity_bands=bands,
        region_weights={
            **DEFAULT_REGION_WEIGHTS,
            **{k: float(v) for k, v in (so.get("region_weights") or {}).items()},
        },
        occluder_types=occluder_types,
        max_attempts=int(so.get("max_attempts", 50)),
        region_provider=str(so.get("region_provider", "mediapipe")),
        mask_params=mask_params,
        hand_params=hand_params,
        seed=int(so.get("seed", 42)),
    )


def _realistic_params(block, *, names_key: str, arg: str) -> dict[str, Any] | None:
    """Parse a realistic-occluder config block (``mask:`` / ``hand:``) into kwargs."""
    block = block or {}
    params: dict[str, Any] = {}
    if block.get(names_key):
        params[arg] = tuple(block[names_key])
    comp_overrides = {k: v for k, v in (block.get("compositing", {}) or {}).items()}
    if comp_overrides:
        params["compositing"] = CompositingConfig(**comp_overrides)
    if block.get("asset_dir"):
        params["asset_dir"] = str(block["asset_dir"])
    return params or None


__all__ = [
    "BACKGROUND_OVERLAP_WARNING_THRESHOLD",
    "DEFAULT_OCCLUDER_TYPES",
    "DEFAULT_REGION_WEIGHTS",
    "DEFAULT_SEVERITY_BANDS",
    "FaceRegionResult",
    "HIGH_ATTEMPT_WARNING_FRACTION",
    "IMPORTANT_REGION_OVERLAP_WARNING_THRESHOLD",
    "MediaPipeFaceRegionProvider",
    "OVERLAP_FLAG_KEYS",
    "OVERLAP_METRIC_KEYS",
    "REQUIRED_REGION_MASKS",
    "SyntheticOcclusionGenerator",
    "SyntheticOcclusionPair",
    "SyntheticOcclusionView",
    "build_aligned_face_region_masks",
    "build_generator_from_config",
    "compute_occluder_overlap_metrics",
    "compute_severity",
]
