"""Public data-loading utilities.

Heavy training objects are imported lazily so lightweight helpers, such as the
synthetic occlusion audit, do not need to import Lightning.
"""

from importlib import import_module
from typing import Any

from .metadata import add_path_metadata
from .synthetic_occlusion import (
    FaceRegionResult,
    MediaPipeFaceRegionProvider,
    SyntheticOcclusionGenerator,
    SyntheticOcclusionPair,
    SyntheticOcclusionView,
    build_generator_from_config,
)

_LAZY_EXPORTS = {
    "FaceOcclusionDataModule": ("face_occlusion.data.datamodule", "FaceOcclusionDataModule"),
    "FaceOcclusionDataset": ("face_occlusion.data.dataset", "FaceOcclusionDataset"),
    "GenderOcclusionBalancedBatchSampler": (
        "face_occlusion.data.samplers",
        "GenderOcclusionBalancedBatchSampler",
    ),
    "build_eval_transform": ("face_occlusion.data.transforms", "build_eval_transform"),
    "build_synthetic_view_transform": (
        "face_occlusion.data.transforms",
        "build_synthetic_view_transform",
    ),
    "build_train_transform": ("face_occlusion.data.transforms", "build_train_transform"),
    "load_split": ("face_occlusion.data.splits", "load_split"),
    "make_stratified_split": ("face_occlusion.data.splits", "make_stratified_split"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "FaceOcclusionDataset",
    "FaceOcclusionDataModule",
    "FaceRegionResult",
    "GenderOcclusionBalancedBatchSampler",
    "MediaPipeFaceRegionProvider",
    "SyntheticOcclusionGenerator",
    "SyntheticOcclusionPair",
    "SyntheticOcclusionView",
    "add_path_metadata",
    "build_eval_transform",
    "build_generator_from_config",
    "build_synthetic_view_transform",
    "build_train_transform",
    "load_split",
    "make_stratified_split",
]
