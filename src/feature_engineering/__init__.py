"""Multimodal feature engineering and fusion module."""

from .fusion import (
    fuse_features,
    get_all_feature_names,
    build_multimodal_dataset,
    prepare_single_sample_features,
)

__all__ = [
    "fuse_features",
    "get_all_feature_names",
    "build_multimodal_dataset",
    "prepare_single_sample_features",
]
