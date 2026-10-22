"""Unit tests for multimodal feature fusion."""

import numpy as np
import pytest
from src.feature_engineering.fusion import (
    fuse_features,
    get_all_feature_names,
)
from src.image_processing.feature_extractor import get_feature_names as get_img_names
from src.sensor_simulation.simulator import get_sensor_feature_names


def test_fuse_features():
    img_names = get_img_names()
    sensor_names = get_sensor_feature_names()

    dummy_img = {k: 1.0 for k in img_names}
    dummy_sensor = {k: 2.0 for k in sensor_names}

    fused = fuse_features(dummy_img, dummy_sensor)
    all_names = get_all_feature_names()

    assert len(fused) == len(all_names)
    assert len(fused) == len(img_names) + len(sensor_names)
    assert isinstance(fused, np.ndarray)
