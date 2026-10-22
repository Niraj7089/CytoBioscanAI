"""Unit and integration tests for the image processing module."""

import numpy as np
import pytest
from src.image_processing.preprocessor import (
    resize_image,
    denoise_image,
    enhance_contrast,
    normalize_image,
    preprocess_image,
)
from src.image_processing.segmentation import compute_cell_mask, segment_cell
from src.image_processing.feature_extractor import extract_image_features, get_feature_names


@pytest.fixture
def synthetic_cell_image():
    """Create a synthetic 128x128 RGB image with a circular cell."""
    img = np.full((128, 128, 3), 220, dtype=np.uint8)  # Light background
    y, x = np.ogrid[:128, :128]
    mask = (x - 64) ** 2 + (y - 64) ** 2 <= 30 ** 2
    img[mask] = [120, 50, 60]  # Darker cell body
    return img


def test_resize_and_preprocess(synthetic_cell_image):
    norm, enhanced = preprocess_image(synthetic_cell_image, target_size=(128, 128))
    assert norm.shape == (128, 128, 3)
    assert enhanced.shape == (128, 128, 3)
    assert 0.0 <= norm.min() and norm.max() <= 1.0
    assert enhanced.dtype == np.uint8


def test_segmentation(synthetic_cell_image):
    mask, contour, overlay = segment_cell(synthetic_cell_image)
    assert mask.shape == (128, 128)
    assert contour is not None
    assert overlay.shape == (128, 128, 3)
    assert np.sum(mask > 0) > 0


def test_feature_extraction(synthetic_cell_image):
    features = extract_image_features(synthetic_cell_image)
    required_names = get_feature_names()
    for name in required_names:
        assert name in features
        assert isinstance(features[name], (int, float))
    assert features["cell_area"] > 0
    assert 0.0 <= features["circularity"] <= 1.0
