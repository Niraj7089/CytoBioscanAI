"""Extraction of morphological, shape, intensity, and texture features from cell images."""

from typing import Dict, Any, List
import cv2
import numpy as np
from .segmentation import compute_cell_mask


def extract_image_features(image_rgb: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """Extract quantitative morphological, shape, intensity, and texture features from a cell image.

    Args:
        image_rgb: uint8 RGB image of shape (H, W, 3).
        mask: Optional precomputed binary cell mask. If None, it will be computed.

    Returns:
        Dictionary of extracted biophysical image features.
    """
    if mask is None:
        mask = compute_cell_mask(image_rgb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Defaults in case of empty mask
    area = 0.0
    perimeter = 0.0
    circularity = 0.0
    aspect_ratio = 1.0
    extent = 0.0
    solidity = 0.0
    equivalent_diameter = 0.0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))

        if perimeter > 0:
            circularity = float((4.0 * np.pi * area) / (perimeter ** 2))
            circularity = min(1.0, max(0.0, circularity))

        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0:
            aspect_ratio = float(w) / float(h)

        rect_area = float(w * h)
        if rect_area > 0:
            extent = area / rect_area

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = area / hull_area

        equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))

    # Masked intensity statistics
    cell_pixels = image_rgb[mask > 0]
    if len(cell_pixels) == 0:
        cell_pixels = image_rgb.reshape(-1, 3)

    mean_r = float(np.mean(cell_pixels[:, 0]))
    mean_g = float(np.mean(cell_pixels[:, 1]))
    mean_b = float(np.mean(cell_pixels[:, 2]))
    std_r = float(np.std(cell_pixels[:, 0]))
    std_g = float(np.std(cell_pixels[:, 1]))
    std_b = float(np.std(cell_pixels[:, 2]))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_cell_pixels = gray[mask > 0]
    if len(gray_cell_pixels) == 0:
        gray_cell_pixels = gray.flatten()

    mean_intensity = float(np.mean(gray_cell_pixels))
    std_intensity = float(np.std(gray_cell_pixels))

    # Texture approximation using Laplacian variance (sharpness/structural texture)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = float(laplacian.var())

    # Gradient magnitude (local edge contrast)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    mean_gradient = float(np.mean(grad_mag))

    return {
        "cell_area": round(area, 2),
        "cell_perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "extent": round(extent, 4),
        "solidity": round(solidity, 4),
        "equivalent_diameter": round(equivalent_diameter, 2),
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "std_r": round(std_r, 2),
        "std_g": round(std_g, 2),
        "std_b": round(std_b, 2),
        "texture_variance": round(texture_variance, 2),
        "mean_gradient": round(mean_gradient, 2),
    }


def get_feature_names() -> List[str]:
    """Return ordered list of morphological/image feature keys."""
    return [
        "cell_area",
        "cell_perimeter",
        "circularity",
        "aspect_ratio",
        "extent",
        "solidity",
        "equivalent_diameter",
        "mean_intensity",
        "std_intensity",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "texture_variance",
        "mean_gradient",
    ]
