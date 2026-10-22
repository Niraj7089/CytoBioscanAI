"""Image preprocessing pipeline for microscopy single-cell analysis."""

from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

import config


def load_image(image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Load an image from various input sources into an RGB uint8 numpy array.

    Args:
        image_input: File path, bytes, PIL Image, or numpy array.

    Returns:
        RGB numpy array of shape (H, W, 3) and uint8 type.
    """
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        img_bgr = cv2.imread(path_str)
        if img_bgr is None:
            raise ValueError(f"Failed to load image from path: {path_str}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from bytes.")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        if image_input.ndim == 3:
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input.copy()

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """Resize image to target (width, height)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering to remove high-frequency noise while preserving cell boundaries."""
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance cell contrast using CLAHE on the luminance channel (LAB color space)."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel intensity values from [0, 255] to [0.0, 1.0]."""
    return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)


def preprocess_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = config.IMAGE_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Run full preprocessing pipeline: load -> resize -> denoise -> contrast enhancement -> normalize.

    Returns:
        Tuple of (normalized_float_array [0, 1], enhanced_uint8_array [0, 255])
    """
    raw_rgb = load_image(image_input)
    resized = resize_image(raw_rgb, target_size)
    denoised = denoise_image(resized)
    enhanced = enhance_contrast(denoised)
    normalized = normalize_image(enhanced)
    return normalized, enhanced
