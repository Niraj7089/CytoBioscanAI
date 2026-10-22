"""Single-cell morphological segmentation and boundary detection."""

from typing import Tuple, Optional
import cv2
import numpy as np


def compute_cell_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Compute binary mask isolating the single cell from background using Otsu thresholding.

    Args:
        image_rgb: uint8 RGB image of shape (H, W, 3).

    Returns:
        Binary mask (uint8) of shape (H, W) where cell pixels are 255 and background is 0.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # In single-cell Giemsa-stained microscopy, cells are typically darker than background.
    # We apply Otsu's thresholding with inverted binary.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to close internal holes and remove spurious background specs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the primary single cell)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if num_labels <= 1:
        return cleaned

    # Label 0 is background; find component with max area >= 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + int(np.argmax(areas))
    cell_mask = np.where(labels == max_label, 255, 0).astype(np.uint8)

    # Fill internal holes of the primary cell
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(cell_mask, contours, -1, 255, thickness=cv2.FILLED)

    return cell_mask


def segment_cell(image_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Perform cell segmentation and generate visual boundary overlay.

    Args:
        image_rgb: uint8 RGB image of shape (H, W, 3).

    Returns:
        Tuple of:
            - binary_mask: uint8 (H, W)
            - primary_contour: (N, 1, 2) contour coordinates or None
            - overlay_image: RGB image with cell contour and centroid annotated
    """
    mask = compute_cell_mask(image_rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = image_rgb.copy()
    primary_contour = None

    if contours:
        # Select largest contour by area
        primary_contour = max(contours, key=cv2.contourArea)

        # Draw smooth neon cyan contour outline
        cv2.drawContours(overlay, [primary_contour], -1, (0, 220, 255), 2)

        # Calculate and draw centroid
        moments = cv2.moments(primary_contour)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(overlay, (cx, cy), 3, (255, 60, 60), -1)

    return mask, primary_contour, overlay
