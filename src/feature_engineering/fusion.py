"""Multimodal feature fusion: combining morphological image descriptors with electrochemical sensor parameters."""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import config
from src.image_processing.feature_extractor import (
    extract_image_features,
    get_feature_names as get_image_feature_names,
)
from src.sensor_simulation.simulator import (
    get_sensor_feature_names,
)


def get_all_feature_names() -> List[str]:
    """Return concatenated list of all multimodal feature names in deterministic order."""
    return get_image_feature_names() + get_sensor_feature_names()


def fuse_features(
    image_features: Dict[str, float],
    sensor_features: Dict[str, float]
) -> np.ndarray:
    """Concatenate extracted image features and sensor features into a unified 1D vector.

    Args:
        image_features: Dictionary of image morphological/intensity features.
        sensor_features: Dictionary of electrochemical features.

    Returns:
        1D float64 numpy array of shape (N_features,).
    """
    img_keys = get_image_feature_names()
    sensor_keys = get_sensor_feature_names()

    vec = []
    for k in img_keys:
        vec.append(float(image_features.get(k, 0.0)))
    for k in sensor_keys:
        vec.append(float(sensor_features.get(k, 0.0)))

    return np.array(vec, dtype=np.float64)


def build_multimodal_dataset(
    sensor_df: Optional[pd.DataFrame] = None
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    List[str],
    StandardScaler
]:
    """Extract and fuse multimodal features for all train/valid/test samples.

    Args:
        sensor_df: Optional preloaded sensor dataframe. If None, loaded from disk.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_names, scaler)
    """
    if sensor_df is None:
        sensor_df = pd.read_csv(config.SENSOR_CSV_PATH)

    feature_names = get_all_feature_names()
    data_by_split = {"train": ([], []), "valid": ([], []), "test": ([], [])}

    # Pre-extract or process image features
    import cv2

    for _, row in sensor_df.iterrows():
        split = row["split"]
        label_str = row["label"]
        label_int = config.LABEL_MAP[label_str]
        img_rel_path = row["image_rel_path"]
        img_full_path = config.BASE_DIR / img_rel_path

        # Load image and extract features
        if img_full_path.exists():
            img_bgr = cv2.imread(str(img_full_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_feat = extract_image_features(img_rgb)
        else:
            # Fallback zeros
            img_feat = {k: 0.0 for k in get_image_feature_names()}

        sensor_feat = {k: float(row[k]) for k in get_sensor_feature_names()}
        fused_vec = fuse_features(img_feat, sensor_feat)

        data_by_split[split][0].append(fused_vec)
        data_by_split[split][1].append(label_int)

    X_train = np.array(data_by_split["train"][0], dtype=np.float64)
    y_train = np.array(data_by_split["train"][1], dtype=np.int64)

    X_val = np.array(data_by_split["valid"][0], dtype=np.float64)
    y_val = np.array(data_by_split["valid"][1], dtype=np.int64)

    X_test = np.array(data_by_split["test"][0], dtype=np.float64)
    y_test = np.array(data_by_split["test"][1], dtype=np.int64)

    # Fit scaler on train only to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler artifact
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)

    return (
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        feature_names,
        scaler,
    )


def prepare_single_sample_features(
    image_features: Dict[str, float],
    sensor_features: Dict[str, float],
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a single sample's fused vector and scaled vector for model inference.

    Returns:
        Tuple of (raw_vector_1d, scaled_vector_2d_for_prediction)
    """
    raw_vec = fuse_features(image_features, sensor_features)
    if scaler is None:
        if config.SCALER_PATH.exists():
            scaler = joblib.load(config.SCALER_PATH)
        else:
            raise FileNotFoundError("Scaler artifact not found. Please train model first.")

    scaled_2d = scaler.transform(raw_vec.reshape(1, -1))
    return raw_vec, scaled_2d
