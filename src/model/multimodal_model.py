"""Multimodal classification model and Research Prototype Risk Score computation."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

import config


def compute_research_risk_score(
    p_abnormal: float,
    image_features: Dict[str, float],
    sensor_features: Dict[str, float]
) -> float:
    """Compute continuous Research Prototype Risk Score (0.0 to 1.0).

    Combines:
    1. Multimodal model abnormal probability (weight: 0.65)
    2. Electrochemical sensor anomaly score (weight: 0.20)
       - Based on charge transfer resistance (Rct) suppression and peak Faradaic current increase
    3. Morphological atypia score (weight: 0.15)
       - Based on loss of circularity and elevated structural texture variance

    Returns:
        Risk score rounded to 2 decimal places in [0.00, 1.00].
    """
    # 1. Base model probability
    w_model = 0.65
    score_model = float(np.clip(p_abnormal, 0.0, 1.0))

    # 2. Electrochemical sensor anomaly
    # Rct: 1500 Ohms is healthy, 600 Ohms is abnormal
    rct = sensor_features.get("charge_transfer_resistance", 1500.0)
    rct_score = float(np.clip((1500.0 - rct) / (1500.0 - 600.0), 0.0, 1.0))

    # Peak current: 12 uA is healthy, 25 uA is abnormal
    ipa = sensor_features.get("peak_anodic_current", 12.0)
    ipa_score = float(np.clip((ipa - 12.0) / (25.0 - 12.0), 0.0, 1.0))

    sensor_score = 0.5 * rct_score + 0.5 * ipa_score
    w_sensor = 0.20

    # 3. Morphological atypia
    # Healthy single cells have high circularity (~0.85-0.95); parasitized/abnormal have lower circularity
    circ = image_features.get("circularity", 0.90)
    circ_score = float(np.clip((0.90 - circ) / (0.90 - 0.55), 0.0, 1.0))

    # Abnormal cells exhibit higher internal texture variance due to parasites/granules
    tex = image_features.get("texture_variance", 20.0)
    tex_score = float(np.clip((tex - 20.0) / (80.0 - 20.0), 0.0, 1.0))

    morpho_score = 0.5 * circ_score + 0.5 * tex_score
    w_morpho = 0.15

    total_risk = (w_model * score_model) + (w_sensor * sensor_score) + (w_morpho * morpho_score)
    return round(float(np.clip(total_risk, 0.0, 1.0)), 2)


class MultimodalClassifier:
    """Ensemble Random Forest classifier trained on fused morphological and electrochemical features."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 8, random_state: int = config.RANDOM_SEED):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state
        )
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "MultimodalClassifier":
        if feature_names is not None:
            self.feature_names = list(feature_names)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if not hasattr(self.model, "feature_importances_") or not self.feature_names:
            return {}
        importances = self.model.feature_importances_
        return {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}


def train_multimodal_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> MultimodalClassifier:
    """Train and persist the multimodal fusion Random Forest model."""
    if save_path is None:
        save_path = config.FUSION_MODEL_PATH

    save_path.parent.mkdir(parents=True, exist_ok=True)
    clf = MultimodalClassifier()
    clf.fit(X_train, y_train, feature_names=feature_names)

    joblib.dump(clf, save_path)
    print(f"Multimodal model successfully saved to {save_path}")
    return clf


def load_multimodal_model(model_path: Optional[Path] = None) -> MultimodalClassifier:
    """Load persisted multimodal classifier artifact."""
    if model_path is None:
        model_path = config.FUSION_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Multimodal model artifact not found at {model_path}. Train the model first.")
    return joblib.load(model_path)
