"""AI model architectures, multimodal fusion classifier, and evaluation metrics."""

try:
    from .cnn_model import build_cell_cnn, train_cnn_model, load_image_model
except (ImportError, Exception):
    # Fallback when TensorFlow C++ DLL is restricted by Windows Application Control policies
    build_cell_cnn = None
    train_cnn_model = None
    load_image_model = None

from .multimodal_model import (
    MultimodalClassifier,
    train_multimodal_model,
    load_multimodal_model,
    compute_research_risk_score,
)
from .evaluator import evaluate_classifier, plot_confusion_matrix_and_roc

__all__ = [
    "build_cell_cnn",
    "train_cnn_model",
    "load_image_model",
    "MultimodalClassifier",
    "train_multimodal_model",
    "load_multimodal_model",
    "compute_research_risk_score",
    "evaluate_classifier",
    "plot_confusion_matrix_and_roc",
]
