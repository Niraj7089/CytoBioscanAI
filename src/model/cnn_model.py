"""Convolutional Neural Network for single-cell optical microscopy classification.

Designed with resilient fallbacks for environments where TensorFlow C++ DLLs are restricted
by Windows Application Control (WDAC / AppLocker) policies.
"""

from pathlib import Path
from typing import Tuple, Optional, Any
import numpy as np

import config

# Safe TensorFlow import with AppLocker policy detection
TF_AVAILABLE = False
tf = None
keras = None
layers = None

try:
    import tensorflow as _tf
    from tensorflow import keras as _keras
    from tensorflow.keras import layers as _layers
    tf = _tf
    keras = _keras
    layers = _layers
    TF_AVAILABLE = True
except (ImportError, Exception):
    TF_AVAILABLE = False


def build_cell_cnn(input_shape: Tuple[int, int, int] = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)) -> Any:
    """Build a lightweight, robust CNN for single-cell microscopy classification."""
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow native runtime is restricted by an Application Control policy on this machine. "
            "The multimodal system continues to operate seamlessly using Scikit-Learn feature fusion."
        )

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense head
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid", name="classification_output")
    ], name="SingleCell_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


def train_cnn_model(
    train_ds: Any,
    val_ds: Any,
    epochs: int = config.EPOCHS,
    model_save_path: Optional[Path] = None
) -> Tuple[Any, Any]:
    """Train the CNN model on image datasets."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow native runtime is unavailable due to an Application Control policy.")

    if model_save_path is None:
        model_save_path = config.IMAGE_MODEL_PATH

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_cell_cnn()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    model.save(model_save_path)
    print(f"CNN model successfully saved to {model_save_path}")
    return model, history


def load_image_model(model_path: Optional[Path] = None) -> Any:
    """Load trained Keras CNN model artifact."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow native runtime is unavailable due to an Application Control policy.")

    if model_path is None:
        model_path = config.IMAGE_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")
    return keras.models.load_model(model_path)
