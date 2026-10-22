"""Main training pipeline for the AI-Assisted Single-Cell Diagnosis Research Prototype.

Executes end-to-end:
1. Dataset verification and sensor dataset generation
2. Deep Learning CNN model training on microscopy images
3. Multimodal feature fusion (Image descriptors + Electrochemical sensor features)
4. Random Forest multimodal classifier training
5. Comprehensive evaluation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC)
6. Model artifact persistence to models/ and publication figures to reports/
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

import config
from src.sensor_simulation.simulator import generate_sensor_dataset
from src.feature_engineering.fusion import build_multimodal_dataset
from src.model.cnn_model import train_cnn_model, build_cell_cnn
from src.model.multimodal_model import train_multimodal_model
from src.model.evaluator import evaluate_classifier, plot_confusion_matrix_and_roc


def main():
    print("=" * 70)
    print("AI-Assisted Single-Cell Imaging & Electrochemical Detection System")
    print("Research Prototype Training Pipeline")
    print("=" * 70)

    # 1. Verify Dataset
    train_dir = config.TRAIN_DIR
    test_dir = config.TEST_DIR
    val_dir = config.VALID_DIR

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print("Error: Dataset not found in data/raw/train. Please ensure images are extracted.")
        sys.exit(1)

    print("\n[Step 1/5] Verifying datasets and generating paired electrochemical sensors...")
    if not config.SENSOR_CSV_PATH.exists():
        sensor_df = generate_sensor_dataset()
    else:
        sensor_df = pd.read_csv(config.SENSOR_CSV_PATH)
        print(f"Loaded existing sensor dataset with {len(sensor_df)} samples.")

    # 2. Train CNN on Optical Single-Cell Images
    print("\n[Step 2/5] Training Deep Convolutional Neural Network (CNN) on Optical Images...")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        shuffle=False,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        shuffle=False,
    )

    # Normalization layer
    normalization_layer = keras.layers.Rescaling(1.0 / 255.0)
    train_ds_norm = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds_norm = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds_norm = test_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    cnn_model, history = train_cnn_model(
        train_ds=train_ds_norm,
        val_ds=val_ds_norm,
        epochs=config.EPOCHS,
        model_save_path=config.IMAGE_MODEL_PATH
    )

    cnn_eval = cnn_model.evaluate(test_ds_norm, verbose=0)
    print(f"CNN Optical Test Accuracy: {cnn_eval[1] * 100:.2f}% | Test AUC: {cnn_eval[2]:.4f}")

    # 3. Extract and Fuse Multimodal Features
    print("\n[Step 3/5] Extracting & Fusing Multimodal Features (Image Morphology + Sensors)...")
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names,
        scaler,
    ) = build_multimodal_dataset(sensor_df)

    print(f"Extracted {len(feature_names)} fused biomarkers across {len(X_train)} training and {len(X_test)} test samples.")

    # 4. Train Multimodal Classifier
    print("\n[Step 4/5] Training Multimodal Fusion Classifier (Random Forest Ensemble)...")
    clf = train_multimodal_model(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        save_path=config.FUSION_MODEL_PATH
    )

    # 5. Evaluate on Independent Test Set
    print("\n[Step 5/5] Evaluating Multimodal Classifier on Held-Out Test Set...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_prob)
    importances = clf.get_feature_importances()

    plot_paths = plot_confusion_matrix_and_roc(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_importances=importances,
        save_dir=config.REPORTS_DIR
    )

    print("\n" + "=" * 50)
    print("FINAL MULTIMODAL TEST SET EVALUATION")
    print("=" * 50)
    print(f"  • Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Precision:   {metrics['precision'] * 100:.2f}%")
    print(f"  • Sensitivity: {metrics['recall'] * 100:.2f}% (Recall)")
    print(f"  • Specificity: {metrics['specificity'] * 100:.2f}%")
    print(f"  • F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  • Confusion Matrix: [TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}, TP={metrics['true_positives']}]")
    print("=" * 50)
    print(f"Saved Evaluation Plots:")
    for k, p in plot_paths.items():
        print(f"  - {k}: {p}")
    print("Training pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
