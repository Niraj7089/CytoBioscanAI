"""Evaluation metrics and publication-ready diagnostic visualizations."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import config


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_confusion_matrix_and_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_importances: Optional[Dict[str, float]] = None,
    save_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Generate and save publication-quality confusion matrix, ROC curve, and feature importances."""
    if save_dir is None:
        save_dir = config.REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = save_dir / "confusion_matrix.png"
    roc_path = save_dir / "roc_curve.png"
    feat_path = save_dir / "feature_importance.png"

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Multimodal Prototype", fontsize=12, pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(config.CLASS_NAMES))
    plt.xticks(tick_marks, config.CLASS_NAMES, fontsize=10)
    plt.yticks(tick_marks, config.CLASS_NAMES, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold"
            )

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, color="#0066cc", lw=2.5, label=f"Multimodal ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # 3. Feature Importances Plot (if provided)
    if feature_importances:
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [x[0].replace("_", " ").title() for x in reversed(sorted_feats)]
        vals = [x[1] for x in reversed(sorted_feats)]

        plt.figure(figsize=(8, 5), dpi=300)
        bars = plt.barh(names, vals, color="#008080", edgecolor="#004d40", alpha=0.85)
        plt.xlabel("Gini Feature Importance", fontsize=11)
        plt.title("Top 10 Fused Multimodal Biomarkers", fontsize=12, pad=12)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(feat_path)
        plt.close()

    return {
        "confusion_matrix": cm_path,
        "roc_curve": roc_path,
        "feature_importance": feat_path,
    }
