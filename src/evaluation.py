"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Evaluation Engine for Competency Assessment Models

Computes and visualizes:
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- Confusion Matrix per model
- Classification Report
- Comparison charts between candidate models
- Saves metrics summary to JSON and generates evaluation figures
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

COMPETENCY_CLASSES = ["Beginner", "Developing", "Proficient", "Advanced"]


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Computes rigorous evaluation metrics for a multi-class model.
    """
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=COMPETENCY_CLASSES).tolist()
    report = classification_report(y_test, y_pred, labels=COMPETENCY_CLASSES, output_dict=True, zero_division=0)

    results = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "class_labels": COMPETENCY_CLASSES,
        "classification_report": report
    }
    return results, y_pred


def plot_confusion_matrix(cm, classes, title, output_path):
    """
    Plots a sleek confusion matrix and saves to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_array = np.array(cm)

    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Competency Level",
        xlabel="Predicted Competency Level"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate numbers
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j, i, format(cm_array[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")


def plot_model_comparison(comparison_dict, output_path):
    """
    Generates a comparative bar chart for Random Forest vs Logistic Regression.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"]

    models = list(comparison_dict.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#2ca02c"]

    for idx, model_name in enumerate(models):
        scores = [comparison_dict[model_name][m] for m in metrics]
        offset = (idx - 0.5) * width + width / 2
        rects = ax.bar(x + offset, scores, width, label=model_name, color=colors[idx % len(colors)], alpha=0.85)
        # Value tags
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("Competency Assessment Model Performance Comparison\n(SIH 2026 - Dream Weavers AD14)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Model comparison chart saved to {output_path}")
