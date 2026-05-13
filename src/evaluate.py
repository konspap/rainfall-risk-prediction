"""Evaluation helpers for rainfall classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.settings import FIGURES_DIR, POSITIVE_CLASS


def predict_positive_probabilities(model, X: pd.DataFrame) -> list[float]:
    """Return probability estimates for the positive class."""
    probabilities = model.predict_proba(X)
    class_index = list(model.classes_).index(POSITIVE_CLASS)
    return probabilities[:, class_index]


def apply_threshold(probabilities, threshold: float = 0.5) -> list[str]:
    """Convert positive-class probabilities into Yes/No labels."""
    return [POSITIVE_CLASS if proba >= threshold else "No" for proba in probabilities]


def find_threshold_for_recall(y_true, probabilities, min_recall: float = 0.80) -> dict[str, float]:
    """Find a threshold that reaches a target recall while maximizing F1.

    If no threshold reaches the requested recall, the best-F1 threshold is returned.
    """
    rows = []
    for threshold in [i / 100 for i in range(5, 96)]:
        preds = apply_threshold(probabilities, threshold)
        rows.append(
            {
                "threshold": threshold,
                "precision_yes": precision_score(y_true, preds, pos_label=POSITIVE_CLASS, zero_division=0),
                "recall_yes": recall_score(y_true, preds, pos_label=POSITIVE_CLASS, zero_division=0),
                "f1_yes": f1_score(y_true, preds, pos_label=POSITIVE_CLASS, zero_division=0),
            }
        )
    scores = pd.DataFrame(rows)
    eligible = scores[scores["recall_yes"] >= min_recall]
    if eligible.empty:
        best = scores.sort_values("f1_yes", ascending=False).iloc[0]
    else:
        best = eligible.sort_values("f1_yes", ascending=False).iloc[0]
    return best.to_dict()


def evaluate_classifier(model, X_test: pd.DataFrame, y_test, threshold: float = 0.5) -> dict[str, Any]:
    """Calculate core classification metrics."""
    probabilities = predict_positive_probabilities(model, X_test)
    predictions = apply_threshold(probabilities, threshold)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_yes": precision_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "recall_yes": recall_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "f1_yes": f1_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0),
        "roc_auc": roc_auc_score((y_test == POSITIVE_CLASS).astype(int), probabilities),
        "pr_auc": average_precision_score((y_test == POSITIVE_CLASS).astype(int), probabilities),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["No", POSITIVE_CLASS]).tolist(),
    }


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_evaluation_plots(model, X_test: pd.DataFrame, y_test, threshold: float = 0.5) -> None:
    """Save confusion matrix, ROC curve, and precision-recall curve."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    probabilities = predict_positive_probabilities(model, X_test)
    predictions = apply_threshold(probabilities, threshold)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=["No", POSITIVE_CLASS])
    plt.title(f"Confusion Matrix at Threshold {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions((y_test == POSITIVE_CLASS).astype(int), probabilities)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions((y_test == POSITIVE_CLASS).astype(int), probabilities)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()


def get_feature_names(model, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    """Get transformed feature names after preprocessing."""
    preprocessor = model.named_steps["preprocessor"]
    cat_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    return numeric_features + list(cat_names)


def save_feature_importance(model, numeric_features: list[str], categorical_features: list[str]) -> None:
    """Save feature importance or coefficient chart when available."""
    classifier = model.named_steps["classifier"]
    feature_names = get_feature_names(model, numeric_features, categorical_features)

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        column = "importance"
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
        column = "absolute_coefficient"
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, column: values})
        .sort_values(column, ascending=False)
        .head(20)
        .sort_values(column, ascending=True)
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    plt.barh(importance_df["feature"], importance_df[column])
    plt.title("Top 20 Model Drivers")
    plt.xlabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close()
