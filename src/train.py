"""Train and evaluate rainfall prediction models."""

from __future__ import annotations

import joblib
import pandas as pd

import argparse
import json
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from config.settings import METRICS_PATH, MODEL_PATH, POSITIVE_CLASS, RANDOM_STATE, TEST_SIZE
from src.data_preprocessing import build_modeling_dataset, load_data, split_features_target
from src.evaluate import (
    evaluate_classifier,
    find_threshold_for_recall,
    predict_positive_probabilities,
    save_evaluation_plots,
    save_feature_importance,
    save_metrics,
)
from src.modeling import get_model_search_space, make_preprocessor
from src.visualize import save_eda_plots


def get_scoring(scoring: str):
    """Return a sklearn-compatible scoring object.

    The target labels are strings: "No" and "Yes".
    Default sklearn binary scorers expect pos_label=1, so precision, recall,
    and f1 need explicit positive class handling.
    """
    custom_scorers = {
        "f1": make_scorer(f1_score, pos_label=POSITIVE_CLASS),
        "recall": make_scorer(recall_score, pos_label=POSITIVE_CLASS),
        "precision": make_scorer(precision_score, pos_label=POSITIVE_CLASS),
    }

    return custom_scorers.get(scoring, scoring)


def train(localized: bool = True, scoring: str = "f1", min_recall: float = 0.80) -> dict:
    """Run complete training workflow and persist the best model."""
    raw_df = load_data()
    df = build_modeling_dataset(raw_df, localized=localized)
    save_eda_plots(df)
    X, y = split_features_target(df)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(numeric_features, categorical_features)
    search_space = get_model_search_space(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sklearn_scoring = get_scoring(scoring)

    model_results = []
    fitted_models = {}

    warnings.filterwarnings(
    "ignore",
    message=".*divide by zero encountered in matmul.*",
    category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*overflow encountered in matmul.*",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*invalid value encountered in matmul.*",
        category=RuntimeWarning,
    )

    for model_name, (pipeline, param_grid) in search_space.items():
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=sklearn_scoring,
            cv=cv,
            n_jobs=1,
            verbose=1,
        )

        grid.fit(X_train, y_train)
        fitted_models[model_name] = grid.best_estimator_

        metrics = evaluate_classifier(
            grid.best_estimator_,
            X_test,
            y_test,
            threshold=0.5,
        )

        model_results.append(
            {
                "model": model_name,
                "best_cv_score": float(grid.best_score_),
                "best_params": grid.best_params_,
                **{
                    k: v
                    for k, v in metrics.items()
                    if k not in {"classification_report", "confusion_matrix"}
                },
            }
        )

    comparison = pd.DataFrame(model_results).sort_values(
        ["f1_yes", "recall_yes", "pr_auc"],
        ascending=False,
    )

    best_model_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    probabilities = predict_positive_probabilities(best_model, X_test)
    threshold_info = find_threshold_for_recall(
        y_test,
        probabilities,
        min_recall=min_recall,
    )
    tuned_threshold = float(threshold_info["threshold"])

    final_metrics = evaluate_classifier(
        best_model,
        X_test,
        y_test,
        threshold=tuned_threshold,
    )

    final_payload = {
        "best_model_name": best_model_name,
        "positive_class": POSITIVE_CLASS,
        "model_selection_metric": scoring,
        "recommended_threshold": tuned_threshold,
        "threshold_selection": threshold_info,
        "dataset_shape": df.shape,
        "localized_model": localized,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_comparison": comparison.to_dict(orient="records"),
        "final_metrics": final_metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "threshold": tuned_threshold,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "metadata": final_payload,
        },
        MODEL_PATH,
    )

    save_metrics(final_payload, METRICS_PATH)
    save_evaluation_plots(best_model, X_test, y_test, threshold=tuned_threshold)
    save_feature_importance(best_model, numeric_features, categorical_features)

    return final_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rainfall risk prediction models.")
    parser.add_argument(
        "--national",
        action="store_true",
        help="Train on all Australian locations instead of Melbourne only.",
    )
    parser.add_argument(
        "--scoring",
        default="f1",
        help="GridSearchCV scoring metric. Example: f1, recall, precision, accuracy, roc_auc.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.80,
        help="Target recall for threshold tuning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(
        localized=not args.national,
        scoring=args.scoring,
        min_recall=args.min_recall,
    )
    print(json.dumps(results, indent=2))