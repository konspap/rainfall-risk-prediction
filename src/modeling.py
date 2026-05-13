"""Model pipelines and hyperparameter grids."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import RANDOM_STATE


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_model_search_space(preprocessor: ColumnTransformer) -> dict[str, tuple[Pipeline, dict]]:
    """Return candidate model pipelines and compact tuning grids."""
    return {
        "dummy_baseline": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            {},
        ),
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "classifier__solver": ["liblinear"],
                "classifier__penalty": ["l1", "l2"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            {
                "classifier__n_estimators": [100, 250],
                "classifier__max_depth": [None, 8, 16],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_iter": [100, 200],
                "classifier__max_leaf_nodes": [15, 31],
            },
        ),
    }
