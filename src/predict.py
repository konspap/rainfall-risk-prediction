"""Prediction utilities for the trained rainfall model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.settings import MODEL_PATH, POSITIVE_CLASS


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the persisted model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run: python -m src.train")
    return joblib.load(model_path)


def predict_rainfall(input_data: dict[str, Any] | pd.DataFrame, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Predict rainfall risk for one or more observations."""
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    threshold = bundle["threshold"]

    if isinstance(input_data, dict):
        X = pd.DataFrame([input_data])
    else:
        X = input_data.copy()

    probabilities = model.predict_proba(X)
    class_index = list(model.classes_).index(POSITIVE_CLASS)
    rain_probability = probabilities[:, class_index]
    predictions = [POSITIVE_CLASS if p >= threshold else "No" for p in rain_probability]

    risk_levels = []
    for probability in rain_probability:
        if probability >= 0.70:
            risk_levels.append("High")
        elif probability >= 0.40:
            risk_levels.append("Medium")
        else:
            risk_levels.append("Low")

    return {
        "prediction": predictions,
        "rain_probability": rain_probability.round(4).tolist(),
        "risk_level": risk_levels,
        "threshold": threshold,
    }
