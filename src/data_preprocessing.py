"""Data loading, cleaning, leakage-safe target setup, and feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from config.settings import DATA_URL, LOCAL_LOCATIONS, RAW_DATA_PATH, TARGET_COLUMN


def load_data(
    source: str | Path | None = None,
    cache_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Load weather data from local cache first, otherwise download it.

    Loading order:
    1. data/raw/weatherAUS-2.csv if it already exists
    2. user-provided source, if given
    3. DATA_URL from config/settings.py

    If the online download fails, the user gets a clear instruction instead of a
    long urllib traceback.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return pd.read_csv(cache_path)

    data_source = source if source is not None else DATA_URL

    try:
        df = pd.read_csv(data_source)
    except Exception as exc:
        raise RuntimeError(
            "\nCould not load the dataset from the internet.\n\n"
            "Please download the dataset manually and save it here:\n"
            f"  {cache_path}\n\n"
            "Expected file name:\n"
            "  weatherAUS-2.csv\n\n"
            "Then run again:\n"
            "  python -m src.train\n\n"
            f"Original error: {exc}"
        ) from exc

    df.to_csv(cache_path, index=False)
    return df


def date_to_australian_season(date_value: pd.Timestamp) -> str:
    """Map a date to the corresponding Australian season."""
    if pd.isna(date_value):
        return "Unknown"

    month = date_value.month

    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rename target-related columns so the prediction task is easier to understand.

    Original dataset:
    - RainToday: rain on the observation day
    - RainTomorrow: rain on the next day

    Project framing:
    - RainYesterday: known previous rainfall flag
    - RainToday: target variable to predict
    """
    required = {"RainToday", "RainTomorrow"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required target columns: {sorted(missing)}")

    return df.rename(
        columns={
            "RainToday": "RainYesterday",
            "RainTomorrow": TARGET_COLUMN,
        }
    )


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weather-specific features that add useful domain signal."""
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["Season"] = df["Date"].apply(date_to_australian_season)
        df = df.drop(columns=["Date"])
    else:
        df["Month"] = pd.NA
        df["Season"] = "Unknown"

    if {"MaxTemp", "MinTemp"}.issubset(df.columns):
        df["TempRange"] = df["MaxTemp"] - df["MinTemp"]

    if {"Humidity3pm", "Humidity9am"}.issubset(df.columns):
        df["HumidityChange"] = df["Humidity3pm"] - df["Humidity9am"]

    if {"Pressure9am", "Pressure3pm"}.issubset(df.columns):
        df["PressureDrop"] = df["Pressure9am"] - df["Pressure3pm"]

    if {"Temp3pm", "Temp9am"}.issubset(df.columns):
        df["TempChange"] = df["Temp3pm"] - df["Temp9am"]

    if {"WindGustSpeed", "WindSpeed3pm"}.issubset(df.columns):
        denominator = df["WindSpeed3pm"].replace(0, float("nan"))
        ratio = df["WindGustSpeed"] / denominator
        df["WindGustTo3pmSpeedRatio"] = ratio.clip(lower=0, upper=10)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_modeling_dataset(
    df: pd.DataFrame,
    locations: Iterable[str] = LOCAL_LOCATIONS,
    localized: bool = True,
) -> pd.DataFrame:
    """Prepare the final dataset for training.

    Missing values are not removed here. They are handled later inside the
    scikit-learn preprocessing pipeline with imputers.
    """
    df = prepare_target(df)
    df = df.dropna(subset=[TARGET_COLUMN])

    if localized:
        if "Location" not in df.columns:
            raise ValueError("Column 'Location' is required for localized training.")

        df = df[df["Location"].isin(list(locations))]

    df = add_weather_features(df)

    if df.empty:
        raise ValueError(
            "The modeling dataset is empty after filtering. "
            "Check the selected locations or the input dataset."
        )

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y for model training."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} was not found.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y