"""Exploratory data analysis visualizations for the rainfall project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.settings import FIGURES_DIR, POSITIVE_CLASS, TARGET_COLUMN


def _save_current_figure(filename: str) -> None:
    """Save the active matplotlib figure in the reports/figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Show whether the target is imbalanced."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Target Distribution: Rain vs No Rain")
    plt.xlabel("Rain Today")
    plt.ylabel("Number of Observations")
    _save_current_figure("eda_target_distribution.png")


def plot_rain_rate_by_season(df: pd.DataFrame) -> None:
    """Show seasonal rainfall patterns."""
    if "Season" not in df.columns:
        return
    order = ["Summer", "Autumn", "Winter", "Spring"]
    rain_rate = (
        df.assign(is_rain=df[TARGET_COLUMN].eq(POSITIVE_CLASS).astype(int))
        .groupby("Season")["is_rain"]
        .mean()
        .reindex(order)
        .dropna()
    )
    plt.figure(figsize=(8, 5))
    plt.bar(rain_rate.index, rain_rate.values)
    plt.title("Rain Rate by Australian Season")
    plt.xlabel("Season")
    plt.ylabel("Share of Rainy Days")
    _save_current_figure("eda_rain_rate_by_season.png")


def plot_rain_rate_by_location(df: pd.DataFrame) -> None:
    """Compare rainfall risk across Melbourne-area locations."""
    if "Location" not in df.columns:
        return
    rain_rate = (
        df.assign(is_rain=df[TARGET_COLUMN].eq(POSITIVE_CLASS).astype(int))
        .groupby("Location")["is_rain"]
        .mean()
        .sort_values()
    )
    plt.figure(figsize=(8, 5))
    plt.barh(rain_rate.index.astype(str), rain_rate.values)
    plt.title("Rain Rate by Location")
    plt.xlabel("Share of Rainy Days")
    plt.ylabel("Location")
    _save_current_figure("eda_rain_rate_by_location.png")


def plot_monthly_rain_rate(df: pd.DataFrame) -> None:
    """Show month-level rainfall seasonality."""
    if "Month" not in df.columns:
        return
    rain_rate = (
        df.assign(is_rain=df[TARGET_COLUMN].eq(POSITIVE_CLASS).astype(int))
        .groupby("Month")["is_rain"]
        .mean()
        .sort_index()
    )
    plt.figure(figsize=(9, 5))
    plt.plot(rain_rate.index, rain_rate.values, marker="o")
    plt.title("Monthly Rain Rate")
    plt.xlabel("Month")
    plt.ylabel("Share of Rainy Days")
    plt.xticks(range(1, 13))
    _save_current_figure("eda_monthly_rain_rate.png")


def plot_numeric_correlation_heatmap(df: pd.DataFrame) -> None:
    """Show correlation among key numerical weather variables."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        return
    preferred_columns = [
        "Rainfall",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Sunshine",
        "WindGustSpeed",
        "MaxTemp",
        "MinTemp",
        "TempRange",
        "HumidityChange",
        "PressureDrop",
    ]
    available = [column for column in preferred_columns if column in numeric_df.columns]
    if len(available) < 2:
        available = numeric_df.columns[:12].tolist()

    corr = numeric_df[available].corr()
    plt.figure(figsize=(11, 8))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap of Weather Variables")
    _save_current_figure("eda_correlation_heatmap.png")


def plot_humidity_pressure_relationship(df: pd.DataFrame) -> None:
    """Support the weather theory with a humidity/pressure scatter plot."""
    required = {"Humidity3pm", "Pressure3pm", TARGET_COLUMN}
    if not required.issubset(df.columns):
        return
    sample = df.dropna(subset=list(required)).sample(
        n=min(2500, len(df.dropna(subset=list(required)))), random_state=42
    )
    plt.figure(figsize=(8, 6))
    for label, group in sample.groupby(TARGET_COLUMN):
        plt.scatter(group["Pressure3pm"], group["Humidity3pm"], alpha=0.45, label=str(label), s=18)
    plt.title("Humidity vs Pressure by Rain Outcome")
    plt.xlabel("Pressure at 3pm")
    plt.ylabel("Humidity at 3pm")
    plt.legend(title="Rain Today")
    _save_current_figure("eda_humidity_pressure_scatter.png")


def save_eda_plots(df: pd.DataFrame) -> None:
    """Generate all EDA plots used in the report and README."""
    plot_target_distribution(df)
    plot_rain_rate_by_season(df)
    plot_rain_rate_by_location(df)
    plot_monthly_rain_rate(df)
    plot_numeric_correlation_heatmap(df)
    plot_humidity_pressure_relationship(df)
