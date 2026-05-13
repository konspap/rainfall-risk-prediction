# Melbourne Rainfall Risk Prediction System

An end-to-end machine learning project that predicts whether it will rain in the Melbourne area using historical Australian weather data.

This repository upgrades a basic rainfall classifier into a professional portfolio project with modular code, feature engineering, model comparison, threshold tuning, evaluation reports, saved model artifacts, and a Streamlit prediction app.

---

## Project Overview

Weather prediction is a high-impact classification problem where accuracy alone is not enough. Missing an actual rainy day can be more costly than incorrectly warning about rain, so this project focuses on recall, F1-score, ROC-AUC, PR-AUC, and threshold tuning.

The model predicts:

```text
RainToday: Yes / No
```

The dataset originally contains `RainToday` and `RainTomorrow`. To make the task clearer and avoid target confusion:

- original `RainToday` becomes `RainYesterday`
- original `RainTomorrow` becomes the new prediction target: `RainToday`

---

## Business / Real-World Framing

The goal is not only to build a classifier, but to build a rainfall risk system that could support:

- daily commuting decisions
- event planning
- logistics scheduling
- basic weather-risk alerts
- operational planning for outdoor services

For this reason, the project prioritizes detecting rainy days instead of optimizing accuracy only.

---

## Dataset

The project uses the Australian weather dataset from the IBM / Coursera data source used in the original project workflow.

The model is localized to the Melbourne region:

- Melbourne
- MelbourneAirport
- Watsonia

A localized model is more realistic because rainfall patterns differ significantly across Australian regions.

---

## Key Improvements Over the Original Version

| Area | Original Version | Upgraded Version |
|---|---|---|
| Code structure | Single Python script | Modular production-style package |
| Missing values | `dropna()` | Pipeline-based imputers |
| Models | Logistic Regression, Random Forest | Baseline, Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting |
| Metrics | Accuracy, classification report | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| Threshold | Default 0.50 | Tuned threshold for rainfall recall |
| Feature engineering | Season | Season, month, temp range, humidity change, pressure drop, temp change, wind ratio |
| Outputs | Plots only | Saved model, metrics JSON, evaluation figures, Streamlit app |
| Deployment | None | Streamlit app ready |

---

## Machine Learning Workflow

1. Load weather data
2. Rename target columns for leakage-safe framing
3. Filter to Melbourne-area locations
4. Engineer domain-specific weather features
5. Split data using stratified sampling
6. Preprocess numeric and categorical columns separately
7. Train multiple models with cross-validation
8. Select the best model using F1-score by default
9. Tune the classification threshold for rainy-day recall
10. Generate exploratory visualizations to support the weather story
11. Save metrics, model artifact, and evaluation plots
12. Serve predictions through a Streamlit app

---

## Feature Engineering

Additional weather-specific features include:

- `Season`
- `Month`
- `TempRange = MaxTemp - MinTemp`
- `HumidityChange = Humidity3pm - Humidity9am`
- `PressureDrop = Pressure9am - Pressure3pm`
- `TempChange = Temp3pm - Temp9am`
- `WindGustTo3pmSpeedRatio = WindGustSpeed / WindSpeed3pm`

These features help the model capture daily weather dynamics instead of relying only on raw observations.

---


## Exploratory Data Analysis Visualizations

The training workflow also generates EDA plots so the repository demonstrates data storytelling, not just model training. These charts support the project theory before the model is introduced:

- Target distribution: shows class imbalance between rainy and non-rainy days
- Rain rate by season: supports the seasonality hypothesis
- Rain rate by location: compares Melbourne, Melbourne Airport, and Watsonia
- Monthly rain rate: shows month-level rainfall patterns
- Correlation heatmap: highlights relationships among weather measurements
- Humidity vs pressure scatter plot: visualizes how atmospheric conditions differ across rain outcomes

These figures are useful for the README, presentation, and interview discussions because they explain why certain features are meaningful.

---
## Models Tested

- Dummy baseline classifier
- Logistic Regression
- Random Forest
- Gradient Boosting
- HistGradientBoosting

The final model is selected based on validation and test-set performance, with special attention to rainy-day recall and F1-score.

---

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision for rainy days
- Recall for rainy days
- F1-score for rainy days
- ROC-AUC
- PR-AUC
- Confusion matrix
- Classification report

### Why Recall Matters

A false negative means the model predicted no rain, but it actually rained. In a rainfall-risk context, this is often worse than a false positive because users may be unprepared for rain.

---

## Repository Structure

```text
rainfall-risk-prediction/
├── app/
│   └── streamlit_app.py
├── config/
│   └── settings.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── modeling.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_features.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rainfall-risk-prediction.git
cd rainfall-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python -m src.train
```

Optional: train a national model using all Australian locations:

```bash
python -m src.train --national
```

Optional: optimize for higher rainy-day recall:

```bash
python -m src.train --scoring recall --min-recall 0.85
```

### 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Generated Artifacts

After training, the project generates:

```text
models/best_rainfall_model.joblib
reports/metrics.json
reports/figures/eda_target_distribution.png
reports/figures/eda_rain_rate_by_season.png
reports/figures/eda_rain_rate_by_location.png
reports/figures/eda_monthly_rain_rate.png
reports/figures/eda_correlation_heatmap.png
reports/figures/eda_humidity_pressure_scatter.png
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/precision_recall_curve.png
reports/figures/feature_importance.png
```

These files are intentionally ignored by Git because they can be regenerated.

---

## Example App Output

The Streamlit app returns:

- predicted class: `Rain` / `No Rain`
- probability of rain
- risk level: Low / Medium / High
- decision threshold used by the model

---

## Testing

Run unit tests with:

```bash
pytest
```

---

## Portfolio Notes

This project demonstrates:

- end-to-end supervised machine learning
- production-style project organization
- data leakage awareness
- preprocessing pipelines
- imputation and encoding
- hyperparameter tuning
- model comparison
- threshold tuning
- model persistence
- Streamlit deployment
- business-oriented ML communication

---

## Future Improvements

- Add SHAP explainability for local predictions
- Deploy the Streamlit app to Streamlit Community Cloud
- Add a time-based train/test split for stronger real-world validation
- Compare Melbourne-only and national models in a formal report
- Add live weather API integration
- Add model monitoring and data drift checks
- Containerize with Docker

---

## Author

Patricia Castresana

