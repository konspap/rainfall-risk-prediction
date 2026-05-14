# 🌧️ Melbourne Rainfall Risk Prediction

Interactive machine learning application for predicting rainfall probability using Australian weather observations and an end-to-end ML pipeline.

---

## 🔗 Live Demo

**Streamlit Application:**  
https://rainfall-risk-prediction-pgzhh6wjoagrkwgfmfax39.streamlit.app/

**GitHub Repository:**  
https://github.com/konspap/rainfall-risk-prediction

---

# 📌 Project Overview

This project was developed as a complete **end-to-end machine learning pipeline** focused on predicting rainfall risk using historical weather observations from Australia.

The objective was not simply to train a machine learning model, but to design a production-style workflow including:

- data preprocessing
- exploratory data analysis (EDA)
- feature engineering
- model training & comparison
- evaluation & threshold tuning
- model persistence
- interactive deployment

The final result is an interactive **Streamlit web application** where users can input atmospheric conditions and receive a real-time rainfall risk prediction.

---

# 🎯 Business Problem

Rainfall prediction has real-world value across multiple industries including:

- agriculture
- transportation
- logistics
- event planning
- operational risk management

Unexpected rainfall can create:

- delays
- financial losses
- safety risks
- planning inefficiencies

The purpose of this project was to build a system capable of estimating:

> **Probability of rainfall given atmospheric conditions**

From a machine learning perspective, this is a:

## **Binary Classification Problem**

### Target Variable

```python
RainToday → Yes / No
```

---

# 🧠 Machine Learning Objective

The model attempts to learn:

```python
P(Rain = Yes | weather conditions)
```

using historical meteorological observations.

Rather than relying on a single variable, the model learns patterns from:

- humidity
- atmospheric pressure
- temperature
- sunshine
- cloud coverage
- wind conditions
- temporal features
- geographical information

---

# 📂 Project Structure

```text
rainfall-risk-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_rainfall_model.joblib
│
├── reports/
│   ├── figures/
│   └── metrics.json
│
├── src/
│   ├── data_preprocessing.py
│   ├── modeling.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── visualize.py
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The dataset contains Australian weather observations including:

- temperature
- humidity
- atmospheric pressure
- rainfall
- sunshine hours
- wind speed
- cloud coverage
- location
- seasonal information

The dataset includes both:

- **numerical variables**
- **categorical variables**

which required extensive preprocessing before modeling.

---

# 🧹 Data Preprocessing

The raw dataset could not be used directly because it contained:

- missing values
- mixed feature types
- inconsistent scales

The preprocessing pipeline handled all transformations automatically.

## Numerical Features

- missing value imputation
- scaling

## Categorical Features

- missing category handling
- encoding

This transformed the raw observations into a machine-learning-ready feature matrix.

---

# ⚙️ Feature Engineering

One of the most important parts of the project was feature engineering.

Instead of relying only on raw variables, additional meteorological features were created to better represent atmospheric behavior and weather instability.

## Engineered Features

| Feature | Description |
|---|---|
| `TempRange` | `MaxTemp - MinTemp` |
| `HumidityChange` | `Humidity3pm - Humidity9am` |
| `PressureDrop` | `Pressure9am - Pressure3pm` |
| `Season` | Seasonal rainfall behavior |
| `Month` | Temporal rainfall patterns |

These engineered variables helped the model capture:

- atmospheric changes
- seasonal dynamics
- weather instability patterns

rather than isolated measurements alone.

---

# 📈 Exploratory Data Analysis (EDA)

EDA was critical because it allowed us to understand:

- rainfall behavior
- seasonality
- correlations
- class imbalance
- predictive signals

before training any machine learning models.

---

# 🌦️ Rain Rate by Season

The analysis showed that rainfall frequency changes significantly across seasons.

- **Winter** and **Spring** exhibited the highest rainfall probability.
- **Summer** displayed the lowest rainfall frequency.

This confirmed that:

- rainfall is highly seasonal
- temporal variables contain strong predictive signal

### 📸 

``` 
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/923b5c20-3d76-4a5c-9634-51ced66632c5" />

```
<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/c28175a3-ba2f-4e79-9d5d-c1c63bc48590" />


---

# 📍 Rain Rate by Location

Different locations demonstrated different rainfall behavior.

Some locations consistently experienced higher rainfall rates, suggesting that geographical information contains important predictive value.

This justified retaining the `Location` feature during modeling.

### 📸 Suggested Screenshot

```text
eda_rain_rate_by_location.png
```

---

# 📅 Monthly Rain Rate

Monthly rainfall analysis revealed cyclical rainfall patterns throughout the year.

Rainfall probability increased during colder months and decreased during warmer periods.

This indicated that rainfall follows clear seasonal dynamics rather than random fluctuations.

### 📸 Suggested Screenshot

```text
eda_monthly_rain_rate.png
```

---

# 🌡️ Humidity vs Pressure Analysis

The scatterplot comparing humidity and pressure revealed important atmospheric behavior.

Rainy observations generally appeared:

- at higher humidity levels
- alongside lower atmospheric pressure

This aligned with real meteorological expectations and validated the dataset’s predictive structure.

### 📸 Suggested Screenshot

```text
eda_humidity_pressure_scatter.png
```

---

# 🔥 Correlation Heatmap

The correlation heatmap helped identify relationships between weather variables.

Important findings included:

- humidity positively associated with rainfall
- pressure negatively associated with rainfall
- sunshine negatively associated with rainfall
- strong relationships between temperature-based variables

This analysis confirmed that the dataset contains meaningful predictive structure rather than random noise.

### 📸 Suggested Screenshot

```text
eda_correlation_heatmap.png
```

---

# ⚠️ Target Distribution & Class Imbalance

The dataset exhibited significant class imbalance.

There were substantially more:

- non-rainy days

than:

- rainy days

This was extremely important because **accuracy alone becomes misleading** in imbalanced datasets.

For example:

> A model predicting always “No Rain” could still achieve high accuracy while being practically useless.

Because of this, evaluation focused heavily on:

- Recall
- F1-score
- ROC-AUC
- Precision-Recall performance

### 📸 Suggested Screenshot

```text
eda_target_distribution.png
```

---

# 🤖 Model Training & Comparison

Multiple machine learning models were evaluated.

The objective was evidence-based model selection rather than choosing a model arbitrarily.

## Models Tested

| Model | Purpose |
|---|---|
| Dummy Classifier | Baseline |
| Logistic Regression | Linear baseline |
| Random Forest | Nonlinear ensemble |
| Gradient Boosting | Advanced boosting |
| HistGradientBoostingClassifier | Final selected model |

---

# 🧪 Why HistGradientBoostingClassifier Was Selected

The final model selected was:

## **HistGradientBoostingClassifier**

because it achieved the strongest balance across:

- recall
- ROC-AUC
- ranking quality
- nonlinear learning capability

The model performed especially well because weather prediction involves:

- nonlinear interactions
- atmospheric dependencies
- complex feature relationships

Boosting methods are particularly effective for structured/tabular datasets like this one.

---

# 📊 Model Performance

## Final Metrics

| Metric | Score |
|---|---|
| Accuracy | ~0.81 |
| Recall | ~0.80 |
| F1-score | ~0.67 |
| ROC-AUC | ~0.90 |
| PR-AUC | ~0.78 |

---

# 📉 ROC Curve

The ROC curve demonstrated strong class separation capability.

An AUC score around **0.90** indicates that the model distinguishes rainy vs non-rainy days very effectively.

### 📸 Suggested Screenshot

```text
roc_curve.png
```

---

# 📈 Precision-Recall Curve

Because of class imbalance, Precision-Recall analysis was especially important.

The model maintained strong precision across a wide recall range, indicating useful performance even when prioritizing rainy-day detection.

### 📸 Suggested Screenshot

```text
precision_recall_curve.png
```

---

# 🔲 Confusion Matrix

The confusion matrix illustrates the trade-off achieved after threshold tuning.

At:

```python
threshold = 0.18
```

the model achieved:

- significantly improved recall
- fewer false negatives
- better rainy-day detection

This threshold was intentionally selected because missing rainfall events was considered more costly than generating additional false alerts.

### 📸 Suggested Screenshot

```text
confusion_matrix.png
```

---

# 🎯 Threshold Tuning

Instead of using the default classification threshold:

```python
0.50
```

the model used:

```python
0.18
```

This was a business-driven decision.

The goal was to:

- reduce missed rainfall events
- improve recall
- detect more true rainy days

In weather-risk scenarios:

> missing rainfall events is often more costly than generating occasional false alarms.

---

# 💾 Model Persistence

The trained pipeline was saved using:

```python
joblib
```

File:

```text
best_rainfall_model.joblib
```

This enabled:

- fast loading
- no retraining during deployment
- efficient inference

---

# 🌐 Streamlit Application

The machine learning pipeline was deployed using **Streamlit**.

The application allows users to:

- input weather conditions
- adjust atmospheric variables
- receive rainfall probability predictions in real time

The app acts as a lightweight production interface for the trained model.

### 📸 Suggested Screenshot

```text
streamlit_app.png
```

---

# 🚀 Live Application

## 🔗 Public Deployment

https://rainfall-risk-prediction-pgzhh6wjoagrkwgfmfax39.streamlit.app/

---

# 🛠️ Technologies Used

## Programming

- Python

## Data Science

- pandas
- numpy

## Visualization

- matplotlib
- seaborn

## Machine Learning

- scikit-learn
- HistGradientBoostingClassifier

## Deployment

- Streamlit

## Model Persistence

- joblib

---

# 📦 Installation

## Clone the repository

```bash
git clone https://github.com/konspap/rainfall-risk-prediction.git
```

## Move into the project directory

```bash
cd rainfall-risk-prediction
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

# 🧪 Testing

Tests are included inside the `tests/` directory.

Run tests with:

```bash
pytest
```

---

# 📌 Key Takeaways

This project demonstrates:

- end-to-end ML workflow design
- structured preprocessing pipelines
- feature engineering
- model comparison & evaluation
- threshold optimization
- deployment of ML systems
- business-oriented ML reasoning
- communication of technical decisions

Most importantly, the project focuses not only on achieving good metrics, but also on understanding:

- why the model works
- why specific decisions were made
- how business objectives influence ML design choices
