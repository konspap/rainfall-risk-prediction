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

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/c28175a3-ba2f-4e79-9d5d-c1c63bc48590" />


---

# 📍 Rain Rate by Location

Different locations demonstrated different rainfall behavior.

Some locations consistently experienced higher rainfall rates, suggesting that geographical information contains important predictive value.

This justified retaining the `Location` feature during modeling.

### 📸

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/639dc640-13cd-4d98-8e7b-e63079cfd5a5" />

---

# 📅 Monthly Rain Rate

Monthly rainfall analysis revealed cyclical rainfall patterns throughout the year.

Rainfall probability increased during colder months and decreased during warmer periods.

This indicated that rainfall follows clear seasonal dynamics rather than random fluctuations.

### 📸

<img width="1440" height="800" alt="image" src="https://github.com/user-attachments/assets/5d9938d5-e4cf-4267-af37-e2b207e4906e" />


---

# 🌡️ Humidity vs Pressure Analysis

The scatterplot comparing humidity and pressure revealed important atmospheric behavior.

Rainy observations generally appeared:

- at higher humidity levels
- alongside lower atmospheric pressure

This aligned with real meteorological expectations and validated the dataset’s predictive structure.

### 📸

<img width="1280" height="960" alt="image" src="https://github.com/user-attachments/assets/5d809a90-fb9e-45b3-a279-47f3119bbbdc" />

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

<img width="1760" height="1280" alt="image" src="https://github.com/user-attachments/assets/3ae776ce-ff5f-4a61-ad7f-8894fc7cfc6a" />

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

### 📸

<img width="1120" height="800" alt="image" src="https://github.com/user-attachments/assets/a71ad3a0-c682-4c07-b26f-b12dc5bbd31c" />


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

### 📸

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/1d24df64-0ef2-4a55-b623-d92eae9741d3" />


---

# 📈 Precision-Recall Curve

Because of class imbalance, Precision-Recall analysis was especially important.

The model maintained strong precision across a wide recall range, indicating useful performance even when prioritizing rainy-day detection.

### 📸

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/b5db3be0-ca8e-42c4-b607-0ff5765ef26a" />

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

### 📸

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/78971088-f8d6-483e-968a-e508c85817f5" />

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

### 📸

<img width="1919" height="842" alt="image" src="https://github.com/user-attachments/assets/c9c70d59-87c8-4338-86b8-f4da91561b43" />
<img width="1920" height="818" alt="image" src="https://github.com/user-attachments/assets/83593837-85cb-479a-8c75-784be3da043f" />

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
