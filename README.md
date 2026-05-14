🌧️ Melbourne Rainfall Risk Prediction
Interactive machine learning application for predicting rainfall probability using Australian weather observations and an end-to-end ML pipeline.
🔗 Live Demo: Rainfall Risk Prediction App
🔗 GitHub Repository: GitHub Repository

📌 Project Overview
This project was developed as an end-to-end machine learning pipeline focused on predicting rainfall risk based on historical weather conditions.
The goal was not simply to train a model, but to build a complete production-style workflow including:
* data preprocessing
* exploratory data analysis (EDA)
* feature engineering
* model training & comparison
* evaluation & threshold tuning
* model persistence
* Streamlit deployment
The final result is an interactive web application where users can input weather conditions and receive a rainfall probability prediction in real time.
🎯 Business Problem

Rainfall prediction has real-world value in areas such as:

agriculture
transportation
logistics
event planning
operational risk management

Unexpected rainfall can create:

delays
financial losses
safety risks
planning inefficiencies

The objective of this project was to build a system capable of estimating:

Probability of rainfall given atmospheric conditions

From a machine learning perspective, this is a:

Binary Classification Problem

Target variable:

RainToday → Yes / No
🧠 Machine Learning Objective

The model attempts to learn:

P(Rain = Yes | weather conditions)

using historical meteorological observations.

Rather than relying on a single variable, the model learns patterns from:

humidity
pressure
temperature
sunshine
cloud coverage
wind
temporal features
geographical information
📂 Project Structure
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
📊 Dataset

The project uses Australian weather observations including:

temperature
humidity
atmospheric pressure
rainfall
sunshine hours
wind speed
cloud coverage
location
seasonal/time information

The dataset contains both:

numerical variables
categorical variables

which required significant preprocessing before modeling.

🧹 Data Preprocessing

The raw dataset could not be used directly because it contained:

missing values
mixed feature types
inconsistent scales

The preprocessing pipeline handled:

Numerical Features
missing value imputation
scaling
Categorical Features
missing category handling
encoding

This transformed the raw weather observations into a machine-learning-ready feature matrix.

⚙️ Feature Engineering

One of the most important parts of the project was feature engineering.

Instead of using only raw variables, additional meteorological features were created to better represent weather dynamics.

Examples include:

Feature	Description
TempRange	MaxTemp - MinTemp
HumidityChange	Humidity3pm - Humidity9am
PressureDrop	Pressure9am - Pressure3pm
Season	Seasonal rainfall behavior
Month	Temporal rainfall patterns

These engineered features helped the model capture:

atmospheric changes
seasonal behavior
weather instability patterns

rather than relying only on isolated measurements.

📈 Exploratory Data Analysis (EDA)

EDA was critical because it allowed us to understand:

rainfall behavior
seasonality
correlations
class imbalance
predictive signals

before training any machine learning models.

🌦️ Rain Rate by Season

The analysis showed that rainfall frequency changes significantly across seasons.

Winter and spring exhibited noticeably higher rainfall probability compared to summer.

This confirmed that:

rainfall is highly seasonal
temporal features are important predictors

📸 Add Screenshot Here:

eda_rain_rate_by_season.png
📍 Rain Rate by Location

Different locations displayed different rainfall behavior.

Some locations consistently experienced higher rainfall rates than others, suggesting that geographical information contains meaningful predictive signal.

This justified retaining the Location feature during modeling.

📸 Add Screenshot Here:

eda_rain_rate_by_location.png
📅 Monthly Rain Rate

Monthly analysis revealed cyclical rainfall patterns throughout the year.

Rainfall probability increased during colder months and decreased during warmer months.

This indicated that rainfall is not random, but follows temporal seasonal dynamics.

📸 Add Screenshot Here:

eda_monthly_rain_rate.png
🌡️ Humidity vs Pressure Analysis

The scatterplot comparing humidity and pressure revealed interesting atmospheric behavior.

Rainy observations generally appeared:

at higher humidity levels
often alongside lower atmospheric pressure

This aligned with real meteorological expectations and validated the dataset’s predictive structure.

📸 Add Screenshot Here:

eda_humidity_pressure_scatter.png
🔥 Correlation Heatmap

The correlation heatmap helped identify relationships between weather variables.

Important findings included:

humidity positively associated with rainfall
pressure negatively associated with rainfall
sunshine negatively associated with rainfall
strong relationships between temperature-based variables

This analysis helped confirm that the dataset contains meaningful predictive structure rather than random noise.

📸 Add Screenshot Here:

eda_correlation_heatmap.png
⚠️ Target Distribution & Class Imbalance

The dataset exhibited significant class imbalance.

There were substantially more:

non-rainy days
than:
rainy days

This was important because accuracy alone could become misleading.

For example:
A model predicting always “No Rain” could still achieve high accuracy while being practically useless.

Because of this, evaluation focused heavily on:

Recall
F1-score
ROC-AUC
Precision-Recall performance

📸 Add Screenshot Here:

eda_target_distribution.png
🤖 Model Training & Comparison

Multiple machine learning models were evaluated.

The objective was evidence-based model selection rather than choosing a model arbitrarily.

Models tested included:

Model	Purpose
Dummy Classifier	Baseline
Logistic Regression	Linear baseline
Random Forest	Nonlinear ensemble
Gradient Boosting	Advanced boosting
HistGradientBoostingClassifier	Final selected model
🧪 Why HistGradientBoostingClassifier Was Selected

The final model chosen was:

HistGradientBoostingClassifier

because it achieved the strongest balance across:

recall
ROC-AUC
ranking quality
nonlinear learning capability

The model performed especially well because weather prediction involves:

nonlinear interactions
atmospheric dependencies
complex feature relationships

Boosting methods are particularly effective for structured/tabular data problems like this one.

📊 Model Performance
Final Metrics
Metric	Score
Accuracy	~0.81
Recall	~0.80
F1-score	~0.67
ROC-AUC	~0.90
PR-AUC	~0.78
📉 ROC Curve

The ROC curve demonstrated strong class separation capability.

An AUC score around 0.90 indicates that the model distinguishes rainy vs non-rainy days very effectively.

📸 Add Screenshot Here:

roc_curve.png
📈 Precision-Recall Curve

Because of class imbalance, Precision-Recall analysis was especially important.

The model maintained strong precision across a wide recall range, indicating useful performance even when prioritizing rainy-day detection.

📸 Add Screenshot Here:

precision_recall_curve.png
🔲 Confusion Matrix

The confusion matrix illustrates the trade-off achieved after threshold tuning.

At threshold = 0.18:

recall improved significantly
false negatives decreased
more rainy days were successfully detected

This threshold was intentionally selected because missing rainfall events was considered more costly than issuing additional false alarms.

📸 Add Screenshot Here:

confusion_matrix.png
🎯 Threshold Tuning

Instead of using the default threshold:

0.50

the model used:

0.18

This decision was business-driven.

The objective was to:

reduce missed rainfall events
improve recall
capture more true rainy days

In weather-risk scenarios:
missing rainfall can often be more problematic than generating occasional false alerts.

💾 Model Persistence

The trained pipeline was saved using:

joblib

File:

best_rainfall_model.joblib

This allowed:

fast loading
no retraining during deployment
efficient inference
🌐 Streamlit Application

The machine learning pipeline was deployed using Streamlit.

The application allows users to:

input weather conditions
adjust atmospheric variables
receive rainfall probability predictions in real time

The app acts as a lightweight production interface for the trained model.

📸 Add App Screenshot Here

🚀 Live Application

🔗 Public Deployment:

Live Streamlit App

🛠️ Technologies Used
Programming
Python
Data Science
pandas
numpy
Visualization
matplotlib
seaborn
Machine Learning
scikit-learn
HistGradientBoostingClassifier
Deployment
Streamlit
Model Persistence
joblib
📦 Installation

Clone the repository:

git clone https://github.com/konspap/rainfall-risk-prediction.git

Move into the project directory:

cd rainfall-risk-prediction

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app/streamlit_app.py
🧪 Testing

Tests are included inside the tests/ directory.

Run tests with:

pytest
📌 Key Takeaways

This project demonstrates:

end-to-end ML workflow design
structured preprocessing pipelines
feature engineering
model comparison & evaluation
threshold optimization
deployment of ML systems
business-oriented ML reasoning
communication of technical decisions

Most importantly, the project focuses not only on achieving good metrics, but also on understanding:

why the model works
why specific decisions were made
how business objectives influence ML design choices
