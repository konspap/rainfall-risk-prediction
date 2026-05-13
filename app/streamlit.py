"""Streamlit app for Melbourne rainfall risk prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config.settings import MODEL_PATH  # noqa: E402
from src.predict import predict_rainfall  # noqa: E402

st.set_page_config(page_title="Melbourne Rainfall Risk Predictor", page_icon="🌧️", layout="wide")

st.title("🌧️ Melbourne Rainfall Risk Prediction")
st.write(
    "This app predicts the probability of rain using a trained machine learning model. "
    "It is designed as an end-to-end portfolio project: preprocessing, model training, "
    "threshold tuning, evaluation, and deployment."
)

if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python -m src.train` first, then restart the app.")
    st.stop()

left, right = st.columns(2)

with left:
    location = st.selectbox("Location", ["Melbourne", "MelbourneAirport", "Watsonia"])
    season = st.selectbox("Season", ["Summer", "Autumn", "Winter", "Spring"])
    month = st.slider("Month", 1, 12, 6)
    rain_yesterday = st.selectbox("Did it rain yesterday?", ["No", "Yes"])
    rainfall = st.number_input("Rainfall yesterday (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.2)
    evaporation = st.number_input("Evaporation", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
    sunshine = st.number_input("Sunshine hours", min_value=0.0, max_value=15.0, value=7.0, step=0.1)

with right:
    min_temp = st.number_input("Min temperature (°C)", value=10.0, step=0.1)
    max_temp = st.number_input("Max temperature (°C)", value=20.0, step=0.1)
    temp_9am = st.number_input("Temperature 9am (°C)", value=14.0, step=0.1)
    temp_3pm = st.number_input("Temperature 3pm (°C)", value=19.0, step=0.1)
    humidity_9am = st.slider("Humidity 9am (%)", 0, 100, 70)
    humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, 55)
    pressure_9am = st.number_input("Pressure 9am (hPa)", value=1015.0, step=0.1)
    pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0, step=0.1)

st.subheader("Wind and cloud conditions")
wind_col1, wind_col2, wind_col3 = st.columns(3)
with wind_col1:
    wind_gust_dir = st.selectbox("Wind gust direction", ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])
    wind_gust_speed = st.number_input("Wind gust speed", min_value=0.0, value=35.0, step=1.0)
with wind_col2:
    wind_dir_9am = st.selectbox("Wind direction 9am", ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])
    wind_speed_9am = st.number_input("Wind speed 9am", min_value=0.0, value=12.0, step=1.0)
with wind_col3:
    wind_dir_3pm = st.selectbox("Wind direction 3pm", ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])
    wind_speed_3pm = st.number_input("Wind speed 3pm", min_value=0.0, value=18.0, step=1.0)

cloud_9am = st.slider("Cloud cover 9am", 0, 9, 5)
cloud_3pm = st.slider("Cloud cover 3pm", 0, 9, 5)

input_row = {
    "Location": location,
    "MinTemp": min_temp,
    "MaxTemp": max_temp,
    "Rainfall": rainfall,
    "Evaporation": evaporation,
    "Sunshine": sunshine,
    "WindGustDir": wind_gust_dir,
    "WindGustSpeed": wind_gust_speed,
    "WindDir9am": wind_dir_9am,
    "WindDir3pm": wind_dir_3pm,
    "WindSpeed9am": wind_speed_9am,
    "WindSpeed3pm": wind_speed_3pm,
    "Humidity9am": humidity_9am,
    "Humidity3pm": humidity_3pm,
    "Pressure9am": pressure_9am,
    "Pressure3pm": pressure_3pm,
    "Cloud9am": cloud_9am,
    "Cloud3pm": cloud_3pm,
    "Temp9am": temp_9am,
    "Temp3pm": temp_3pm,
    "RainYesterday": rain_yesterday,
    "Month": month,
    "Season": season,
    "TempRange": max_temp - min_temp,
    "HumidityChange": humidity_3pm - humidity_9am,
    "PressureDrop": pressure_9am - pressure_3pm,
    "TempChange": temp_3pm - temp_9am,
    "WindGustTo3pmSpeedRatio": wind_gust_speed / wind_speed_3pm if wind_speed_3pm else 0,
}

if st.button("Predict rainfall risk", type="primary"):
    result = predict_rainfall(input_row)
    probability = result["rain_probability"][0]
    prediction = result["prediction"][0]
    risk_level = result["risk_level"][0]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Prediction", prediction)
    metric_cols[1].metric("Rain Probability", f"{probability:.1%}")
    metric_cols[2].metric("Risk Level", risk_level)

    st.progress(min(float(probability), 1.0))
    st.caption(f"Decision threshold used by model: {result['threshold']:.2f}")

    with st.expander("Input data sent to the model"):
        st.dataframe(pd.DataFrame([input_row]))
