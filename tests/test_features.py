import pandas as pd

from src.data_preprocessing import add_weather_features, date_to_australian_season


def test_date_to_australian_season():
    assert date_to_australian_season(pd.Timestamp("2020-01-01")) == "Summer"
    assert date_to_australian_season(pd.Timestamp("2020-04-01")) == "Autumn"
    assert date_to_australian_season(pd.Timestamp("2020-07-01")) == "Winter"
    assert date_to_australian_season(pd.Timestamp("2020-10-01")) == "Spring"


def test_add_weather_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "Date": ["2020-06-01"],
            "MaxTemp": [20.0],
            "MinTemp": [10.0],
            "Humidity3pm": [60.0],
            "Humidity9am": [70.0],
            "Pressure9am": [1018.0],
            "Pressure3pm": [1012.0],
            "Temp3pm": [18.0],
            "Temp9am": [12.0],
            "WindGustSpeed": [30.0],
            "WindSpeed3pm": [15.0],
        }
    )
    output = add_weather_features(df)
    assert output.loc[0, "Season"] == "Winter"
    assert output.loc[0, "TempRange"] == 10.0
    assert output.loc[0, "HumidityChange"] == -10.0
    assert output.loc[0, "PressureDrop"] == 6.0
    assert output.loc[0, "TempChange"] == 6.0
    assert output.loc[0, "WindGustTo3pmSpeedRatio"] == 2.0
