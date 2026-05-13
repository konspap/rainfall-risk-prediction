"""Project configuration constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"

LOCAL_LOCATIONS = ["Melbourne", "MelbourneAirport", "Watsonia"]
TARGET_COLUMN = "RainToday"
POSITIVE_CLASS = "Yes"
RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "weatherAUS-2.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "melbourne_weather_processed.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_rainfall_model.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
