"""Configuration and filesystem paths for the project."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"

RAW_DATA_PATH = RAW_DIR / "sales_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "sales_data_cleaned.csv"
SEGMENTS_PATH = PROCESSED_DIR / "customer_segments.csv"
FORECAST_PATH = PROCESSED_DIR / "daily_revenue_forecast.csv"
METRICS_PATH = ROOT_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "random_forest_forecaster.joblib"

DEFAULT_RECORDS = 25_000
RANDOM_STATE = 42


def ensure_project_dirs() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        VISUALIZATIONS_DIR,
        SCREENSHOTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
