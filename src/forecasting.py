"""Revenue forecasting pipeline using a Random Forest regressor."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RANDOM_STATE

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


FEATURE_COLUMNS = [
    "trend_index",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "month",
    "quarter",
    "is_weekend",
    "orders",
    "avg_discount",
    "avg_quantity",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]


def build_daily_revenue_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    sales_df = dataframe.copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.floor("D")

    daily = (
        sales_df.groupby("date")
        .agg(
            actual_revenue=("total_amount", "sum"),
            orders=("transaction_id", "count"),
            avg_discount=("discount_percent", "mean"),
            avg_quantity=("quantity", "mean"),
        )
        .sort_index()
        .reset_index()
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["trend_index"] = np.arange(len(daily))
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["lag_1"] = daily["actual_revenue"].shift(1)
    daily["lag_7"] = daily["actual_revenue"].shift(7)
    daily["lag_14"] = daily["actual_revenue"].shift(14)
    daily["lag_28"] = daily["actual_revenue"].shift(28)
    daily["rolling_mean_7"] = daily["actual_revenue"].shift(1).rolling(7).mean()
    daily["rolling_mean_14"] = daily["actual_revenue"].shift(1).rolling(14).mean()
    daily["rolling_std_7"] = (
        daily["actual_revenue"].shift(1).rolling(7).std().fillna(0.0)
    )

    return daily.dropna().reset_index(drop=True)


def train_forecaster(daily_frame: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame, dict]:
    split_index = int(len(daily_frame) * 0.8)
    train_df = daily_frame.iloc[:split_index].copy()
    test_df = daily_frame.iloc[split_index:].copy()

    model = RandomForestRegressor(
        n_estimators=450,
        max_depth=16,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["actual_revenue"])

    test_df["predicted_revenue"] = model.predict(test_df[FEATURE_COLUMNS])
    train_df["predicted_revenue"] = model.predict(train_df[FEATURE_COLUMNS])
    train_df["split"] = "train"
    test_df["split"] = "test"

    forecast_frame = pd.concat([train_df, test_df], ignore_index=True)
    metrics = {
        "forecasting_model": "RandomForestRegressor",
        "r2_score": float(r2_score(test_df["actual_revenue"], test_df["predicted_revenue"])),
        "mae": float(mean_absolute_error(test_df["actual_revenue"], test_df["predicted_revenue"])),
        "rmse": float(
            np.sqrt(mean_squared_error(test_df["actual_revenue"], test_df["predicted_revenue"]))
        ),
    }
    return model, forecast_frame, metrics


def run_forecasting_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
) -> dict:
    dataframe = pd.read_csv(input_path)
    daily_frame = build_daily_revenue_frame(dataframe)
    model, forecast_frame, metrics = train_forecaster(daily_frame)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_frame.to_csv(output_path, index=False)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "forecast_df": forecast_frame,
        "metrics": metrics,
        "output_path": output_path,
        "model_path": model_path,
    }
