"""Data cleaning and feature engineering for the retail dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataPreprocessor:
    """Prepare raw retail transactions for downstream analytics models."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.original_shape = dataframe.shape
        self.outlier_summary: dict[str, dict[str, float | int]] = {}

    def handle_missing_values(self) -> "DataPreprocessor":
        missing = self.df.isnull().sum()
        if missing.sum() and "discount_percent" in self.df.columns:
            median_discount = float(self.df["discount_percent"].median())
            self.df["discount_percent"] = self.df["discount_percent"].fillna(median_discount)
        return self

    def detect_outliers(self, column: str) -> tuple[pd.Series, float, float]:
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        return outliers, float(lower_bound), float(upper_bound)

    def handle_outliers(self, columns: list[str] | None = None) -> "DataPreprocessor":
        if columns is None:
            columns = ["total_amount"]

        for column in columns:
            outliers, lower, upper = self.detect_outliers(column)
            original_column = f"{column}_original"
            flag_column = f"is_outlier_{column}"

            self.df[original_column] = self.df[column]
            self.df[flag_column] = outliers.astype(bool)
            self.df.loc[self.df[column] < lower, column] = lower
            self.df.loc[self.df[column] > upper, column] = upper
            self.outlier_summary[column] = {
                "count": int(outliers.sum()),
                "lower_bound": lower,
                "upper_bound": upper,
            }
        return self

    def feature_engineering(self) -> "DataPreprocessor":
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        self.df["revenue_per_unit"] = self.df["total_amount"] / self.df["quantity"]
        return self

    def get_processed_data(self) -> pd.DataFrame:
        return self.df

    def build_summary(self) -> dict:
        return {
            "original_shape": list(self.original_shape),
            "processed_shape": list(self.df.shape),
            "outliers": self.outlier_summary,
        }


def preprocess_pipeline(input_path: str | Path, output_path: str | Path) -> tuple[pd.DataFrame, dict]:
    dataframe = pd.read_csv(input_path)
    preprocessor = DataPreprocessor(dataframe)
    processed_df = (
        preprocessor.handle_missing_values()
        .handle_outliers()
        .feature_engineering()
        .get_processed_data()
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    return processed_df, preprocessor.build_summary()
