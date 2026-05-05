"""Orchestrates the end-to-end sales analytics pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.analytics import export_preview_images
from src.clustering import run_segmentation_pipeline
from src.config import (
    DEFAULT_RECORDS,
    FORECAST_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SEGMENTS_PATH,
    ensure_project_dirs,
)
from src.data_generation import save_sales_data
from src.data_preprocessing import preprocess_pipeline
from src.forecasting import run_forecasting_pipeline


def outputs_exist() -> bool:
    required_files = [PROCESSED_DATA_PATH, SEGMENTS_PATH, FORECAST_PATH, METRICS_PATH]
    return all(path.exists() for path in required_files)


def build_metrics(
    processed_df,
    preprocessing_summary: dict,
    segmentation_bundle: dict,
    forecasting_bundle: dict,
) -> dict:
    outlier_count = preprocessing_summary["outliers"]["total_amount"]["count"]
    return {
        "dataset": "Synthetic retail sales dataset",
        "records": int(len(processed_df)),
        "unique_customers": int(processed_df["customer_id"].nunique()),
        "total_revenue": float(processed_df["total_amount"].sum()),
        "forecasting_model": forecasting_bundle["metrics"]["forecasting_model"],
        "r2_score": forecasting_bundle["metrics"]["r2_score"],
        "mae": forecasting_bundle["metrics"]["mae"],
        "rmse": forecasting_bundle["metrics"]["rmse"],
        "segmentation_model": "KMeans",
        "optimal_clusters": segmentation_bundle["metrics"]["optimal_clusters"],
        "silhouette_score": segmentation_bundle["metrics"]["silhouette_score"],
        "outlier_rate": float(outlier_count / max(len(processed_df), 1)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(n_records: int = DEFAULT_RECORDS, export_previews: bool = True) -> dict:
    ensure_project_dirs()
    save_sales_data(RAW_DATA_PATH, n_records=n_records)
    processed_df, preprocessing_summary = preprocess_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    segmentation_bundle = run_segmentation_pipeline(PROCESSED_DATA_PATH, SEGMENTS_PATH)
    forecasting_bundle = run_forecasting_pipeline(
        PROCESSED_DATA_PATH,
        FORECAST_PATH,
        MODEL_PATH,
    )

    metrics = build_metrics(
        processed_df=processed_df,
        preprocessing_summary=preprocessing_summary,
        segmentation_bundle=segmentation_bundle,
        forecasting_bundle=forecasting_bundle,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    bundle = {
        "sales_df": processed_df,
        "segments_df": segmentation_bundle["segments_df"],
        "segment_profiles": segmentation_bundle["profiles"],
        "forecast_df": forecasting_bundle["forecast_df"],
        "metrics": metrics,
        "raw_path": str(RAW_DATA_PATH),
        "processed_path": str(PROCESSED_DATA_PATH),
        "segments_path": str(SEGMENTS_PATH),
        "forecast_path": str(FORECAST_PATH),
        "metrics_path": str(METRICS_PATH),
    }

    if export_previews:
        export_preview_images(bundle)

    return bundle
