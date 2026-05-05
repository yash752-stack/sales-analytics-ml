"""Dashboard analytics helpers and preview chart exports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.config import FORECAST_PATH, METRICS_PATH, PROCESSED_DATA_PATH, SCREENSHOTS_DIR, SEGMENTS_PATH


def load_dashboard_bundle() -> dict:
    sales_df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["date"])
    segments_df = pd.read_csv(SEGMENTS_PATH)
    forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    metrics = json.loads(METRICS_PATH.read_text())

    segment_profiles = (
        segments_df.groupby("cluster")
        .agg(
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            customers=("customer_id", "count"),
        )
        .round(2)
        .reset_index()
    )

    return {
        "sales_df": sales_df,
        "segments_df": segments_df,
        "forecast_df": forecast_df,
        "segment_profiles": segment_profiles,
        "metrics": metrics,
    }


def create_revenue_trend_figure(sales_df: pd.DataFrame) -> go.Figure:
    monthly = (
        sales_df.assign(month=sales_df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["total_amount"]
        .sum()
    )
    fig = px.line(
        monthly,
        x="month",
        y="total_amount",
        markers=True,
        title="Monthly Revenue Trend",
        labels={"month": "Month", "total_amount": "Revenue"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_category_sales_figure(sales_df: pd.DataFrame) -> go.Figure:
    category_sales = (
        sales_df.groupby("product_category", as_index=False)["total_amount"].sum()
        .sort_values("total_amount", ascending=False)
    )
    fig = px.bar(
        category_sales,
        x="product_category",
        y="total_amount",
        color="product_category",
        title="Revenue by Product Category",
        labels={"product_category": "Category", "total_amount": "Revenue"},
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_region_sales_figure(sales_df: pd.DataFrame) -> go.Figure:
    region_sales = sales_df.groupby("region", as_index=False)["total_amount"].sum()
    fig = px.pie(
        region_sales,
        names="region",
        values="total_amount",
        title="Regional Revenue Contribution",
        hole=0.45,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_segment_distribution_figure(segments_df: pd.DataFrame) -> go.Figure:
    counts = segments_df["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "customers"]
    fig = px.bar(
        counts,
        x="cluster",
        y="customers",
        color="cluster",
        title="Customer Segment Distribution",
        labels={"cluster": "Cluster", "customers": "Customers"},
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_forecast_vs_actual_figure(forecast_df: pd.DataFrame) -> go.Figure:
    test_df = forecast_df.loc[forecast_df["split"] == "test"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test_df["date"],
            y=test_df["actual_revenue"],
            mode="lines+markers",
            name="Actual revenue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_df["date"],
            y=test_df["predicted_revenue"],
            mode="lines",
            name="Forecasted revenue",
        )
    )
    fig.update_layout(
        title="Forecasted vs Actual Daily Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def create_outlier_figure(sales_df: pd.DataFrame) -> go.Figure:
    flagged = sales_df.copy()
    flagged["outlier_status"] = flagged["is_outlier_total_amount"].map(
        {True: "Outlier", False: "Normal"}
    )
    sample = flagged.sample(min(800, len(flagged)), random_state=42)
    fig = px.scatter(
        sample,
        x="quantity",
        y="total_amount_original",
        color="outlier_status",
        hover_data=["product_category", "region", "customer_id"],
        title="Outlier Transactions",
        labels={"total_amount_original": "Original total amount"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def export_preview_images(bundle: dict) -> list[Path]:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    sales_df = bundle["sales_df"]
    segments_df = bundle["segments_df"]
    forecast_df = bundle["forecast_df"]

    monthly = (
        sales_df.assign(month=sales_df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["total_amount"]
        .sum()
    )
    category_sales = (
        sales_df.groupby("product_category", as_index=False)["total_amount"].sum()
        .sort_values("total_amount", ascending=False)
    )
    segment_counts = segments_df["cluster"].value_counts().sort_index()
    forecast_test = forecast_df.loc[forecast_df["split"] == "test"]

    created_files: list[Path] = []

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(monthly["month"], monthly["total_amount"], marker="o", linewidth=2.2)
    axis.set_title("Revenue Trend")
    axis.set_xlabel("Month")
    axis.set_ylabel("Revenue")
    axis.tick_params(axis="x", rotation=45)
    revenue_path = SCREENSHOTS_DIR / "revenue-trend.png"
    fig.tight_layout()
    fig.savefig(revenue_path, dpi=180)
    plt.close(fig)
    created_files.append(revenue_path)

    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=category_sales, x="product_category", y="total_amount", ax=axis, color="#4472c4")
    axis.set_title("Category-wise Sales")
    axis.set_xlabel("Category")
    axis.set_ylabel("Revenue")
    axis.tick_params(axis="x", rotation=25)
    category_path = SCREENSHOTS_DIR / "category-sales.png"
    fig.tight_layout()
    fig.savefig(category_path, dpi=180)
    plt.close(fig)
    created_files.append(category_path)

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(x=segment_counts.index.astype(str), y=segment_counts.values, ax=axis, color="#2ca58d")
    axis.set_title("Customer Segments")
    axis.set_xlabel("Cluster")
    axis.set_ylabel("Customers")
    segment_path = SCREENSHOTS_DIR / "customer-segments.png"
    fig.tight_layout()
    fig.savefig(segment_path, dpi=180)
    plt.close(fig)
    created_files.append(segment_path)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(forecast_test["date"], forecast_test["actual_revenue"], label="Actual", linewidth=2)
    axis.plot(forecast_test["date"], forecast_test["predicted_revenue"], label="Forecast", linewidth=2)
    axis.set_title("Forecasted vs Actual Revenue")
    axis.set_xlabel("Date")
    axis.set_ylabel("Revenue")
    axis.legend()
    forecast_path = SCREENSHOTS_DIR / "forecast-vs-actual.png"
    fig.tight_layout()
    fig.savefig(forecast_path, dpi=180)
    plt.close(fig)
    created_files.append(forecast_path)

    return created_files
