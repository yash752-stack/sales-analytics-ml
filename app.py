"""Streamlit dashboard for the sales analytics project."""

from __future__ import annotations

import streamlit as st

from src.analytics import (
    create_category_sales_figure,
    create_forecast_vs_actual_figure,
    create_outlier_figure,
    create_region_sales_figure,
    create_revenue_trend_figure,
    create_segment_distribution_figure,
    load_dashboard_bundle,
)
from src.config import DEFAULT_RECORDS
from src.pipeline import outputs_exist, run_pipeline


st.set_page_config(
    page_title="Sales Analytics ML",
    page_icon="📈",
    layout="wide",
)


def ensure_demo_data(record_count: int) -> None:
    if outputs_exist():
        return
    with st.spinner("Generating the demo dataset and ML artifacts..."):
        run_pipeline(n_records=record_count, export_previews=True)


def render_metric_row(bundle: dict) -> None:
    metrics = bundle["metrics"]
    sales_df = bundle["sales_df"]
    forecast_df = bundle["forecast_df"]

    total_revenue = sales_df["total_amount"].sum()
    avg_order_value = sales_df["total_amount"].mean()
    forecast_delta = (
        forecast_df["predicted_revenue"] - forecast_df["actual_revenue"]
    ).abs().mean()

    card_1, card_2, card_3, card_4 = st.columns(4)
    card_1.metric("Transactions", f"{metrics['records']:,}")
    card_2.metric("Revenue", f"${total_revenue:,.0f}")
    card_3.metric("Forecast R²", f"{metrics['r2_score']:.2f}")
    card_4.metric("Avg abs forecast error", f"${forecast_delta:,.0f}")

    card_5, card_6, card_7, card_8 = st.columns(4)
    card_5.metric("Customers", f"{metrics['unique_customers']:,}")
    card_6.metric("AOV", f"${avg_order_value:,.2f}")
    card_7.metric("Clusters", f"{metrics['optimal_clusters']}")
    card_8.metric("Silhouette", f"{metrics['silhouette_score']:.2f}")


def main() -> None:
    st.sidebar.title("Sales Analytics & Forecasting")
    record_count = st.sidebar.slider(
        "Synthetic dataset size",
        min_value=2_000,
        max_value=25_000,
        value=DEFAULT_RECORDS,
        step=1_000,
    )

    if st.sidebar.button("Regenerate demo data", width="stretch"):
        with st.spinner("Rebuilding the retail analytics pipeline..."):
            run_pipeline(n_records=record_count, export_previews=True)
        st.success("Pipeline refreshed with a new synthetic dataset.")

    ensure_demo_data(record_count=record_count)
    bundle = load_dashboard_bundle()
    metrics = bundle["metrics"]

    st.title("Sales Analytics & Forecasting with ML")
    st.caption(
        "Recruiter-ready retail analytics demo using a synthetic transaction dataset, "
        "K-Means segmentation, Random Forest forecasting, and interactive business dashboards."
    )

    render_metric_row(bundle)

    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Project snapshot")
        st.write(
            "This dashboard showcases an end-to-end retail analytics workflow: data generation, "
            "cleaning, customer segmentation, outlier handling, and revenue forecasting."
        )
        st.write(
            f"Latest pipeline run: `{metrics['generated_at']}` | "
            f"Outlier rate: `{metrics['outlier_rate']:.2%}` | "
            f"Model: `{metrics['forecasting_model']}`"
        )
    with right:
        st.subheader("Business value")
        st.markdown(
            "- Identify high-value customer cohorts\n"
            "- Forecast daily revenue from transactional behavior\n"
            "- Surface category, region, and anomaly trends\n"
            "- Export reproducible metrics for portfolio and resume claims"
        )

    tab_overview, tab_segments, tab_forecast, tab_outliers, tab_data = st.tabs(
        [
            "Overview",
            "Customer Segments",
            "Forecasting",
            "Outliers",
            "Data Preview",
        ]
    )

    with tab_overview:
        row_one_left, row_one_right = st.columns(2)
        row_one_left.plotly_chart(
            create_revenue_trend_figure(bundle["sales_df"]),
            width="stretch",
            key="overview_revenue_trend",
        )
        row_one_right.plotly_chart(
            create_category_sales_figure(bundle["sales_df"]),
            width="stretch",
            key="overview_category_sales",
        )

        row_two_left, row_two_right = st.columns(2)
        row_two_left.plotly_chart(
            create_region_sales_figure(bundle["sales_df"]),
            width="stretch",
            key="overview_region_sales",
        )
        row_two_right.plotly_chart(
            create_segment_distribution_figure(bundle["segments_df"]),
            width="stretch",
            key="overview_segment_distribution",
        )

    with tab_segments:
        st.subheader("RFM-style customer segmentation")
        st.dataframe(bundle["segment_profiles"], width="stretch")
        st.plotly_chart(
            create_segment_distribution_figure(bundle["segments_df"]),
            width="stretch",
            key="segments_distribution_detail",
        )

    with tab_forecast:
        st.subheader("Random Forest daily revenue forecasting")
        st.plotly_chart(
            create_forecast_vs_actual_figure(bundle["forecast_df"]),
            width="stretch",
            key="forecast_vs_actual_detail",
        )
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("R²", f"{metrics['r2_score']:.3f}")
        metric_b.metric("MAE", f"${metrics['mae']:,.0f}")
        metric_c.metric("RMSE", f"${metrics['rmse']:,.0f}")

    with tab_outliers:
        st.subheader("Outlier transactions")
        st.plotly_chart(
            create_outlier_figure(bundle["sales_df"]),
            width="stretch",
            key="outlier_scatter_detail",
        )
        outlier_rows = bundle["sales_df"].loc[bundle["sales_df"]["is_outlier_total_amount"]]
        st.dataframe(
            outlier_rows[
                [
                    "transaction_id",
                    "date",
                    "customer_id",
                    "product_category",
                    "region",
                    "total_amount_original",
                    "total_amount",
                ]
            ].head(100),
            width="stretch",
        )

    with tab_data:
        st.subheader("Processed transaction data")
        st.dataframe(bundle["sales_df"].head(250), width="stretch")


if __name__ == "__main__":
    main()
