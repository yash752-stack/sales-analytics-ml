"""CLI entrypoint for the sales analytics pipeline."""

from __future__ import annotations

import argparse
from textwrap import dedent

from src.config import DEFAULT_RECORDS
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end sales analytics and forecasting pipeline."
    )
    parser.add_argument(
        "--records",
        type=int,
        default=DEFAULT_RECORDS,
        help="Number of synthetic retail transactions to generate.",
    )
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Skip exporting preview charts to the screenshots directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_pipeline(
        n_records=args.records,
        export_previews=not args.skip_previews,
    )
    metrics = artifacts["metrics"]

    summary = dedent(
        f"""
        ==============================================================================
          SALES ANALYTICS & FORECASTING WITH MACHINE LEARNING
        ==============================================================================

        Synthetic records generated : {metrics["records"]}
        Unique customers            : {metrics["unique_customers"]}
        Revenue forecast model      : {metrics["forecasting_model"]}
        Forecast R²                 : {metrics["r2_score"]:.3f}
        K-Means clusters            : {metrics["optimal_clusters"]}
        Silhouette score            : {metrics["silhouette_score"]:.3f}
        Outlier rate                : {metrics["outlier_rate"]:.2%}

        Output files
          - Clean dataset      : {artifacts["processed_path"]}
          - Customer segments  : {artifacts["segments_path"]}
          - Forecast output    : {artifacts["forecast_path"]}
          - Metrics            : {artifacts["metrics_path"]}
        """
    ).strip()

    print(summary)


if __name__ == "__main__":
    main()
