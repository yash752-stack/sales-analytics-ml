"""Convenience script for generating the synthetic retail dataset only."""

from __future__ import annotations

from src.config import DEFAULT_RECORDS, RAW_DATA_PATH, ensure_project_dirs
from src.data_generation import save_sales_data


def main() -> None:
    ensure_project_dirs()
    dataframe = save_sales_data(RAW_DATA_PATH, n_records=DEFAULT_RECORDS)
    print(f"Generated {len(dataframe)} synthetic transactions at {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
