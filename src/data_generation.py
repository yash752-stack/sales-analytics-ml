"""Synthetic retail data generation utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


def generate_sales_data(n_records: int = 10_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="h")
    date_weights = np.ones(len(dates), dtype=float)
    date_weights *= np.where(dates.month.isin([11, 12]), 1.55, 1.0)
    date_weights *= np.where(dates.month.isin([6, 7]), 1.10, 1.0)
    date_weights *= np.where(dates.dayofweek >= 5, 1.25, 1.0)
    date_weights *= np.where(dates.year == 2024, 1.08, 1.0)
    date_weights *= np.where(dates.hour.isin([11, 12, 18, 19, 20]), 1.15, 1.0)
    date_weights /= date_weights.sum()
    transaction_dates = rng.choice(dates, size=n_records, p=date_weights)

    categories = [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Books",
        "Sports",
        "Toys",
    ]
    category_weights = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
    customer_pool = np.arange(1, 2001)
    customer_weights = rng.dirichlet(np.ones(len(customer_pool)) * 0.8)

    base_prices = {
        "Electronics": 300,
        "Clothing": 50,
        "Home & Kitchen": 80,
        "Books": 20,
        "Sports": 60,
        "Toys": 35,
    }

    rows: list[dict] = []
    for index in range(n_records):
        date = pd.Timestamp(transaction_dates[index])
        category = rng.choice(categories, p=category_weights)
        seasonal_factor = 1.35 if date.month in (11, 12) else 1.05 if date.month in (6, 7) else 1.0
        weekend_factor = 1.15 if date.weekday() >= 5 else 1.0
        yearly_growth_factor = 1.08 if date.year == 2024 else 1.0

        base_price = base_prices[category]
        unit_price = base_price * rng.uniform(0.92, 1.18) * seasonal_factor * yearly_growth_factor
        quantity = rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.24, 0.12, 0.06, 0.03])
        amount = unit_price * quantity * weekend_factor

        customer_id = rng.choice(customer_pool, p=customer_weights)
        discount = rng.choice([0, 5, 10, 15, 20], p=[0.4, 0.25, 0.2, 0.1, 0.05])
        final_amount = amount * (1 - discount / 100)
        payment_method = rng.choice(
            ["Credit Card", "Debit Card", "Cash", "Digital Wallet"],
            p=[0.4, 0.3, 0.15, 0.15],
        )
        region = rng.choice(["North", "South", "East", "West"], p=[0.28, 0.25, 0.25, 0.22])

        rows.append(
            {
                "transaction_id": f"TXN{index + 1:06d}",
                "date": date,
                "customer_id": f"CUST{customer_id:04d}",
                "product_category": category,
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "discount_percent": discount,
                "total_amount": round(final_amount, 2),
                "payment_method": payment_method,
                "region": region,
            }
        )

    dataframe = pd.DataFrame(rows)
    missing_indices = rng.choice(dataframe.index, size=int(0.02 * len(dataframe)), replace=False)
    outlier_indices = rng.choice(dataframe.index, size=int(0.01 * len(dataframe)), replace=False)

    dataframe.loc[missing_indices, "discount_percent"] = np.nan
    dataframe.loc[outlier_indices, "total_amount"] *= rng.uniform(2.5, 4.0, size=len(outlier_indices))

    return dataframe


def save_sales_data(output_path: Path, n_records: int = 10_000) -> pd.DataFrame:
    dataframe = generate_sales_data(n_records=n_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe
