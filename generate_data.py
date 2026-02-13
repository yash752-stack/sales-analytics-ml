"""
Generate synthetic retail sales data for analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

np.random.seed(42)

def generate_sales_data(n_records=10000):
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    transaction_dates = np.random.choice(dates, size=n_records)
    
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Toys']
    category_weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    product_categories = np.random.choice(categories, size=n_records, p=category_weights)
    
    base_prices = {
        'Electronics': 300, 'Clothing': 50, 'Home & Kitchen': 80,
        'Books': 20, 'Sports': 60, 'Toys': 35
    }
    
    data = []
    customer_id_pool = np.arange(1, 2001)
    
    for i in range(n_records):
        date = pd.Timestamp(transaction_dates[i])
        category = product_categories[i]
        seasonal_factor = 1.3 if date.month in [11, 12] else 1.0
        weekend_factor = 1.15 if date.weekday() >= 5 else 1.0
        
        base_price = base_prices[category]
        price = base_price * np.random.uniform(0.8, 1.5) * seasonal_factor
        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])
        amount = price * quantity * weekend_factor
        
        customer_id = np.random.choice(customer_id_pool, p=np.random.dirichlet(np.ones(2000)))
        discount = np.random.choice([0, 5, 10, 15, 20], p=[0.4, 0.25, 0.2, 0.1, 0.05])
        final_amount = amount * (1 - discount/100)
        
        payment_method = np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'], 
                                         p=[0.4, 0.3, 0.15, 0.15])
        region = np.random.choice(['North', 'South', 'East', 'West'], p=[0.28, 0.25, 0.25, 0.22])
        
        data.append({
            'transaction_id': f'TXN{i+1:06d}',
            'date': date,
            'customer_id': f'CUST{customer_id:04d}',
            'product_category': category,
            'quantity': quantity,
            'unit_price': round(price, 2),
            'discount_percent': discount,
            'total_amount': round(final_amount, 2),
            'payment_method': payment_method,
            'region': region
        })
    
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'discount_percent'] = np.nan
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[outlier_indices, 'total_amount'] *= np.random.uniform(3, 5, size=len(outlier_indices))
    
    return df

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    print("Generating synthetic sales data...")
    sales_df = generate_sales_data(n_records=10000)
    sales_df.to_csv('data/raw/sales_data.csv', index=False)
    print(f"✅ Generated {len(sales_df)} sales records")
    print(f"💰 Total revenue: ${sales_df['total_amount'].sum():,.2f}")
    print("Data saved to: data/raw/sales_data.csv")
