"""
Data Preprocessing Module
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self):
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing values found:")
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]}")
            if 'discount_percent' in self.df.columns:
                median_discount = self.df['discount_percent'].median()
                self.df['discount_percent'].fillna(median_discount, inplace=True)
                print(f"\n✅ Imputed discount_percent with median: {median_discount}")
        else:
            print("\n✅ No missing values")
        return self
    
    def detect_outliers(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        return outliers, lower_bound, upper_bound
        
    def handle_outliers(self, columns=['total_amount']):
        print("\n" + "="*60)
        print("HANDLING OUTLIERS")
        print("="*60)
        for col in columns:
            outliers, lower, upper = self.detect_outliers(col)
            n_outliers = outliers.sum()
            if n_outliers > 0:
                print(f"\n{col}: {n_outliers} outliers")
                self.df.loc[self.df[col] < lower, col] = lower
                self.df.loc[self.df[col] > upper, col] = upper
                print(f"  ✅ Capped to [{lower:.2f}, {upper:.2f}]")
        return self
    
    def feature_engineering(self):
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['revenue_per_unit'] = self.df['total_amount'] / self.df['quantity']
        
        print("\n✅ Created temporal features")
        return self
    
    def get_processed_data(self):
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Original: {self.original_shape}")
        print(f"Processed: {self.df.shape}")
        return self.df

def preprocess_pipeline(input_path, output_path):
    import os
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} records")
    
    preprocessor = DataPreprocessor(df)
    processed_df = (preprocessor
                   .handle_missing_values()
                   .handle_outliers()
                   .feature_engineering()
                   .get_processed_data())
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    return processed_df
