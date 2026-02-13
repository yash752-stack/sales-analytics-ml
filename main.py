"""
Main Execution Script - Runs complete pipeline
"""
import os
import sys

print("="*70)
print("  SALES ANALYTICS & FORECASTING WITH MACHINE LEARNING")
print("="*70)

# Create directories
for directory in ['data/raw', 'data/processed', 'visualizations', 'models', 'src']:
    os.makedirs(directory, exist_ok=True)

# Add src to path
sys.path.insert(0, 'src')

# Step 1: Generate Data
print("\n" + "="*70)
print("STEP 1: DATA GENERATION")
print("="*70)
exec(open('generate_data.py').read())

# Step 2: Preprocessing
print("\n" + "="*70)
print("STEP 2: DATA PREPROCESSING")
print("="*70)
from data_preprocessing import preprocess_pipeline
preprocess_pipeline('data/raw/sales_data.csv', 'data/processed/sales_data_cleaned.csv')

# Step 3: Clustering
print("\n" + "="*70)
print("STEP 3: CUSTOMER SEGMENTATION")
print("="*70)
from clustering import run_segmentation_pipeline
run_segmentation_pipeline('data/processed/sales_data_cleaned.csv')

# Step 4: Forecasting
print("\n" + "="*70)
print("STEP 4: SALES FORECASTING")
print("="*70)
from forecasting import run_forecasting_pipeline
run_forecasting_pipeline('data/processed/sales_data_cleaned.csv')

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print("\n📁 Check these folders:")
print("   - data/processed/")
print("   - visualizations/")
print("   - models/")
print("\n🎉 Your project is ready to push to GitHub!")
