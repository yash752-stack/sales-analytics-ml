"""
Customer Segmentation using K-Means
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.scaler = StandardScaler()
        
    def calculate_rfm(self):
        print("\n" + "="*60)
        print("CALCULATING RFM")
        print("="*60)
        reference_date = self.df['date'].max() + pd.Timedelta(days=1)
        rfm = self.df.groupby('customer_id').agg({
            'date': lambda x: (reference_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        })
        rfm.columns = ['recency', 'frequency', 'monetary']
        print(f"\n✅ RFM calculated for {len(rfm)} customers")
        self.rfm_df = rfm
        return self
    
    def find_optimal_clusters(self, max_clusters=8):
        print("\n" + "="*60)
        print("FINDING OPTIMAL CLUSTERS")
        print("="*60)
        X = self.rfm_df[['recency', 'frequency', 'monetary']]
        X_scaled = self.scaler.fit_transform(X)
        
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n✅ Optimal k: {optimal_k} (Silhouette: {max(silhouette_scores):.3f})")
        return optimal_k
    
    def perform_clustering(self, n_clusters=4):
        print(f"\n🎯 K-Means clustering (k={n_clusters})")
        X = self.rfm_df[['recency', 'frequency', 'monetary']]
        X_scaled = self.scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df['cluster'] = kmeans.fit_predict(X_scaled)
        print(f"✅ Clustering complete!")
        self.clustered_df = self.rfm_df
        return self
    
    def analyze_segments(self):
        print("\n" + "="*60)
        print("SEGMENT ANALYSIS")
        print("="*60)
        profiles = self.clustered_df.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        profiles['count'] = self.clustered_df.groupby('cluster').size()
        print("\n" + profiles.to_string())
        return profiles
    
    def save_results(self, output_path):
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.clustered_df.to_csv(output_path)
        print(f"\n💾 Saved: {output_path}")

def run_segmentation_pipeline(input_path):
    print("\n🎯 CUSTOMER SEGMENTATION")
    df = pd.read_csv(input_path)
    seg = CustomerSegmentation(df)
    seg.calculate_rfm()
    optimal_k = seg.find_optimal_clusters()
    seg.perform_clustering(n_clusters=optimal_k)
    seg.analyze_segments()
    seg.save_results('data/processed/customer_segments.csv')
    print("\n✅ SEGMENTATION COMPLETE!")
    return seg
