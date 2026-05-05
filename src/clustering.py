"""Customer segmentation using RFM features and K-Means clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


class CustomerSegmentation:
    """Fit an RFM-style segmentation model for retail customers."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.scaler = StandardScaler()
        self.silhouette = 0.0
        self.optimal_clusters = 0

    def calculate_rfm(self) -> "CustomerSegmentation":
        reference_date = self.df["date"].max() + pd.Timedelta(days=1)
        rfm = self.df.groupby("customer_id").agg(
            recency=("date", lambda values: (reference_date - values.max()).days),
            frequency=("transaction_id", "count"),
            monetary=("total_amount", "sum"),
        )
        self.rfm_df = rfm
        return self

    def find_optimal_clusters(self, max_clusters: int = 8) -> int:
        features = self.rfm_df[["recency", "frequency", "monetary"]]
        scaled_features = self.scaler.fit_transform(features)
        candidate_range = range(2, max_clusters + 1)

        silhouette_scores = []
        for cluster_count in candidate_range:
            estimator = KMeans(
                n_clusters=cluster_count,
                random_state=RANDOM_STATE,
                n_init=10,
            )
            labels = estimator.fit_predict(scaled_features)
            silhouette_scores.append(silhouette_score(scaled_features, labels))

        best_index = int(np.argmax(silhouette_scores))
        self.optimal_clusters = list(candidate_range)[best_index]
        self.silhouette = float(max(silhouette_scores))
        return self.optimal_clusters

    def perform_clustering(self, n_clusters: int | None = None) -> "CustomerSegmentation":
        cluster_count = n_clusters or self.optimal_clusters or 4
        features = self.rfm_df[["recency", "frequency", "monetary"]]
        scaled_features = self.scaler.fit_transform(features)
        estimator = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        self.rfm_df["cluster"] = estimator.fit_predict(scaled_features)
        self.clustered_df = self.rfm_df.reset_index()
        return self

    def analyze_segments(self) -> pd.DataFrame:
        profiles = (
            self.clustered_df.groupby("cluster")
            .agg(
                recency=("recency", "mean"),
                frequency=("frequency", "mean"),
                monetary=("monetary", "mean"),
                customers=("customer_id", "count"),
            )
            .round(2)
            .reset_index()
        )
        self.segment_profiles = profiles
        return profiles

    def save_results(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.clustered_df.to_csv(output_path, index=False)
        return output_path


def run_segmentation_pipeline(input_path: str | Path, output_path: str | Path) -> dict:
    dataframe = pd.read_csv(input_path)
    segmenter = CustomerSegmentation(dataframe)
    segmenter.calculate_rfm()
    optimal_clusters = segmenter.find_optimal_clusters()
    segmenter.perform_clustering(n_clusters=optimal_clusters)
    profiles = segmenter.analyze_segments()
    saved_path = segmenter.save_results(output_path)

    return {
        "segmenter": segmenter,
        "segments_df": segmenter.clustered_df,
        "profiles": profiles,
        "output_path": saved_path,
        "metrics": {
            "optimal_clusters": optimal_clusters,
            "silhouette_score": segmenter.silhouette,
        },
    }
