"""Lightweight repository contract checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_metrics_artifact_contains_expected_keys(self) -> None:
        metrics = json.loads((ROOT / "metrics.json").read_text())
        expected_keys = {
            "dataset",
            "records",
            "unique_customers",
            "forecasting_model",
            "r2_score",
            "segmentation_model",
            "optimal_clusters",
            "silhouette_score",
            "generated_at",
        }
        self.assertTrue(expected_keys.issubset(metrics.keys()))

    def test_preview_images_exist(self) -> None:
        preview_paths = {
            ROOT / "screenshots" / "revenue-trend.png",
            ROOT / "screenshots" / "category-sales.png",
            ROOT / "screenshots" / "customer-segments.png",
            ROOT / "screenshots" / "forecast-vs-actual.png",
        }
        self.assertTrue(all(path.exists() for path in preview_paths))

    def test_readme_mentions_streamlit_dashboard(self) -> None:
        readme_text = (ROOT / "README.md").read_text().lower()
        self.assertIn("streamlit", readme_text)
        self.assertIn("random forest", readme_text)


if __name__ == "__main__":
    unittest.main()
