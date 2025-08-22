"""
Geometric Baseline Clustering Implementation.
Simple, interpretable clustering based on hand-crafted geometric features.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from core.schemas import Layout
from features.geometric import extract_all_features
from utils.evaluation import evaluate_clustering
from utils.logger import get_logger

logger = get_logger(__name__)


def run_baseline_clustering(layouts: List[Layout], min_clusters: int = 2, max_clusters: int = 8) -> Dict[str, Any]:
    """
    Run the baseline geometric feature clustering.

    Args:
        layouts: List of layout objects
        min_clusters: Minimum number of clusters to test (default: 2)
        max_clusters: Maximum number of clusters to test (default: 8)

    Returns:
        Dictionary with clustering results and metrics
    """
    logger.info("=" * 60)
    logger.info("GEOMETRIC BASELINE CLUSTERING")
    logger.info("=" * 60)

    # Extract features
    logger.info(f"Extracting geometric features for {len(layouts)} layouts...")
    feature_records = [extract_all_features(layout) for layout in layouts]
    feature_df = pd.DataFrame(feature_records)

    # Prepare features for clustering (exclude id)
    feature_columns = [col for col in feature_df.columns if col != "id"]
    features = feature_df[feature_columns].values

    # Handle any NaN values
    features = np.nan_to_num(features, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Try different cluster numbers and find the best
    best_score = -1
    best_k = 2
    best_labels = None
    quality_labels = [layout.quality for layout in layouts]

    logger.info("Testing different cluster numbers...")
    effective_max_clusters = min(max_clusters, len(layouts) // 5)
    logger.info(f"Testing k from {min_clusters} to {effective_max_clusters}")

    for k in range(min_clusters, effective_max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)

        # Evaluate clustering
        metrics = evaluate_clustering(scaled_features, cluster_labels, quality_labels)
        combined_score = metrics["combined_score"]

        logger.info(
            f"  k={k}: silhouette={metrics['silhouette_score']:.3f}, "
            f"purity={metrics['quality_purity']:.3f}, "
            f"combined={combined_score:.3f}"
        )

        if combined_score > best_score:
            best_score = combined_score
            best_k = k
            best_labels = cluster_labels

    # Final clustering with best parameters
    logger.info(f"Best configuration: k={best_k}")
    feature_df["cluster"] = best_labels

    # Analyze clusters
    logger.info("--- Geometric Baseline Clustering Results ---")
    cluster_summary = (
        feature_df.groupby("cluster")
        .agg({"id": "count", "balance_score": "mean", "edge_alignment_score": "mean", "content_density": "mean"})
        .round(3)
    )
    cluster_summary.columns = ["size", "avg_balance", "avg_alignment", "avg_density"]
    logger.info(f"\n{cluster_summary}")

    # Quality analysis per cluster
    logger.info("Quality distribution per cluster:")
    for cluster_id in sorted(feature_df["cluster"].unique()):
        cluster_layouts = [layouts[i] for i in range(len(layouts)) if best_labels[i] == cluster_id]
        pass_count = sum(1 for layout in cluster_layouts if layout.quality == "pass")
        total_count = len(cluster_layouts)
        pass_rate = pass_count / total_count if total_count > 0 else 0
        logger.info(f"  Cluster {cluster_id}: {pass_count}/{total_count} pass ({pass_rate:.1%})")

    # Final metrics
    final_metrics = evaluate_clustering(scaled_features, best_labels, quality_labels)

    logger.info("Final Metrics:")
    logger.info(f"  Silhouette Score: {final_metrics['silhouette_score']:.3f}")
    logger.info(f"  Quality Purity: {final_metrics['quality_purity']:.3f}")
    logger.info(f"  Combined Score: {final_metrics['combined_score']:.3f}")

    return {
        "feature_df": feature_df,
        "cluster_labels": best_labels.tolist(),
        "silhouette_score": final_metrics["silhouette_score"],
        "quality_purity": final_metrics["quality_purity"],
        "combined_score": final_metrics["combined_score"],
        "n_clusters": best_k,
        "scaler": scaler,
        "feature_names": feature_columns,
    }


def main() -> Dict[str, Any]:
    """Main function for standalone execution."""
    from pathlib import Path

    from core.schemas import load_layouts_from_json

    # Load data
    data_path = Path(__file__).resolve().parent.parent / "data" / "01_raw" / "assignment_data.json"

    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        logger.error("Please copy the assignment_data.json file to data/01_raw/")
        return {}

    layouts = load_layouts_from_json(str(data_path))

    # Run clustering
    results = run_baseline_clustering(layouts)

    logger.info("✅ Baseline clustering complete!")
    return results


if __name__ == "__main__":
    main()
