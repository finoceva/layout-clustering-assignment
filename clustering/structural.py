"""
Structural Clustering Implementation using LayoutLMv3.
Track 1: Find layouts that are structurally and visually similar.
"""

from typing import Any, Dict, List

import numpy as np
from embeddings.layoutlmv3 import LayoutLMv3Embedder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.evaluation import evaluate_clustering
from utils.logger import get_logger

from core.schemas import Layout

logger = get_logger(__name__)


def run_structural_clustering(layouts: List[Layout]) -> Dict[str, Any]:
    """Run structural clustering using LayoutLMv3."""
    logger.info("="*60)
    logger.info("STRUCTURAL CLUSTERING (LayoutLMv3)")
    logger.info("="*60)
    
    # Extract embeddings
    logger.info(f"Extracting LayoutLMv3 embeddings for {len(layouts)} layouts...")
    embedder = LayoutLMv3Embedder()
    embeddings = embedder.extract_embeddings_batch(layouts)
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA for dimensionality reduction
    pca_components = min(50, len(layouts) - 1)  # Ensure valid PCA components
    pca = PCA(n_components=pca_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    logger.info(f"PCA reduction: {embeddings.shape[1]} -> {embeddings_pca.shape[1]} dimensions")
    logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Try different cluster numbers and find the best
    best_score = -1
    best_k = 2
    best_labels = None
    quality_labels = [layout.quality for layout in layouts]
    
    logger.info("Testing different cluster numbers...")
    max_k = min(8, len(layouts) // 5)
    
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_pca)
        
        # Evaluate clustering
        metrics = evaluate_clustering(embeddings_pca, cluster_labels, quality_labels)
        combined_score = metrics['combined_score']
        
        logger.info(f"  k={k}: silhouette={metrics['silhouette_score']:.3f}, "
                   f"purity={metrics['quality_purity']:.3f}, "
                   f"combined={combined_score:.3f}")
        
        if combined_score > best_score:
            best_score = combined_score
            best_k = k
            best_labels = cluster_labels
    
    logger.info(f"Best configuration: k={best_k}")
    
    # Analyze clusters
    logger.info("--- Structural Clustering (LayoutLMv3) Results ---")
    unique_clusters = np.unique(best_labels)
    logger.info(f"Discovered clusters: {unique_clusters}")
    
    for cluster_id in unique_clusters:
        cluster_layouts = [layouts[i] for i in range(len(layouts)) if best_labels[i] == cluster_id]
        pass_count = sum(1 for layout in cluster_layouts if layout.quality == 'pass')
        total_count = len(cluster_layouts)
        pass_rate = pass_count / total_count if total_count > 0 else 0
        logger.info(f"  Cluster {cluster_id}: {total_count} layouts, "
                   f"{pass_count}/{total_count} pass ({pass_rate:.1%})")
    
    # Final metrics
    final_metrics = evaluate_clustering(embeddings_pca, best_labels, quality_labels)
    
    logger.info("Final Metrics:")
    logger.info(f"  Silhouette Score: {final_metrics['silhouette_score']:.3f}")
    logger.info(f"  Quality Purity: {final_metrics['quality_purity']:.3f}")
    logger.info(f"  Combined Score: {final_metrics['combined_score']:.3f}")
    
    return {
        "layout_ids": [layout.id for layout in layouts],
        "cluster_labels": best_labels.tolist(),
        "silhouette_score": final_metrics['silhouette_score'],
        "quality_purity": final_metrics['quality_purity'],
        "combined_score": final_metrics['combined_score'],
        "n_clusters": best_k,
        "pca_components": pca_components,
        "explained_variance": pca.explained_variance_ratio_.sum()
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
    results = run_structural_clustering(layouts)
    
    logger.info("✅ Structural clustering complete!")
    return results


if __name__ == "__main__":
    main()
