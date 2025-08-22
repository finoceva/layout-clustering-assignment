"""
Structural clustering using LayoutLMv3 embeddings.
(Compatibility wrapper around flexible clustering system)

Usage Examples:

# Simple usage with defaults (LayoutLMv3 + PCA + K-means)
results = run_structural_clustering(layouts)

# Custom configuration
config = {
    "embedding_model": "layoutlmv1",
    "pooling_method": "mean",
    "reduction_method": "umap",
    "clustering_method": "hdbscan",
    "n_components": 15,
    "min_cluster_size": 3
}
results = run_structural_clustering(layouts, config)

# Minimal config override
config = {"n_clusters": 8}  # Only change number of clusters
results = run_structural_clustering(layouts, config)

# Override specific parameters
results = run_structural_clustering(layouts, {
    "embedding_model": "layoutlmv3",
    "n_components": 20,
    "n_clusters": 5
})

# Optimization examples
# Limited sampling (default)
opt_results = run_structural_clustering_optimization(layouts, max_combinations=50)

# Random sampling strategy
opt_results = run_structural_clustering_optimization(layouts, max_combinations=30)

# Run ALL configurations (may take hours!)
opt_results = run_structural_clustering_optimization(layouts, max_combinations=-1)
"""

from typing import Any, Dict, List

from clustering.flexible import FlexibleStructuralClusterer  # type: ignore
from core.schemas import Layout  # type: ignore
from utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)


def run_structural_clustering(
    layouts: List[Layout], config: Dict[str, Any] = None, output_path: str = None
) -> Dict[str, Any]:
    """
    Run structural clustering with configurable parameters.

    Args:
        layouts: List of layout objects
        config: Configuration dictionary with clustering parameters. If None, uses defaults.
                Expected keys:
                - embedding_model: "layoutlmv1" or "layoutlmv3" (default: "layoutlmv3")
                - pooling_method: "cls", "mean", "max" (default: "cls")
                - reduction_method: "pca" or "umap" (default: "pca")
                - clustering_method: "kmeans" or "hdbscan" (default: "kmeans")
                - n_components: Number of components for reduction (default: 20)
                - n_clusters: Number of clusters for kmeans (default: 5)
        output_path: Optional path to save results

    Returns:
        Dictionary with clustering results
    """
    # Load default configuration from YAML file
    from config.manager import ConfigurationManager

    config_manager = ConfigurationManager()
    yaml_defaults = config_manager.get_default_configuration()

    default_config = {
        "embedding_model": yaml_defaults.get("embedding_model", "layoutlmv3"),
        "pooling_method": yaml_defaults.get("pooling_method", "cls"),
        "reduction_method": yaml_defaults.get("dimensionality_reduction", "pca"),
        "clustering_method": yaml_defaults.get("clustering_algorithm", "kmeans"),
        "n_components": 10,  # OPTIMAL: 10 components
        "n_clusters": 2,  # OPTIMAL: 2 clusters
    }

    # Merge with provided config
    if config is None:
        config = default_config
    else:
        merged_config = default_config.copy()
        merged_config.update(config)
        config = merged_config

    logger.info("=" * 60)
    logger.info("STRUCTURAL CLUSTERING")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")

    # Use the flexible clusterer
    clusterer = FlexibleStructuralClusterer()

    # Build config in the format expected by run_single_configuration
    flexible_config = {
        "embedding_model": config["embedding_model"],
        "pooling_method": config["pooling_method"],
        "dimensionality_reduction": {"method": config["reduction_method"], "n_components": config["n_components"]},
        "clustering_algorithm": {"method": config["clustering_method"]},
    }

    # Add clustering-specific parameters
    if config["clustering_method"] == "kmeans":
        flexible_config["clustering_algorithm"]["n_clusters"] = config["n_clusters"]
    elif config["clustering_method"] == "hdbscan":
        flexible_config["clustering_algorithm"]["min_cluster_size"] = config.get("min_cluster_size", 5)

    # Run with configuration
    result = clusterer.run_single_configuration(layouts, flexible_config)

    # Extract results in the expected format
    if "error" in result:
        logger.error(f"Clustering failed: {result['error']}")
        return {"error": result["error"]}

    cluster_labels = result["cluster_labels"]
    metrics = result["metrics"]

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    logger.info(f"Quality Purity: {metrics['quality_purity']:.3f}")
    logger.info(f"Number of Clusters: {metrics['n_clusters']}")
    logger.info(f"Balance Score: {metrics['balance_score']:.3f}")

    # Cluster analysis
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise cluster

    logger.info("\n📊 Cluster Analysis:")
    for cluster_id in sorted(unique_labels):
        cluster_layouts = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_qualities = [layouts[i].quality for i in cluster_layouts]
        pass_count = sum(1 for q in cluster_qualities if q == "pass")
        fail_count = len(cluster_qualities) - pass_count

        logger.info(
            f"  Cluster {cluster_id}: {len(cluster_layouts)} layouts " f"({pass_count} pass, {fail_count} fail)"
        )

    # Save results if requested
    if output_path:
        clusterer.all_results = [result]
        clusterer.best_result = result
        clusterer.save_results(output_path)
        logger.info(f"Results saved to: {output_path}")

    # Return in original format for compatibility (include layout_ids for recommendation engine)
    layout_ids = [layout.id for layout in layouts]

    return {
        "cluster_labels": cluster_labels,
        "layout_ids": layout_ids,  # Required by recommendation engine
        "silhouette_score": metrics["silhouette_score"],
        "quality_purity": metrics["quality_purity"],
        "balance_score": metrics["balance_score"],
        "n_clusters": metrics["n_clusters"],
        "embeddings_shape": result["embeddings_shape"],
        "reduced_shape": result["reduced_shape"],
        "best_config": result["config"],
    }


def run_structural_clustering_optimization(
    layouts: List[Layout],
    embedding_models: List[str] = None,
    max_combinations: int | None = 20,
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Run structural clustering optimization across multiple configurations.

    Args:
        layouts: List of layout objects
        embedding_models: List of embedding models to test
        max_combinations: Maximum configurations to test. Use -1 to test ALL configurations.
        output_path: Optional path to save results

    Returns:
        Optimization results with best configuration
    """
    logger.info("=" * 60)
    logger.info("STRUCTURAL CLUSTERING OPTIMIZATION")
    logger.info("=" * 60)

    # Use the flexible clusterer for optimization
    clusterer = FlexibleStructuralClusterer()

    results = clusterer.run_optimization(
        layouts=layouts, embedding_models=embedding_models, max_combinations=max_combinations
    )

    # Save results if requested
    if output_path and "best_result" in results:
        clusterer.save_results(output_path)
        logger.info(f"Results saved to: {output_path}")

    return results
