"""
Flexible structural clustering with configurable embedding and clustering methods.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from clustering.components import (  # type: ignore
    create_clustering_algorithm,
    create_dimensionality_reducer,
    evaluate_clustering,
)
from config.manager import ConfigurationManager  # type: ignore
from core.schemas import Layout  # type: ignore
from embeddings.factory import create_embedding_extractor  # type: ignore
from utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)


class FlexibleStructuralClusterer:
    """Flexible structural clustering with configurable methods."""

    def __init__(self, config_path: str = None):
        """
        Initialize flexible clusterer.

        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigurationManager(config_path)
        self.layouts: Optional[List[Layout]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.best_result: Optional[Dict[str, Any]] = None
        self.all_results: List[Dict[str, Any]] = []

    def run_single_configuration(self, layouts: List[Layout], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run clustering with a single configuration.

        Args:
            layouts: List of layout objects
            config: Configuration dictionary

        Returns:
            Results dictionary with metrics and configuration
        """
        logger.info(
            f"Testing configuration: {config['embedding_model']} + "
            f"{config['dimensionality_reduction']['method']} + "
            f"{config['clustering_algorithm']['method']}"
        )

        try:
            # Extract embeddings
            extractor = create_embedding_extractor(
                model_type=config["embedding_model"], pooling_method=config["pooling_method"]
            )

            logger.info("Extracting embeddings...")
            embeddings = extractor.extract_embeddings_batch(layouts)
            logger.info(f"Extracted embeddings shape: {embeddings.shape}")

            # Apply dimensionality reduction
            reduction_config = config["dimensionality_reduction"]
            reducer = create_dimensionality_reducer(
                method=reduction_config["method"], **{k: v for k, v in reduction_config.items() if k != "method"}
            )

            logger.info("Applying dimensionality reduction...")
            embeddings_reduced = reducer.fit_transform(embeddings)
            logger.info(f"Reduced embeddings shape: {embeddings_reduced.shape}")

            # Apply clustering
            clustering_config = config["clustering_algorithm"]
            clusterer = create_clustering_algorithm(
                method=clustering_config["method"], **{k: v for k, v in clustering_config.items() if k != "method"}
            )

            logger.info("Performing clustering...")
            cluster_labels = clusterer.fit_predict(embeddings_reduced)

            # Evaluate results
            quality_labels = [layout.quality for layout in layouts]
            metrics = evaluate_clustering(embeddings_reduced, cluster_labels, quality_labels)

            # Prepare result
            result = {
                "config": config,
                "metrics": metrics,
                "cluster_labels": cluster_labels.tolist(),
                "embeddings_shape": embeddings.shape,
                "reduced_shape": embeddings_reduced.shape,
                "reduction_params": reducer.get_params(),
                "clustering_params": clusterer.get_params(),
            }

            logger.info(
                f"Results - Silhouette: {metrics['silhouette_score']:.3f}, "
                f"Quality Purity: {metrics['quality_purity']:.3f}, "
                f"N Clusters: {metrics['n_clusters']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in configuration: {e}")
            return {
                "config": config,
                "error": str(e),
                "metrics": {"silhouette_score": -1.0, "quality_purity": 0.0, "combined_score": -1.0},
            }

    def run_optimization(
        self,
        layouts: List[Layout],
        embedding_models: List[str] = None,
        max_combinations: int | None = None,
        sampling_strategy: str = "random",
    ) -> Dict[str, Any]:
        """
        Run optimization across multiple configurations with smart sampling.

        Args:
            layouts: List of layout objects
            embedding_models: List of embedding models to test
            max_combinations: Maximum configurations to test. Use -1 to test ALL configurations.
            sampling_strategy: How to select combinations when max_combinations is limited:
                - "random": Random sampling across all combinations
                - "stratified": Ensure sampling across different method types
                - "first": Original behavior (first N combinations)

        Returns:
            Best configuration and all results
        """
        self.layouts = layouts
        self.all_results = []

        logger.info("=" * 60)
        logger.info("FLEXIBLE STRUCTURAL CLUSTERING - OPTIMIZATION")
        logger.info("=" * 60)

        # Generate configurations
        optimization_settings = self.config_manager.get_optimization_settings()
        primary_metric = optimization_settings.get("primary_metric", "silhouette_score")

        if max_combinations is None:
            max_combinations = optimization_settings.get("max_combinations", 20)

        # Generate ALL possible configurations first (without limit)
        all_configurations = list(
            self.config_manager.generate_configurations(
                embedding_models=embedding_models,
                max_combinations=None,  # Generate all combinations
            )
        )

        # Determine final configurations to test
        if max_combinations == -1:
            configurations = all_configurations
            logger.info(f"🚀 Running ALL {len(configurations)} configurations (max_combinations=-1)")
            logger.info(f"⏱️  Estimated time: ~{len(configurations)*0.5:.1f} minutes")
        else:
            configurations = self._select_configurations(all_configurations, max_combinations, sampling_strategy)
            logger.info(f"🎯 Testing {len(configurations)} out of {len(all_configurations)} total configurations")
            logger.info(f"📊 Sampling strategy: {sampling_strategy}")
            logger.info(f"⚡ Coverage: {(len(configurations)/len(all_configurations))*100:.1f}%")

        logger.info(f"📈 Primary optimization metric: {primary_metric}")

        # Test each configuration
        for i, config in enumerate(configurations, 1):
            logger.info(f"\n--- Configuration {i}/{len(configurations)} ---")
            result = self.run_single_configuration(layouts, config)
            self.all_results.append(result)

        # Find best configuration
        valid_results = [r for r in self.all_results if "error" not in r]

        if not valid_results:
            logger.error("No valid results found!")
            return {"error": "All configurations failed"}

        # Sort by primary metric
        valid_results.sort(key=lambda x: x["metrics"].get(primary_metric, -1), reverse=True)

        self.best_result = valid_results[0]

        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)

        best_config = self.best_result["config"]
        best_metrics = self.best_result["metrics"]

        logger.info("🏆 Best Configuration:")
        logger.info(f"  Embedding: {best_config['embedding_model']} ({best_config['pooling_method']})")
        logger.info(f"  Reduction: {best_config['dimensionality_reduction']['method']}")
        logger.info(f"  Clustering: {best_config['clustering_algorithm']['method']}")

        logger.info("\n📊 Best Metrics:")
        logger.info(f"  Silhouette Score: {best_metrics['silhouette_score']:.3f}")
        logger.info(f"  Quality Purity: {best_metrics['quality_purity']:.3f}")
        logger.info(f"  Balance Score: {best_metrics['balance_score']:.3f}")
        logger.info(f"  Combined Score: {best_metrics['combined_score']:.3f}")
        logger.info(f"  Number of Clusters: {best_metrics['n_clusters']}")

        return {
            "best_result": self.best_result,
            "all_results": self.all_results,
            "total_configurations": len(configurations),
            "valid_configurations": len(valid_results),
        }

    def run_single_method(
        self,
        layouts: List[Layout],
        embedding_model: str = "layoutlmv3",
        pooling_method: str = "cls",
        reduction_method: str = "pca",
        clustering_method: str = "kmeans",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run clustering with a single specific method combination.

        Args:
            layouts: List of layout objects
            embedding_model: Embedding model to use
            pooling_method: Pooling method
            reduction_method: Dimensionality reduction method
            clustering_method: Clustering algorithm
            **kwargs: Additional parameters for methods

        Returns:
            Results dictionary
        """
        # Build configuration
        config = {
            "embedding_model": embedding_model,
            "pooling_method": pooling_method,
            "dimensionality_reduction": {
                "method": reduction_method,
                **{k: v for k, v in kwargs.items() if k.startswith(f"{reduction_method}_")},
            },
            "clustering_algorithm": {
                "method": clustering_method,
                **{k: v for k, v in kwargs.items() if k.startswith(f"{clustering_method}_")},
            },
        }

        return self.run_single_configuration(layouts, config)

    def _select_configurations(
        self, all_configs: List[Dict[str, Any]], max_combinations: int, strategy: str
    ) -> List[Dict[str, Any]]:
        """
        Select configurations using specified sampling strategy.

        Args:
            all_configs: List of all possible configurations
            max_combinations: Maximum number to select
            strategy: Sampling strategy ("random", "stratified", "first")

        Returns:
            Selected configurations
        """
        import random

        if len(all_configs) <= max_combinations:
            return all_configs

        if strategy == "first":
            # Original behavior - just take first N
            return all_configs[:max_combinations]

        elif strategy == "random":
            # Random sampling across all configurations
            return random.sample(all_configs, max_combinations)

        elif strategy == "stratified":
            # Stratified sampling - ensure representation across method types
            return self._stratified_sample(all_configs, max_combinations)

        else:
            logger.warning(f"Unknown sampling strategy: {strategy}. Using 'random'")
            return random.sample(all_configs, max_combinations)

    def _stratified_sample(self, all_configs: List[Dict[str, Any]], max_combinations: int) -> List[Dict[str, Any]]:
        """
        Perform stratified sampling to ensure representation across method types.

        Args:
            all_configs: All possible configurations
            max_combinations: Target number of configurations

        Returns:
            Stratified sample of configurations
        """
        import random
        from collections import defaultdict

        # Group configurations by method combinations
        strata = defaultdict(list)
        for config in all_configs:
            # Create stratification key based on methods used
            key = (
                config["embedding_model"],
                config["dimensionality_reduction"]["method"],
                config["clustering_algorithm"]["method"],
            )
            strata[key].append(config)

        # Calculate samples per stratum
        num_strata = len(strata)
        base_samples = max_combinations // num_strata
        extra_samples = max_combinations % num_strata

        selected = []
        strata_keys = list(strata.keys())
        random.shuffle(strata_keys)  # Randomize which strata get extra samples

        for i, key in enumerate(strata_keys):
            stratum_configs = strata[key]
            # Some strata get one extra sample
            samples_for_stratum = base_samples + (1 if i < extra_samples else 0)
            samples_for_stratum = min(samples_for_stratum, len(stratum_configs))

            if samples_for_stratum > 0:
                selected.extend(random.sample(stratum_configs, samples_for_stratum))

        # If we still need more samples (due to small strata), fill randomly
        if len(selected) < max_combinations:
            remaining = [c for c in all_configs if c not in selected]
            needed = max_combinations - len(selected)
            if remaining and needed > 0:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))

        return selected[:max_combinations]

    def save_results(self, output_path: str) -> None:
        """
        Save all results to file.

        Args:
            output_path: Path to save results
        """
        if not self.all_results:
            logger.warning("No results to save")
            return

        save_data = {
            "best_result": self.best_result,
            "all_results": self.all_results,
            "total_configurations": len(self.all_results),
        }

        self.config_manager.save_results([save_data], output_path)


def run_flexible_clustering(
    layouts: List[Layout],
    config_path: str = None,
    embedding_models: List[str] = None,
    max_combinations: int = None,
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Convenience function to run flexible clustering optimization.

    Args:
        layouts: List of layout objects
        config_path: Path to configuration file
        embedding_models: List of embedding models to test
        max_combinations: Maximum configurations to test
        output_path: Path to save results

    Returns:
        Optimization results
    """
    clusterer = FlexibleStructuralClusterer(config_path)
    results = clusterer.run_optimization(
        layouts=layouts, embedding_models=embedding_models, max_combinations=max_combinations
    )

    if output_path:
        clusterer.save_results(output_path)

    return results
