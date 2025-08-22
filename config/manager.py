"""
Configuration manager for clustering pipeline.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, Generator, List, cast

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationManager:
    """Manages configuration for embedding and clustering experiments."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path or Path(__file__).parent / "clustering_config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return cast(Dict[str, Any], config or {})
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "embedding_models": {
                "layoutlmv3": {
                    "model_name": "microsoft/layoutlmv3-base",
                    "pooling_methods": ["cls"],
                    "default_pooling": "cls",
                }
            },
            "dimensionality_reduction": {"pca": {"parameters": {"n_components": [20]}}},
            "clustering_algorithms": {"kmeans": {"parameters": {"n_clusters": [5]}}},
            "default_config": {
                "embedding_model": "layoutlmv3",
                "pooling_method": "cls",
                "dimensionality_reduction": "pca",
                "clustering_algorithm": "kmeans",
            },
            "optimization": {"max_combinations": 10, "primary_metric": "silhouette_score"},
        }

    def get_default_configuration(self) -> Dict[str, str]:
        """Get the default configuration for quick runs."""
        default_config: Dict[str, str] = {
            "embedding_model": "layoutlmv3",
            "pooling_method": "cls",
            "dimensionality_reduction": "pca",
            "clustering_algorithm": "kmeans",
        }
        return cast(Dict[str, str], self.config.get("default_config", default_config))

    def get_embedding_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available embedding models and their configurations."""
        embedding_models: Dict[str, Dict[str, Any]] = {
            "layoutlmv3": {
                "model_name": "microsoft/layoutlmv3-base",
                "pooling_methods": ["cls"],
                "default_pooling": "cls",
            }
        }
        return cast(Dict[str, Dict[str, Any]], self.config.get("embedding_models", embedding_models))

    def get_available_pooling_methods(self, model_type: str) -> List[str]:
        """Get available pooling methods for a specific model type."""
        models = self.get_embedding_models()
        if model_type in models:
            default_pooling: List[str] = ["cls"]
            return cast(List[str], models[model_type].get("pooling_methods", default_pooling))
        return ["cls"]

    def generate_configurations(
        self,
        embedding_models: List[str] | None = None,
        pooling_methods: List[str] | None = None,
        reduction_methods: List[str] | None = None,
        clustering_methods: List[str] | None = None,
        max_combinations: int | None = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate all possible configuration combinations.

        Args:
            embedding_models: List of embedding models to test
            pooling_methods: List of pooling methods to test
            reduction_methods: List of dimensionality reduction methods
            clustering_methods: List of clustering algorithms
            max_combinations: Maximum number of combinations to generate

        Yields:
            Configuration dictionaries
        """
        # Use provided lists or generate from config
        if embedding_models is None:
            embedding_models = list(self.get_embedding_models().keys())

        if pooling_methods is None:
            pooling_methods = ["cls", "mean"]

        if reduction_methods is None:
            reduction_methods = list(self.config.get("dimensionality_reduction", {}).keys())

        if clustering_methods is None:
            clustering_methods = list(self.config.get("clustering_algorithms", {}).keys())

        if max_combinations is None:
            max_combinations = self.config.get("optimization", {}).get("max_combinations", 50)

        # Generate parameter combinations for each method
        combinations = []

        for embed_model in embedding_models:
            # Filter pooling methods for this embedding model
            valid_pooling = self.get_available_pooling_methods(embed_model)
            for pooling in pooling_methods:
                if pooling not in valid_pooling:
                    continue

                for reduction in reduction_methods:
                    # Get parameter combinations for reduction method
                    reduction_params = self._get_method_parameters("dimensionality_reduction", reduction)

                    for clustering in clustering_methods:
                        # Get parameter combinations for clustering method
                        clustering_params = self._get_method_parameters("clustering_algorithms", clustering)

                        # Generate all parameter combinations
                        for red_params in reduction_params:
                            for clust_params in clustering_params:
                                config = {
                                    "embedding_model": embed_model,
                                    "pooling_method": pooling,
                                    "dimensionality_reduction": {"method": reduction, **red_params},
                                    "clustering_algorithm": {"method": clustering, **clust_params},
                                }
                                combinations.append(config)

        # Limit combinations and yield
        if len(combinations) > max_combinations:
            logger.info(f"Generated {len(combinations)} combinations, limiting to {max_combinations}")
            combinations = combinations[:max_combinations]
        else:
            logger.info(f"Generated {len(combinations)} configuration combinations")

        for config in combinations:
            yield config

    def _get_method_parameters(self, category: str, method: str) -> List[Dict[str, Any]]:
        """
        Get parameter combinations for a specific method.

        Args:
            category: "dimensionality_reduction" or "clustering_algorithms"
            method: Specific method name

        Returns:
            List of parameter dictionaries
        """
        method_config = self.config.get(category, {}).get(method, {})
        parameters = method_config.get("parameters", {})

        if not parameters:
            return [{}]

        # Generate all combinations of parameter values
        param_names = list(parameters.keys())
        param_values = list(parameters.values())

        # Convert single values to lists
        param_values = [v if isinstance(v, list) else [v] for v in param_values]

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get optimization settings."""
        default_optimization: Dict[str, Any] = {
            "max_combinations": 50,
            "evaluation_metrics": ["silhouette_score", "quality_purity", "balance_score"],
            "primary_metric": "silhouette_score",
        }
        return cast(Dict[str, Any], self.config.get("optimization", default_optimization))

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save experiment results to file.

        Args:
            results: List of experiment results
            output_path: Path to save results
        """
        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj: Any) -> Any:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            elif hasattr(obj, "item"):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj

        serializable_results = convert_numpy_types(results)

        if output_file_path.suffix.lower() == ".yaml":
            with open(output_file_path, "w") as f:
                yaml.safe_dump(serializable_results, f, default_flow_style=False)
        else:
            import json

            with open(output_file_path, "w") as f:
                json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved {len(results)} experiment results to {output_file_path}")


class RecommendationConfigManager:
    """Manages configuration for layout recommendation engine."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize recommendation configuration manager.

        Args:
            config_path: Path to recommendation configuration YAML file
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "recommendation_config.yaml")

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load recommendation configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                config: Dict[str, Any] = yaml.safe_load(f) or {}
            logger.info(f"Loaded recommendation configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load recommendation configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default recommendation configuration if file loading fails."""
        return {
            "openai": {"model": "gpt-4.1-mini", "temperature": 0.7, "max_tokens": 300},
            "training": {
                "clustering_config_path": "config/clustering_config.yaml",
                "max_llm_enhanced_issues": 2,
                "min_effect_size_for_llm_enhancement": 0.5,
            },
            "recommendation": {"max_similar_layouts": 3, "max_quality_issues": 5, "min_significance_level": 0.05},
            "logging": {"level": "INFO", "log_token_usage": True},
        }

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI LLM configuration."""
        default_openai: Dict[str, Any] = self._get_default_config()["openai"]
        return cast(Dict[str, Any], self.config.get("openai", default_openai))

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        default_training: Dict[str, Any] = self._get_default_config()["training"]
        return cast(Dict[str, Any], self.config.get("training", default_training))

    def get_recommendation_config(self) -> Dict[str, Any]:
        """Get recommendation behavior configuration."""
        default_recommendation: Dict[str, Any] = self._get_default_config()["recommendation"]
        return cast(Dict[str, Any], self.config.get("recommendation", default_recommendation))

    def get_prompts_config(self) -> Dict[str, Any]:
        """Get prompts configuration."""
        default_prompts: Dict[str, Any] = self._get_default_config()["prompts"]
        return cast(Dict[str, Any], self.config.get("prompts", default_prompts))

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        default_logging: Dict[str, Any] = self._get_default_config()["logging"]
        return cast(Dict[str, Any], self.config.get("logging", default_logging))

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance settings."""
        default_performance: Dict[str, Any] = self._get_default_config()["performance"]
        return cast(Dict[str, Any], self.config.get("performance", default_performance))

    def get_clustering_config_path(self) -> str:
        """Get path to clustering configuration file."""
        training_config = self.get_training_config()
        return cast(str, training_config.get("clustering_config_path", "config/clustering_config.yaml"))

    def validate_config(self) -> bool:
        """
        Validate the configuration for completeness and correctness.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ["openai", "training", "recommendation"]
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False

            # Validate OpenAI config
            openai_config = self.get_openai_config()
            if not openai_config.get("model"):
                logger.error("OpenAI model not specified in configuration")
                return False

            # Validate temperature range
            temp = openai_config.get("temperature", 0.7)
            if not 0.0 <= temp <= 2.0:
                logger.error(f"Invalid temperature value: {temp}. Must be between 0.0 and 2.0")
                return False

            # Validate max_tokens
            max_tokens = openai_config.get("max_tokens", 300)
            if not isinstance(max_tokens, int) or max_tokens < 1:
                logger.error(f"Invalid max_tokens value: {max_tokens}. Must be positive integer")
                return False

            # Check clustering config path exists
            clustering_path = Path(self.get_clustering_config_path())
            if not clustering_path.exists():
                logger.warning(f"Clustering config file not found: {clustering_path}")

            logger.info("Recommendation configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_effective_config(self, override_env: bool = True) -> Dict[str, Any]:
        """
        Get effective configuration with environment variable overrides.

        Args:
            override_env: Whether to apply environment variable overrides

        Returns:
            Effective configuration dictionary
        """
        import os

        config = self.config.copy()

        if override_env:
            # Override OpenAI settings from environment
            openai_config = config.get("openai", {})

            if os.getenv("OPENAI_MODEL"):
                openai_config["model"] = os.getenv("OPENAI_MODEL")

            if os.getenv("OPENAI_TEMPERATURE"):
                try:
                    temp = os.getenv("OPENAI_TEMPERATURE")
                    openai_config["temperature"] = float(temp) if temp else 0.7
                except ValueError:
                    logger.warning("Invalid OPENAI_TEMPERATURE environment variable")

            if os.getenv("OPENAI_MAX_TOKENS"):
                try:
                    max_tokens = os.getenv("OPENAI_MAX_TOKENS")
                    openai_config["max_tokens"] = int(max_tokens) if max_tokens else 300
                except ValueError:
                    logger.warning("Invalid OPENAI_MAX_TOKENS environment variable")

            config["openai"] = openai_config

        return config
