"""
Modular clustering components for flexible configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from umap import UMAP  # type: ignore

from utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)


class DimensionalityReducer(ABC):
    """Abstract base for dimensionality reduction."""

    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform embeddings."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        pass


class PCAReducer(DimensionalityReducer):
    """PCA dimensionality reduction."""

    def __init__(self, n_components: int = 20, random_state: int = 42):
        """Initialize PCA reducer."""
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = PCA(n_components=n_components, random_state=random_state)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA reduction."""
        # Validate n_components doesn't exceed data dimensions
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        if self.n_components > max_components:
            logger.warning(
                f"Reducing PCA components from {self.n_components} to {max_components} "
                f"due to data constraints (n_samples={embeddings.shape[0]}, "
                f"n_features={embeddings.shape[1]})"
            )
            # Create a new PCA with valid n_components
            from sklearn.decomposition import PCA  # type: ignore

            valid_reducer = PCA(n_components=max_components, random_state=self.random_state)
            return valid_reducer.fit_transform(embeddings)

        return self.reducer.fit_transform(embeddings)

    def get_params(self) -> Dict[str, Any]:
        """Get PCA parameters."""
        return {"method": "pca", "n_components": self.n_components, "random_state": self.random_state}


class UMAPReducer(DimensionalityReducer):
    """UMAP dimensionality reduction."""

    def __init__(self, n_components: int = 20, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        """Initialize UMAP reducer."""
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.reducer = UMAP(
            n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
        )

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP reduction."""
        return self.reducer.fit_transform(embeddings)

    def get_params(self) -> Dict[str, Any]:
        """Get UMAP parameters."""
        return {
            "method": "umap",
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "random_state": self.random_state,
        }


class ClusteringAlgorithm(ABC):
    """Abstract base for clustering algorithms."""

    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        pass


class KMeansClusterer(ClusteringAlgorithm):
    """K-Means clustering."""

    def __init__(self, n_clusters: int = 5, random_state: int = 42, n_init: int = 10):
        """Initialize K-Means clusterer."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply K-Means clustering."""
        return self.clusterer.fit_predict(embeddings)

    def get_params(self) -> Dict[str, Any]:
        """Get K-Means parameters."""
        return {
            "method": "kmeans",
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "n_init": self.n_init,
        }


class HDBSCANClusterer(ClusteringAlgorithm):
    """HDBSCAN clustering."""

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3, cluster_selection_epsilon: float = 0.0):
        """Initialize HDBSCAN clusterer."""
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply HDBSCAN clustering."""
        return self.clusterer.fit_predict(embeddings)

    def get_params(self) -> Dict[str, Any]:
        """Get HDBSCAN parameters."""
        return {
            "method": "hdbscan",
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
        }


def create_dimensionality_reducer(method: str, **kwargs) -> DimensionalityReducer:
    """
    Create a dimensionality reducer.

    Args:
        method: "pca" or "umap"
        **kwargs: Method-specific parameters

    Returns:
        Configured dimensionality reducer
    """
    if method == "pca":
        return PCAReducer(**kwargs)
    elif method == "umap":
        return UMAPReducer(**kwargs)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}. Available: pca, umap")


def create_clustering_algorithm(method: str, **kwargs) -> ClusteringAlgorithm:
    """
    Create a clustering algorithm.

    Args:
        method: "kmeans" or "hdbscan"
        **kwargs: Method-specific parameters

    Returns:
        Configured clustering algorithm
    """
    if method == "kmeans":
        return KMeansClusterer(**kwargs)
    elif method == "hdbscan":
        return HDBSCANClusterer(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def evaluate_clustering(
    embeddings: np.ndarray, labels: np.ndarray, quality_labels: List[str], weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Evaluate clustering results.

    Args:
        embeddings: Reduced embeddings used for clustering
        labels: Cluster labels
        quality_labels: Ground truth quality labels ("pass"/"fail")
        weights: Optional weights for combining metrics. If None, uses default weights.

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import silhouette_score  # type: ignore

    from utils.evaluation import calculate_quality_purity  # type: ignore

    results = {}

    # Silhouette score (only if we have multiple clusters)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        try:
            # Filter out noise points (-1) for silhouette calculation
            valid_mask = labels != -1
            if valid_mask.sum() > 1:
                results["silhouette_score"] = silhouette_score(embeddings[valid_mask], labels[valid_mask])
            else:
                results["silhouette_score"] = -1.0
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            results["silhouette_score"] = -1.0
    else:
        results["silhouette_score"] = -1.0

    # Quality purity
    results["quality_purity"] = calculate_quality_purity(labels, quality_labels)

    # Balance score (preference for balanced cluster sizes)
    cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]
    if cluster_sizes:
        balance_score = 1.0 - np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-8)
        # Penalty for very small clusters (< 5 layouts)
        small_cluster_penalty = sum(1 for size in cluster_sizes if size < 5) * 0.1
        balance_score = max(0, balance_score - small_cluster_penalty)
        results["balance_score"] = balance_score
    else:
        results["balance_score"] = 0.0

    # Combined score with configurable weights
    if weights is None:
        # Default weights
        weights = {"silhouette_score": 0.3, "quality_purity": 0.4, "balance_score": 0.3}

    results["combined_score"] = (
        results["silhouette_score"] * weights.get("silhouette_score", 0.3)
        + results["quality_purity"] * weights.get("quality_purity", 0.4)
        + results["balance_score"] * weights.get("balance_score", 0.3)
    )

    # Additional metrics
    results["n_clusters"] = n_clusters
    results["noise_ratio"] = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0.0

    return results
