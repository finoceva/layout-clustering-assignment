"""
Base embedding extractor interface.
"""

from abc import ABC, abstractmethod
from typing import List, Literal

import numpy as np

from core.schemas import Layout  # type: ignore

PoolingMethod = Literal["cls", "mean", "max"]


class BaseEmbeddingExtractor(ABC):
    """Abstract base class for layout embedding extractors."""

    def __init__(self, model_name: str, pooling_method: PoolingMethod = "cls"):
        """
        Initialize the embedding extractor.

        Args:
            model_name: Name/path of the model to use
            pooling_method: How to pool token embeddings ("cls", "mean", "max")
        """
        self.model_name = model_name
        self.pooling_method = pooling_method

    @abstractmethod
    def extract_single_embedding(self, layout: Layout) -> np.ndarray:
        """
        Extract embedding for a single layout.

        Args:
            layout: Layout object to extract embedding from

        Returns:
            Embedding vector as numpy array
        """
        pass

    def extract_embeddings_batch(self, layouts: List[Layout]) -> np.ndarray:
        """
        Extract embeddings for multiple layouts.

        Args:
            layouts: List of Layout objects

        Returns:
            Array of embeddings with shape (n_layouts, embedding_dim)
        """
        embeddings = []
        for layout in layouts:
            embedding = self.extract_single_embedding(layout)
            embeddings.append(embedding)

        return np.array(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of embeddings produced by this extractor."""
        # This should be overridden by subclasses with actual dimensions
        return 768  # Common transformer dimension
