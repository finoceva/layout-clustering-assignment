"""
Factory for creating embedding extractors.
"""

from typing import Dict, Type

from embeddings.base import BaseEmbeddingExtractor, PoolingMethod  # type: ignore
from embeddings.layoutlmv1 import LayoutLMv1Extractor  # type: ignore
from embeddings.layoutlmv3 import LayoutLMv3Extractor  # type: ignore
from utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)

# Registry of available embedding extractors
EMBEDDING_EXTRACTORS: Dict[str, Type[BaseEmbeddingExtractor]] = {
    "layoutlmv1": LayoutLMv1Extractor,
    "layoutlmv3": LayoutLMv3Extractor,
}


def create_embedding_extractor(
    model_type: str, model_name: str = None, pooling_method: PoolingMethod = "cls"
) -> BaseEmbeddingExtractor:
    """
    Create an embedding extractor instance.

    Args:
        model_type: Type of model ("layoutlmv1", "layoutlmv3")
        model_name: Specific model name/path (optional, uses defaults)
        pooling_method: Pooling method for embeddings

    Returns:
        Configured embedding extractor instance

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in EMBEDDING_EXTRACTORS:
        available = ", ".join(EMBEDDING_EXTRACTORS.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    extractor_class = EMBEDDING_EXTRACTORS[model_type]

    # Use default model names if not specified
    if model_name is None:
        default_models = {"layoutlmv1": "microsoft/layoutlm-base-uncased", "layoutlmv3": "microsoft/layoutlmv3-base"}
        model_name = default_models[model_type]

    logger.info(f"Creating {model_type} extractor with model: {model_name}, pooling: {pooling_method}")

    return extractor_class(model_name=model_name, pooling_method=pooling_method)


def list_available_extractors() -> Dict[str, str]:
    """
    List all available embedding extractors.

    Returns:
        Dictionary mapping model types to descriptions
    """
    descriptions = {
        "layoutlmv1": "LayoutLM v1 - Original layout understanding model",
        "layoutlmv3": "LayoutLM v3 - Advanced multimodal layout model",
    }

    return {
        model_type: descriptions.get(model_type, "No description available")
        for model_type in EMBEDDING_EXTRACTORS.keys()
    }
