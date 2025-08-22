"""
LayoutLMv1 embedding extractor.
"""

from typing import List, Tuple

import numpy as np
import torch  # type: ignore
from transformers import LayoutLMModel, LayoutLMTokenizer  # type: ignore

from core.schemas import Layout  # type: ignore
from embeddings.base import BaseEmbeddingExtractor, PoolingMethod  # type: ignore
from utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)


class LayoutLMv1Extractor(BaseEmbeddingExtractor):
    """LayoutLMv1-based embedding extractor."""

    def __init__(self, model_name: str = "microsoft/layoutlm-base-uncased", pooling_method: PoolingMethod = "cls"):
        """Initialize LayoutLMv1 extractor."""
        super().__init__(model_name, pooling_method)

        logger.info(f"Loading LayoutLMv1 model: {model_name}")
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
        self.model = LayoutLMModel.from_pretrained(model_name)
        self.model.eval()

        logger.info(f"LayoutLMv1 model loaded with pooling: {pooling_method}")

    def _convert_bbox(
        self, x: float, y: float, width: float, height: float, canvas_width: int, canvas_height: int
    ) -> List[int]:
        """
        Convert element bbox to LayoutLMv1 format [x0, y0, x1, y1] in range [0, 1000].

        Args:
            x, y, width, height: Element position and size
            canvas_width, canvas_height: Canvas dimensions

        Returns:
            Normalized bbox coordinates [x0, y0, x1, y1]
        """
        # Normalize to [0, 1] then scale to [0, 1000]
        x0 = int((x / canvas_width) * 1000)
        y0 = int((y / canvas_height) * 1000)
        x1 = int(((x + width) / canvas_width) * 1000)
        y1 = int(((y + height) / canvas_height) * 1000)

        # Clamp to valid range
        x0 = max(0, min(1000, x0))
        y0 = max(0, min(1000, y0))
        x1 = max(0, min(1000, x1))
        y1 = max(0, min(1000, y1))

        return [x0, y0, x1, y1]

    def _prepare_inputs(self, layout: Layout) -> Tuple[List[int], List[List[int]]]:
        """
        Prepare LayoutLMv1 inputs from layout.

        Args:
            layout: Layout object

        Returns:
            Tuple of (input_ids, bbox) for LayoutLMv1
        """
        # Create text sequence from element classes
        element_texts = [elem.element_class for elem in layout.elements]
        text = " ".join(element_texts)

        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Create bounding boxes for each token
        bbox = []
        token_idx = 0

        # [CLS] token gets [0,0,0,0]
        bbox.append([0, 0, 0, 0])
        token_idx += 1

        # Map element bboxes to their tokens
        element_idx = 0
        for token in tokens[1:-1]:  # Skip [CLS] and [SEP]
            if element_idx < len(layout.elements):
                elem = layout.elements[element_idx]
                elem_bbox = self._convert_bbox(elem.x, elem.y, elem.width, elem.height, layout.width, layout.height)
                bbox.append(elem_bbox)

                # Move to next element when we finish tokenizing current element class
                if token_idx >= len(element_texts[element_idx]):
                    element_idx += 1
            else:
                bbox.append([0, 0, 0, 0])
            token_idx += 1

        # [SEP] token gets [0,0,0,0]
        bbox.append([0, 0, 0, 0])

        # Ensure bbox list matches input_ids length
        while len(bbox) < len(input_ids):
            bbox.append([0, 0, 0, 0])

        return input_ids, bbox

    def extract_single_embedding(self, layout: Layout) -> np.ndarray:
        """Extract LayoutLMv1 embedding for a single layout."""
        try:
            input_ids, bbox = self._prepare_inputs(layout)

            # Convert to tensors
            input_ids_tensor = torch.tensor([input_ids])
            bbox_tensor = torch.tensor([bbox])
            attention_mask = torch.ones_like(input_ids_tensor)

            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids_tensor, bbox=bbox_tensor, attention_mask=attention_mask)

                hidden_states = outputs.last_hidden_state[0]  # Remove batch dim

                # Apply pooling
                if self.pooling_method == "cls":
                    embedding = hidden_states[0].cpu().numpy()  # [CLS] token
                elif self.pooling_method == "mean":
                    embedding = hidden_states.mean(dim=0).cpu().numpy()
                elif self.pooling_method == "max":
                    embedding = hidden_states.max(dim=0)[0].cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling_method}")

                return embedding

        except Exception as e:
            logger.error(f"Error extracting LayoutLMv1 embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """LayoutLMv1 embedding dimension."""
        return 768
