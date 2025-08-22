"""
LayoutLMv3 embedding extraction for layout clustering.
"""


import numpy as np
import torch  # type: ignore
from transformers import LayoutLMv3Model, LayoutLMv3Processor  # type: ignore

from core.schemas import Layout  # type: ignore
from embeddings.base import BaseEmbeddingExtractor, PoolingMethod  # type: ignore
from utils.logger import get_logger  # type: ignore

logger = get_logger(__name__)


class LayoutLMv3Extractor(BaseEmbeddingExtractor):
    """LayoutLMv3-based embedding extractor for layout documents."""

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", pooling_method: PoolingMethod = "cls"):
        """
        Initialize LayoutLMv3 extractor.

        Args:
            model_name: HuggingFace model name or path
            pooling_method: How to pool embeddings ("cls", "mean", "max")
        """
        super().__init__(model_name, pooling_method)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading LayoutLMv3 model: {model_name}")
        logger.info(f"Using device: {self.device}")

        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"LayoutLMv3 model loaded with pooling: {pooling_method}")

    def _prepare_layout_input(self, layout: Layout) -> dict:
        """
        Prepare layout data for LayoutLMv3 input.

        Args:
            layout: Layout object

        Returns:
            Dictionary with text, boxes, and image for processor
        """
        # Generate synthetic text from element classes
        texts = []
        boxes = []

        for elem in layout.elements:
            # Use original coordinates (preserve structural relationships)
            x0 = elem.x
            y0 = elem.y
            x1 = elem.x + elem.width
            y1 = elem.y + elem.height

            # Normalize to [0, 1000] range for LayoutLMv3 (preserve proportions)
            norm_x0 = (x0 / layout.width) * 1000
            norm_y0 = (y0 / layout.height) * 1000
            norm_x1 = (x1 / layout.width) * 1000
            norm_y1 = (y1 / layout.height) * 1000

            # Conservative clamping: Only fix out-of-bounds values, preserve structure
            norm_x0 = max(0, min(norm_x0, 1000))
            norm_y0 = max(0, min(norm_y0, 1000))
            norm_x1 = max(0, min(norm_x1, 1000))
            norm_y1 = max(0, min(norm_y1, 1000))

            # Ensure valid box (x0 < x1, y0 < y1)
            if norm_x0 >= norm_x1:
                # Preserve center, adjust bounds
                center_x = (norm_x0 + norm_x1) / 2
                norm_x0 = max(0, center_x - 1)
                norm_x1 = min(1000, center_x + 1)

            if norm_y0 >= norm_y1:
                # Preserve center, adjust bounds
                center_y = (norm_y0 + norm_y1) / 2
                norm_y0 = max(0, center_y - 1)
                norm_y1 = min(1000, center_y + 1)

            # Convert to integers for LayoutLMv3
            norm_x0 = int(norm_x0)
            norm_y0 = int(norm_y0)
            norm_x1 = int(norm_x1)
            norm_y1 = int(norm_y1)

            # Final validation: Ensure we have a valid box
            if norm_x0 >= norm_x1 or norm_y0 >= norm_y1:
                # Last resort: create minimal valid box
                norm_x1 = norm_x0 + 1
                norm_y1 = norm_y0 + 1

            # Use element class as text placeholder
            texts.append(elem.element_class)
            boxes.append([norm_x0, norm_y0, norm_x1, norm_y1])

        # Create synthetic text
        text = " ".join(texts)

        # Create dummy image (LayoutLMv3 expects image input)
        image = torch.zeros(3, layout.height, layout.width, dtype=torch.uint8)

        return {"text": text, "boxes": boxes, "image": image}

    def extract_single_embedding(self, layout: Layout) -> np.ndarray:
        """Extract LayoutLMv3 embedding for a single layout."""
        try:
            # Prepare input
            layout_input = self._prepare_layout_input(layout)

            # Process with LayoutLMv3 processor
            # LayoutLMv3 expects words (list of strings) not a single text string
            words = layout_input["text"].split()

            encoding = self.processor(
                layout_input["image"],
                words,  # Use word list instead of full text
                boxes=layout_input["boxes"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            )

            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**encoding)
                hidden_states = outputs.last_hidden_state[0]  # Remove batch dimension

                # Apply pooling
                if self.pooling_method == "cls":
                    embedding = hidden_states[0].cpu().numpy()  # [CLS] token
                elif self.pooling_method == "mean":
                    mask = encoding["attention_mask"][0].cpu()
                    masked_embeddings = hidden_states.cpu() * mask.unsqueeze(-1)
                    embedding = masked_embeddings.sum(dim=0) / mask.sum()
                    embedding = embedding.numpy()
                elif self.pooling_method == "max":
                    embedding = hidden_states.cpu().max(dim=0)[0].numpy()
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling_method}")

                return embedding

        except Exception as e:
            logger.error(f"Error extracting LayoutLMv3 embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """LayoutLMv3 embedding dimension."""
        return 768
