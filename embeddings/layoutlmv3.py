"""
LayoutLMv3 Embedder for structural clustering.
Simplified implementation for generating layout embeddings.
"""

from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from transformers import LayoutLMv3Model, LayoutLMv3Processor

from core.schemas import Layout


class LayoutLMv3Embedder:
    """Simplified LayoutLMv3 embedder for layout analysis."""
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        """Initialize the LayoutLMv3 embedder."""
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = LayoutLMv3Processor.from_pretrained(self.model_name)
        self.model = LayoutLMv3Model.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _layout_to_layoutlmv3_format(self, layout: Layout) -> Dict[str, Any]:
        """Convert Layout object to LayoutLMv3 input format."""
        # Create synthetic text for each element (element class names)
        words = [elem.element_class for elem in layout.elements]
        
        # If no elements, create a dummy
        if not words:
            words = ["empty"]
        
        # Create bounding boxes (normalized to 0-1000 scale)
        boxes = []
        for elem in layout.elements:
            box = [
                int((elem.x / layout.width) * 1000),
                int((elem.y / layout.height) * 1000),
                int(((elem.x + elem.width) / layout.width) * 1000),
                int(((elem.y + elem.height) / layout.height) * 1000)
            ]
            # Ensure valid box coordinates
            box = [max(0, min(1000, coord)) for coord in box]
            boxes.append(box)
        
        # Handle empty layout
        if not boxes:
            boxes = [[0, 0, 100, 100]]  # Dummy box
        
        return {
            "words": words,
            "boxes": boxes
        }
    
    def extract_embedding(self, layout: Layout) -> np.ndarray:
        """Extract embedding for a single layout."""
        try:
            # Convert to LayoutLMv3 format
            layout_data = self._layout_to_layoutlmv3_format(layout)
            
            # Process with LayoutLMv3 processor
            # Create a dummy image (LayoutLMv3 expects image input)
            from PIL import Image
            dummy_image = Image.new('RGB', (layout.width, layout.height), color='white')
            
            encoding = self.processor(
                dummy_image,
                layout_data["words"],
                boxes=layout_data["boxes"],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**encoding)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
                embedding = embeddings[0].cpu().numpy()  # Get first (and only) item
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding for layout {layout.id}: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.embedding_dim)
    
    def extract_embeddings_batch(self, layouts: List[Layout], batch_size: int = 4) -> np.ndarray:
        """Extract embeddings for a batch of layouts."""
        logger.info(f"Extracting LayoutLMv3 embeddings for {len(layouts)} layouts")
        
        all_embeddings = []
        
        for i in range(0, len(layouts), batch_size):
            batch_layouts = layouts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(layouts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for layout in batch_layouts:
                embedding = self.extract_embedding(layout)
                batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(all_embeddings)
        logger.info(f"Extracted embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array


def test_embedder() -> None:
    """Test the LayoutLMv3 embedder."""
    logger.info("Testing LayoutLMv3 Embedder...")
    
    # Create a dummy layout for testing
    from core.schemas import Element, Layout
    
    test_elements = [
        Element(element_class="headline", x=100, y=50, width=300, height=50),
        Element(element_class="body", x=100, y=120, width=250, height=80),
        Element(element_class="image", x=400, y=50, width=200, height=150),
    ]
    
    test_layout = Layout(
        id="test_layout",
        width=600,
        height=400,
        group_id="test",
        elements=test_elements,
        quality="pass"
    )
    
    # Test embedder
    embedder = LayoutLMv3Embedder()
    embedding = embedder.extract_embedding(test_layout)
    
    logger.info("✅ Test successful!")
    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Embedding stats: mean={embedding.mean():.3f}, std={embedding.std():.3f}")


if __name__ == "__main__":
    test_embedder()
