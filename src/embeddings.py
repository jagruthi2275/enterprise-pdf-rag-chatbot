"""HuggingFace Inference API embeddings (no local PyTorch required)."""

import os
import logging
from typing import List
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Calls HuggingFace Inference API to embed text — no local model download."""

    def __init__(self):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN environment variable is not set.")
        self.client = InferenceClient(token=token)
        logger.info(f"HuggingFace Inference API ready: {MODEL_NAME}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings via HF Inference API."""
        try:
            vectors = self.client.feature_extraction(texts, model=MODEL_NAME)
            if hasattr(vectors, "tolist"):
                return vectors.tolist()
            return [list(v) for v in vectors]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
