"""FAISS vector store: build, save, load, and search."""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH = "vectorstore/faiss.index"
META_PATH  = "vectorstore/metadata.pkl"


class VectorStore:
    """Manages FAISS index and chunk metadata."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []

    def build(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """Build FAISS index from chunk embeddings."""
        vectors = np.array(embeddings, dtype="float32")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)
        self.metadata = chunks
        logger.info(f"Built FAISS index with {len(chunks)} vectors (dim={dim})")

    def save(self) -> None:
        """Persist index and metadata to disk."""
        Path("vectorstore").mkdir(exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Vector store saved to disk")

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded vector store ({self.index.ntotal} vectors)")
            return True
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            return False

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Return top_k closest chunks with distances."""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Process documents first.")
        vec = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self.metadata[idx], float(dist)))
        return results
