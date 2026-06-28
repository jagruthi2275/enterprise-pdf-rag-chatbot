"""Retriever: embeds the query and searches FAISS for top-K chunks with similarity scores."""

import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves the top-K most relevant chunks for a query, including similarity scores."""

    def __init__(self, embedder, vector_store, top_k: int = 5):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.top_k        = top_k

    def retrieve(self, query: str) -> List[Dict]:
        """
        Embed the query and return top-K chunks with metadata and similarity scores.
        Each returned dict contains: text, source, page, chunk_id, similarity_score.
        """
        query_vector = self.embedder.embed([query])[0]
        query_vector = np.array([query_vector], dtype="float32")

        # FAISS search_with_scores returns (distances, indices)
        # For IndexFlatL2, distance = L2 distance (lower = more similar)
        # We convert to a 0-1 similarity score: score = 1 / (1 + distance)
        distances, indices = self.vector_store.index.search(query_vector, self.top_k)

        chunks = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            chunk = dict(self.vector_store.metadata[idx])  # copy to avoid mutation
            similarity_score = float(1 / (1 + dist))     # normalise to 0–1
            chunk["similarity_score"] = round(similarity_score, 4)
            chunk["chunk_id"]         = int(idx)
            chunk["rank"]             = rank + 1
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks for query (top_k={self.top_k})")
        return chunks