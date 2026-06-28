"""Split document pages into overlapping chunks."""

import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits raw page text into smaller chunks for embedding."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, docs: List[Dict]) -> List[Dict]:
        """
        Split each page into chunks, preserving metadata.

        Returns:
            List of dicts: {"text": str, "page": int, "source": str}
        """
        chunks = []
        for doc in docs:
            splits = self.splitter.split_text(doc["text"])
            for split in splits:
                chunks.append({
                    "text": split,
                    "page": doc["page"],
                    "source": doc["source"],
                })
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} pages")
        return chunks
