"""Orchestrates the full RAG pipeline."""

import logging
from typing import List, Dict
from src.pdf_loader import PDFLoader
from src.chunking import DocumentChunker
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retrieval import Retriever
from src.llm import LLMClient
from src.prompts import build_prompt

logger = logging.getLogger(__name__)


class RAGChain:
    """End-to-end RAG pipeline: ingest PDFs → answer questions."""

    def __init__(self):
        self.loader       = PDFLoader()
        self.chunker      = DocumentChunker(chunk_size=500, chunk_overlap=50)
        self.embedder     = EmbeddingModel()
        self.vector_store = VectorStore()
        self.llm          = LLMClient()
        self._ready       = False

    def ingest(self, file_paths: List[str]) -> int:
        """
        Full ingestion pipeline: load → chunk → embed → index.
        Returns the number of chunks indexed.
        """
        docs       = self.loader.load(file_paths)
        chunks     = self.chunker.chunk(docs)
        texts      = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.build(chunks, embeddings)
        self.vector_store.save()
        self._ready = True
        logger.info(f"Ingestion complete: {len(chunks)} chunks indexed")
        return len(chunks)

    def ask(self, question: str) -> Dict:
        """
        Answer a question using retrieved context.
        Returns {"answer": str, "sources": List[Dict]}
        """
        if not self._ready:
            loaded = self.vector_store.load()
            if not loaded:
                raise RuntimeError("No documents have been processed yet.")
            self._ready = True

        retriever = Retriever(self.embedder, self.vector_store, top_k=5)
        chunks    = retriever.retrieve(question)
        prompt    = build_prompt(chunks, question)
        answer    = self.llm.invoke(prompt)

        sources = chunks
        return {"answer": answer, "sources": sources}