"""PDF loading and text extraction with page tracking."""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads one or multiple PDFs and extracts text per page."""

    def load(self, file_paths: List[str], original_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract text from PDFs.

        Args:
            file_paths:     List of temp file paths on disk.
            original_names: Optional list of original filenames (same order as file_paths).

        Returns:
            List of dicts: {"text": str, "page": int, "source": str}
        """
        docs = []
        for i, path in enumerate(file_paths):
            try:
                reader = PdfReader(path)
                source = original_names[i] if original_names else Path(path).name
                for j, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append({"text": text, "page": j, "source": source})
                logger.info(f"Loaded {len(reader.pages)} pages from {source}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                raise
        return docs