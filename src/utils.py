"""Logging configuration and shared utilities."""

import logging
import os
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to stdout."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    os.makedirs("vectorstore", exist_ok=True)
    os.makedirs("data", exist_ok=True)
