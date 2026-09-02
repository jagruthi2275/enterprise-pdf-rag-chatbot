"""Groq LLM client (Llama-3.3-70b-versatile)."""

import os
import logging
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-120b"


class LLMClient:
    """Thin wrapper around Groq's ChatGroq for Llama3-8b."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        self.llm = ChatGroq(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0.2,
        )
        logger.info(f"LLM client ready: {MODEL_NAME}")

    def invoke(self, prompt: str) -> str:
        """Send a prompt and return the response text."""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
