"""Query rewriter: reformulates ambiguous follow-up questions into self-contained search queries."""

import os
import logging
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are a search query optimizer for a document retrieval system.

Your task: rewrite the user's latest question into a clear, self-contained search query
that can be used to retrieve relevant document chunks from a vector database.

Rules:
- If the question is already clear and self-contained, return it unchanged.
- If the question is a follow-up (e.g. "simplify that", "give an example", "how does it differ"),
  use the conversation history to infer the actual topic and rewrite accordingly.
- Output ONLY the rewritten query. No explanation, no preamble, no quotes.

Conversation History:
{history}

Latest Question: {question}

Rewritten Query:"""


class QueryRewriter:
    """Uses Groq LLM to rewrite ambiguous queries using conversation history."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            temperature=0.0,
            max_tokens=128,
        )

    def rewrite(self, question: str, history: list) -> str:
        if not history:
            return question

        history_text = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 300:
                content = content[:300] + "…"
            history_text += f"{role}: {content}\n"

        prompt = REWRITE_PROMPT.format(history=history_text.strip(), question=question)

        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            if rewritten and rewritten != question:
                logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
            return rewritten if rewritten else question
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            return question