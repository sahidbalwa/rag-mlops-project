"""
Module for formatting retrieved chunks into a clean context string.
=== FILE: retrieval/context_builder.py ===
"""

import logging
from typing import List, Tuple
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builds the final text context to be injected into LLM prompts.
    """

    @staticmethod
    def build_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
        """
        Formats documents sequentially, including source information.

        Args:
            docs_with_scores (List[Tuple[Document, float]]): The documents to format.

        Returns:
            str: The formatted context string.
        """
        logger.info(f"Building context from {len(docs_with_scores)} documents.")
        
        if not docs_with_scores:
            return "No relevant context found."

        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "Unknown")
            page_content = doc.page_content.strip()
            # We append numbering, text, and metadata
            formatted_doc = f"--- Document [{i}] ---\nSource: {source}\nRelevance Score: {score:.4f}\nContent:\n{page_content}\n"
            context_parts.append(formatted_doc)

        return "\n".join(context_parts)
